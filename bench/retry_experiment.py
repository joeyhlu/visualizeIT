"""Quality-first retry on the exact same 60 FPS windows, without label updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .vision import cv2
from .glb_model import read_glb
from .show3d_experiment import initialize,reference_pose,evaluation_truth,enrich_report
from .surface_tracker import SurfaceTracker,SurfaceRays,geometry_corners,follow,projected
from .performance_experiment import compact_bank
from .video_experiment import scorer,projection
from .hd_experiment import tracking_input
from .storage import write_artifact
from .ycb_assets import ROOT


def onboarding(capture,root,report,mesh,calibration,height):
    start=report['sourceStart'];cache=root/f'{report["name"]}-onboarding-{height}-v1.npz'
    poses_path=next(p for p in report['sourceHashes'] if p.endswith('object_pose.json'))
    # Restrict all pose data passed to onboarding to the initial ten frames.
    records=json.loads((root/poses_path).read_text())
    poses={i:reference_pose(records[str(i)],calibration,i) for i in range(start,start+10)}
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            return tuple(data[k].copy() for k in ('reference','descriptors','gray','pixels','points')),(poses[start+9]),json.loads(str(data['diagnostics']))
    refs=initialize(capture,start,poses,mesh,calibration,height)
    capture.set(cv2.CAP_PROP_POS_FRAMES,start);images=[];k=None
    for _ in range(10):
        ok,bgr=capture.read()
        if not ok:raise ValueError('Onboarding video incomplete.')
        working,k=tracking_input(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),calibration,height);images.append(working)
    rays=SurfaceRays(mesh);pixels,points=geometry_corners(images[-1],rays,k,poses[start+9],320)
    original=pixels.copy();tracked=pixels.copy();consistent=np.ones(len(pixels),int);alive=np.ones(len(pixels),bool)
    # A corner seen on a moving finger must not automatically become a 3D
    # surface reference. Require multi-frame object-motion consistency.
    for i in range(8,-1,-1):
        current,good=follow(images[i+1],images[i],tracked)
        predicted=projected(points,*poses[start+i],k)
        consistent+=alive&good&(np.linalg.norm(predicted-current,axis=1)<=2.5*height/360)
        alive&=good;tracked=current
    accepted=consistent>=7
    pixels=original[accepted];points=points[accepted]
    # ORB references still provide independent descriptor-based relocalization.
    reference,descriptors=compact_bank(refs[0],refs[1])
    diagnostics=dict(cornersConsidered=len(original),motionConsistentCorners=int(accepted.sum()),
        requiredConsistentOnboardingViews=7,referenceDescriptors=len(reference),
        scope='Geometry-consistent corners in first ten frames; not a foreground/hand segmentation.')
    if len(points)<6:
        # Some reference windows have noisy automatic pose labels. Keep this
        # limitation explicit instead of quietly weakening the consistency test.
        pixels,points=refs[3:]
        diagnostics['fallback']='Last-onboarding ORB features; multi-view consistency insufficient.'
    if len(points)<6:raise ValueError('Not enough onboarding surface features.')
    result=(reference,descriptors,images[-1],pixels,points)
    np.savez_compressed(cache,reference=reference,descriptors=descriptors,gray=images[-1],pixels=pixels,points=points,diagnostics=json.dumps(diagnostics))
    return result,poses[start+9],diagnostics


def run(root,source,output,height=720,objects=None,contour=False,mask=False,synthetic=False,learned=False):
    cv2.setNumThreads(1);output.mkdir(parents=True,exist_ok=True)
    originals=json.loads((source/'report.json').read_text())['objects'];reports=[]
    for original in originals:
        if objects and original['name'] not in objects:continue
        alias=original['name'];report=json.loads(json.dumps(original));report['trackingHeight']=height
        report['renderScope']='Offline saved poses; recovery player can clip designs with experimental predicted foreground. No PBR or validated hand-occlusion accuracy.'
        # Verify the same sources/window. Evaluation labels never enter update.
        for name,expected in report['sourceHashes'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source hash mismatch.')
        prefix='scenes/'+report['scene']+'/'
        calibration=json.loads((root/prefix/'camera_calibration/headset0.json').read_text())
        calibration['cam_K']=report['cameraCalibration']
        mesh=read_glb(root/f'models/obj_{report["modelId"]:06d}.glb',alias)
        capture=cv2.VideoCapture(str(root/prefix/'headset0.mp4'))
        refs,pose,diagnostics=onboarding(capture,root,report,mesh,calibration,height)
        if synthetic:
            from .model_references import build
            model_points,model_descriptors=build(root,mesh,root/f'models/obj_{report["modelId"]:06d}.glb')
            refs=(np.concatenate((refs[0],model_points)),np.concatenate((refs[1],model_descriptors)),*refs[2:])
            diagnostics['modelOnlyReferenceFeatures']=len(model_points)
        print(alias,diagnostics,flush=True)
        if contour:
            from .contour_tracker import ContourTracker
        foreground=None
        if learned:
            if height!=720:raise ValueError('Recorded learned masks require 720-pixel tracking height.')
            mask_report=json.loads((source/'learned-mask'/f'{alias}-full.json').read_text())
            masks={f['frameId']:f for f in mask_report['frames']}
            def foreground(gray,frame_id):
                record=masks[frame_id];paths=record.get('visibleMaskContours')
                if paths is None:return None,dict(maskReason=record.get('maskReason'),maskScope=record['maskScope'])
                mask_image=np.zeros(gray.shape,np.uint8)
                # Draw exterior paths before holes; preserve foreground islands.
                for hole in (False,True):
                    for p in paths:
                        if p['hole']==hole:cv2.fillPoly(mask_image,[np.array(p['points'],np.int32)],0 if hole else 255)
                return mask_image,dict(visibleMaskContours=paths,maskReason=None,maskScope=record['maskScope'],maskInferenceMs=mask_report['timingsMs'][frame_id-report['sourceStart']-10])
        tracker=(ContourTracker if contour else SurfaceTracker)(*refs[:2],mesh.positions,*refs[2:],mesh=mesh,initial_pose=pose,visible_mask=mask,replenish=not synthetic,external_foreground=foreground)
        capture.set(cv2.CAP_PROP_POS_FRAMES,report['sourceStart']+10)
        landmarks=np.linspace(0,len(mesh.positions)-1,min(128,len(mesh.positions)),dtype=int)
        native_k=np.array(report['cameraCalibration']).reshape(3,3)
        timings=[];traces=[];rows=[];reference_frames=report['referenceFrames']
        try:
            for offset,record in enumerate(reference_frames):
                ok,bgr=capture.read()
                if not ok:raise ValueError('Missing evaluation frame.')
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                begin=time.perf_counter();working,k=tracking_input(gray,calibration,height)
                result=tracker.update(working,k,record['frameId']);timings.append((time.perf_counter()-begin)*1000)
                traces.append(result.manifest())
                matrix=record['cameraFromObject'];pose=None if matrix is None else (np.array(matrix)[:3,:3],np.array(matrix)[:3,3])
                visible,expected=evaluation_truth(mesh,landmarks,pose,native_k,*report['nativeResolution'])
                actual=projection(mesh.positions[landmarks],result.rotation,result.translation,native_k)[0] if result.state=='tracking' else None
                rows.extend(dict(frame=record['frameId'],point=int(i),valid=int(actual is not None),
                    expected_x=float(expected[i,0]*720/report['nativeResolution'][1]),expected_y=float(expected[i,1]*720/report['nativeResolution'][1]),
                    observed_x='' if actual is None else float(actual[i,0]*720/report['nativeResolution'][1]),
                    observed_y='' if actual is None else float(actual[i,1]*720/report['nativeResolution'][1])) for i in visible)
                if offset%40==0:print(f'{alias}: retry {offset+1}/240 · {result.state} · {result.statistics.get("contourReason",result.reason)}',flush=True)
        finally:capture.release()
        ms=np.array(timings[5:]);score=scorer(rows);availability=sum(f['state']=='tracking' for f in traces)/240
        score['allFrameAvailability']=availability
        candidate=dict(name=f'{"model-mask" if synthetic and mask else "model" if synthetic else "mask" if mask else "contour" if contour else "surface"}-{height}-v1',label=f'{"Model references + foreground" if synthetic else "Foreground + recovery" if mask else "Contour + recovery" if contour else "Recovery candidate"} · {height}p tracking',levels=8,trackingHeight=height,
            medianMs=float(np.median(ms)),p95Ms=float(np.percentile(ms,95)),measuredFrames=len(ms),
            framesOver60FpsFrameBudget=int(np.sum(ms>1000/60)),score=score,
            accuracyAnd60FpsGatePassed=bool(score['attachmentGatePassed'] and availability>=.9 and np.percentile(ms,95)<=10),
            onboarding=diagnostics,frames=traces,frameTimingMs=timings,
            comparisonScope='Same footage/windows as baseline; fresh quality retry, not a paired timing comparison.')
        if learned:
            candidate.update(name='learned-mask-pose-720-v1',label='Learned mask + model references · 720p tracking')
            total=np.array(timings)+np.array(mask_report['timingsMs'])
            candidate['maskAndTrackerLatency']=dict(medianMs=float(np.median(total[5:])),p95Ms=float(np.percentile(total[5:],95)),scope='Sum of separately measured model and tracker calls; excludes serialization/decode/render. Not concurrent live performance.')
            candidate['accuracyAnd60FpsGatePassed']=False
            candidate['comparisonScope']+=' Independent learned-mask inference precomputed; tracker-only latency excludes the recorded mask-model cost.'
            report['latencyScope']='Tracker resize + vision with independent precomputed learned masks. Model cost excluded from tracker timing and shown separately as summed model + tracker latency. Decode, scoring, rendering and serialization excluded; five warmups only from timing.'
        report['variants']=[candidate];enrich_report(report)
        directory=output/alias;directory.mkdir(exist_ok=True)
        write_artifact(directory/'report.json',json.dumps(report,indent=2))
        reports.append(report);write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1,objects=reports),indent=2))
        print(json.dumps({k:v for k,v in candidate.items() if k not in ('frames','frameTimingMs')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache'/'show3d')
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts'/'video-60')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'video-60'/'retry-1')
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--objects',nargs='*',choices=['keyboard','mug','ranch'])
    parser.add_argument('--contour',action='store_true')
    parser.add_argument('--mask',action='store_true')
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--learned',action='store_true')
    args=parser.parse_args()
    if not 240<=args.height<=720:parser.error('Tracking height must be 240–720.')
    run(args.assets,args.source,args.output,args.height,args.objects,args.contour,args.mask,args.synthetic,args.learned)
