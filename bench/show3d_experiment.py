"""Genuine 60 FPS moving-object replay; reference poses are estimated labels.

SHOW3D source is CC BY-NC 4.0. Meshes retain HOT3D dataset terms.
No evaluation pose, silhouette or mask enters the tracker after onboarding.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import time
import urllib.request
import numpy as np
from .vision import cv2
from .model import Camera
from .glb_model import read_glb
from .renderer import render,ray_triangle
from .fast_tracker import FastRigidTracker
from .performance_experiment import compact_bank
from .video_experiment import projection,scorer
from .hd_experiment import unoccluded,tracking_input
from .storage import write_artifact
from .ycb_assets import ROOT

REVISION='720104c5793a55aab352ca933f1f73bde8582d3c'
MODEL_REVISION='30fe9674782f32e1e5edba98476b6ff4300132c5'
BASE=f'https://huggingface.co/datasets/facebook/show3d-dataset/resolve/{REVISION}/'
MODEL_BASE=f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{MODEL_REVISION}/'
CASES=(('keyboard','ASC023/keyboard_inspecting_8039','keyboard'),
       ('mug','ASC023/mug_inspecting_817d','mug_white'),
       ('ranch','LWA828/ranch_inspecting_ca23','bottle_ranch'))
MAX_CACHE=384*1024*1024


def fetch(root,path,url,maximum=64*1024*1024):
    target=root/path
    if target.exists():return target
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.part')
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    count=0
    try:
        with urllib.request.urlopen(url,timeout=45) as response,temporary.open('wb') as output:
            while chunk:=response.read(1024*1024):
                count+=len(chunk)
                if count>maximum or used+count>MAX_CACHE or shutil.disk_usage(root).free-count<512*1024*1024:
                    raise ValueError('Bounded acquisition/disk reserve reached.')
                output.write(chunk)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return target


def proper_rotation(value):
    matrix=np.array(value,float).reshape(3,3)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite rotation.')
    u,_,v=np.linalg.svd(matrix);rotation=u@v
    if np.linalg.det(rotation)<0 or np.linalg.norm(rotation-matrix)>1e-3:raise ValueError('Invalid rotation.')
    return rotation


def reference_pose(pose,calibration,index):
    camera=calibration['T_WorldFromCamera_by_index'].get(str(index),{})
    if pose.get('confidence',0)<.5 or not camera.get('is_pose_valid',False):return None
    if calibration['DistortionModel']!='PinholePlane':raise ValueError('Released pinhole view required.')
    world_from_camera=np.array(camera['T_WorldFromCamera'],float).reshape(4,4)
    rc=proper_rotation(world_from_camera[:3,:3]);tc=world_from_camera[:3,3]*.001
    ro=proper_rotation(pose['R']);to=np.array(pose['t'],float).reshape(3)*.001
    if abs(camera['timestamp']-pose['timestamp'])>1e-4:raise ValueError('Pose/camera timestamps disagree.')
    return rc.T@ro,rc.T@(to-tc)


def choose_window(poses,calibration,count=250):
    valid={i:reference_pose(p,calibration,i) for i,p in ((int(k),v) for k,v in poses.items())}
    for start in range(max(valid)-count+2):
        if all(valid.get(i) is not None for i in range(start,start+10)) and sum(valid.get(i) is not None for i in range(start,start+count))>=.9*count:
            times=np.array([poses[str(i)]['timestamp'] for i in range(start,start+count)])
            if not np.allclose(np.diff(times),1/60,atol=1e-4):continue
            return start,valid
    raise ValueError('No consecutive window with ten valid onboarding poses and >=90% reference coverage.')


def camera_for(pose,k,width,height):
    r,t=pose
    return Camera(width,height,k[0,0],k[1,1],k[0,2],k[1,2],-r.T@t,r)


def initialize(capture,start,poses,mesh,calibration,height):
    capture.set(cv2.CAP_PROP_POS_FRAMES,start)
    orb=cv2.ORB_create(nfeatures=2000)
    all_points=[];all_descriptors=[]
    for index in range(start,start+10):
        ok,bgr=capture.read()
        if not ok:raise ValueError('Incomplete onboarding decode.')
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        working,k=tracking_input(gray,calibration,height)
        camera=camera_for(poses[index],k,working.shape[1],working.shape[0]);frame=render(mesh,camera)
        features,descriptors=orb.detectAndCompute(working,(frame.triangle>=0).astype(np.uint8)*255)
        points=[];pixels=[];accepted=[]
        if descriptors is not None:
            for feature,descriptor in zip(features,descriptors):
                x,y=feature.pt;triangle=frame.triangle[int(y),int(x)]
                if triangle<0:continue
                indices=mesh.triangles[triangle];ray=camera.rays([x,y])
                hit=ray_triangle(camera.eye,ray,mesh.positions[indices],mesh.uv[indices])
                if hit is None:continue
                points.append(camera.eye+ray*hit[0]);pixels.append([x,y]);accepted.append(descriptor)
        all_points.extend(points);all_descriptors.extend(accepted)
        print(f'Onboarding {index}: {len(points)} geometric surface features',flush=True)
    if len(all_points)<12:raise ValueError('Insufficient surface reference features.')
    return (np.array(all_points),np.array(all_descriptors,np.uint8),working,
            np.array(pixels).reshape(-1,2),np.array(points).reshape(-1,3))


def evaluation_truth(mesh,landmarks,pose,k,width,height):
    if pose is None:return [],None
    r,t=pose;p=mesh.positions[landmarks];n=mesh.normals[landmarks]
    expected,z=projection(p,r,t,k)
    visible=(z>.03)&(expected[:,0]>=0)&(expected[:,0]<width)&(expected[:,1]>=0)&(expected[:,1]<height)
    visible&=np.sum((n@r.T)*(p@r.T+t),axis=1)<0
    selected=np.flatnonzero(visible)
    visible[selected]&=unoccluded(p[selected],mesh.positions,mesh.triangles,-r.T@t)
    return np.flatnonzero(visible),expected


def matrix_for(pose):
    if pose is None:return None
    r,t=pose;matrix=np.eye(4);matrix[:3,:3]=r;matrix[:3,3]=t
    return matrix.tolist()


def enrich_report(report):
    """Retain timings for accepted states separately from cheap failed frames."""
    report['environment']=dict(platform=platform.platform(),processor=platform.processor(),
        python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,openCVThreads=1)
    report['timingLimit']='Short desktop replay, not sustained phone throughput. Failed states count in all-state latency; inspect state-specific samples.'
    for variant in report['variants']:
        variant['framesOver60FpsFrameBudget']=variant.pop('processingFramesOver16ms',variant.get('framesOver60FpsFrameBudget',0))
        for field in ('medianErrorPixels','p95ErrorPixels'):
            value=variant['score'][field]
            variant['score']['native'+field[0].upper()+field[1:]]=None if value is None else value*report['nativeResolution'][1]/720
        variant['latencyByTrackingState']={}
        for state in ('tracking','limited','lost'):
            samples=[t for t,f in zip(variant['frameTimingMs'][5:],variant['frames'][5:]) if f['state']==state]
            variant['latencyByTrackingState'][state]=dict(samples=len(samples),
                medianMs=float(np.median(samples)) if samples else None,
                p95Ms=float(np.percentile(samples,95)) if samples else None)
    return report


def run(root,output,height=360):
    cv2.setNumThreads(1);root.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    for path in ('README.md','LICENSE','object_pose/README.md','scenes/README.md'):
        fetch(root,path,BASE+path,2*1024*1024)
    info_path=fetch(root,'models/models_info.json',MODEL_BASE+'object_models/models_info.json',2*1024*1024)
    fetch(root,'models/dataset-license.pdf',MODEL_BASE+'hot3d_dataset_license_agreement.pdf',4*1024*1024)
    models=json.loads(info_path.read_text());reports=[]
    for alias,scene,model_name in CASES:
        prefix='scenes/'+scene+'/'
        paths={name:fetch(root,prefix+name,BASE+prefix+name) for name in (
            'headset0.mp4','camera_calibration/headset0.json','metadata/recording_info.json')}
        pose_path=fetch(root,'object_pose/v1/'+prefix+'object_pose.json',BASE+'object_pose/v1/'+prefix+'object_pose.json',8*1024*1024)
        model_id=next(int(i) for i,m in models.items() if m['name']==model_name)
        model_path=fetch(root,f'models/obj_{model_id:06d}.glb',MODEL_BASE+f'object_models/obj_{model_id:06d}.glb',24*1024*1024)
        asset_paths=list(paths.values())+[pose_path,model_path]
        hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_paths}
        hash_path=root/f'{alias}-hashes.json'
        if hash_path.exists() and json.loads(hash_path.read_text())!=hashes:raise ValueError('Cached source changed.')
        hash_path.write_text(json.dumps(hashes,indent=2))
        metadata=json.loads(paths['metadata/recording_info.json'].read_text())
        calibration=json.loads(paths['camera_calibration/headset0.json'].read_text());source_poses=json.loads(pose_path.read_text())
        calibration['cam_K']=[calibration['fx'],0,calibration['cx'],0,calibration['fy'],calibration['cy'],0,0,1]
        start,poses=choose_window(source_poses,calibration)
        mesh=read_glb(model_path,alias)
        extent=np.ptp(mesh.positions,axis=0);expected_extent=np.array([models[str(model_id)][k] for k in ('size_x','size_y','size_z')])
        if not np.allclose(extent,expected_extent,atol=1e-5,rtol=.01):raise ValueError('Canonical mesh unit/basis mismatch.')
        capture=cv2.VideoCapture(str(paths['headset0.mp4']))
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height_native=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=capture.get(cv2.CAP_PROP_FPS)
        if abs(fps-60)>.01 or metadata['fps']!=60 or [width,height_native]!=[calibration['ImageSizeX'],calibration['ImageSizeY']]:
            raise ValueError('Capture rate/image calibration mismatch.')
        print(f'{alias}: verified {width}x{height_native}, {fps} FPS; selected frames {start}..{start+249}',flush=True)
        refs=initialize(capture,start,poses,mesh,calibration,height)
        points,descriptors=compact_bank(refs[0],refs[1])
        trackers=[FastRigidTracker(points,descriptors,mesh.positions,*refs[2:],feature_limit=400,feature_levels=levels) for levels in (8,3)]
        traces=[[],[]];timings=[[],[]];rows=[[],[]];reference_frames=[]
        directory=output/alias;directory.mkdir(exist_ok=True)
        # Only source imagery is encoded; WebGL applies surfaces during playback.
        export_size=(640,800)
        preview=root/f'.{alias}-preview.webm'
        writer=cv2.VideoWriter(str(preview),cv2.VideoWriter_fourcc(*'VP80'),60,export_size)
        if not writer.isOpened():raise ValueError('Browser video encoder unavailable.')
        landmarks=np.linspace(0,len(mesh.positions)-1,min(128,len(mesh.positions)),dtype=int)
        native_k=np.array(calibration['cam_K']).reshape(3,3)
        try:
            for offset in range(240):
                index=start+10+offset;ok,bgr=capture.read()
                if not ok:raise ValueError('Missing consecutive source frame.')
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);estimates=[None,None]
                for candidate in ((0,1) if offset%2==0 else (1,0)):
                    begin=time.perf_counter();working,k=tracking_input(gray,calibration,height)
                    estimates[candidate]=trackers[candidate].update(working,k,index)
                    timings[candidate].append((time.perf_counter()-begin)*1000)
                    traces[candidate].append(estimates[candidate].manifest())
                reference=poses[index];visible,expected=evaluation_truth(mesh,landmarks,reference,native_k,width,height_native)
                reference_frames.append(dict(frameId=index,timestamp=source_poses[str(index)]['timestamp'],
                    confidence=source_poses[str(index)]['confidence'],cameraFromObject=matrix_for(reference)))
                for candidate,result in enumerate(estimates):
                    actual=projection(mesh.positions[landmarks],result.rotation,result.translation,native_k)[0] if result.state=='tracking' else None
                    rows[candidate].extend(dict(frame=index,point=int(i),valid=int(actual is not None),
                        expected_x=float(expected[i,0]*720/height_native),expected_y=float(expected[i,1]*720/height_native),
                        observed_x='' if actual is None else float(actual[i,0]*720/height_native),
                        observed_y='' if actual is None else float(actual[i,1]*720/height_native)) for i in visible)
                writer.write(cv2.resize(bgr,export_size,interpolation=cv2.INTER_AREA))
                if offset%40==0:print(f'{alias}: 60 FPS replay {offset+1}/240',flush=True)
                if preview.stat().st_size>16*1024*1024 or shutil.disk_usage(output).free<512*1024*1024:
                    raise ValueError('Preview size/disk reserve reached.')
        finally:capture.release();writer.release()
        try:
            check=cv2.VideoCapture(str(preview))
            valid=int(check.get(cv2.CAP_PROP_FRAME_COUNT))==240 and abs(check.get(cv2.CAP_PROP_FPS)-60)<.01
            check.release()
            if not valid or preview.stat().st_size>16*1024*1024:raise ValueError('Export frame/rate/size validation failed.')
            write_artifact(directory/'original.webm',preview.read_bytes())
        finally:preview.unlink(missing_ok=True)
        variants=[]
        for candidate,levels in enumerate((8,3)):
            ms=np.array(timings[candidate][5:]);score=scorer(rows[candidate]);availability=sum(f['state']=='tracking' for f in traces[candidate])/240
            score['allFrameAvailability']=availability
            summary=dict(levels=levels,medianMs=float(np.median(ms)),p95Ms=float(np.percentile(ms,95)),
                processingFramesOver16ms=int(np.sum(ms>1000/60)),measuredFrames=len(ms),score=score,
                accuracyAnd60FpsGatePassed=bool(score['attachmentGatePassed'] and availability>=.9 and np.percentile(ms,95)<=10),
                frames=traces[candidate],frameTimingMs=timings[candidate])
            variants.append(summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('frames','frameTimingMs')}),flush=True)
        report=dict(name=alias,scene=scene,sourceRevision=REVISION,modelRevision=MODEL_REVISION,modelName=model_name,modelId=model_id,
            source='https://huggingface.co/datasets/facebook/show3d-dataset',sourceLicense='SHOW3D CC BY-NC 4.0; meshes HOT3D dataset agreement retained',
            sourceHashes=hashes,nativeResolution=[width,height_native],previewResolution=list(export_size),sourceFps=60,previewFps=60,
            onboardingFrames=10,evaluationFrames=240,sourceStart=start,trackingHeight=height,referenceFeatures=len(points),
            referencePoseFrames=sum(f['cameraFromObject'] is not None for f in reference_frames),cameraCalibration=calibration['cam_K'],
            referenceScope='Dataset estimated object poses, confidence >=0.5 and valid camera pose; reference labels are imperfect, not human ground truth.',
            visibilityScope='Mesh normals and independent self-visibility only; hand/background occlusion is not labelled or suppressed.',
            latencyScope='CPU native-monochrome resize + tracker; first five evaluation frames excluded only from timing; excludes decoding, scoring, encoding and rendering.',
            renderScope='Browser GPU checker over actual mesh UVs; geometric silhouette is not a predicted visible-object mask. No PBR or hand occlusion.',
            targetFps=60,visionP95BudgetMs=10,phonePerformanceValidated=False,variants=variants,referenceFrames=reference_frames)
        enrich_report(report)
        write_artifact(directory/'report.json',json.dumps(report,indent=2)+'\n')
        # Portable mesh buffer: positions, normals, UVs and uint32 triangle indices.
        vertex=np.column_stack((mesh.positions,mesh.normals,mesh.uv)).astype('<f4')
        indices=mesh.triangles.astype('<u4').ravel()
        write_artifact(directory/'mesh.bin',vertex.tobytes()+indices.tobytes())
        player=dict(name=alias,width=width,height=height_native,k=calibration['cam_K'],vertices=len(vertex),indices=len(indices),
            frames=[dict(reference=r['cameraFromObject'],poses=[v['frames'][i]['cvCameraFromSourceObject'] if v['frames'][i]['state']=='tracking' else None for v in variants],
                         states=[v['frames'][i]['state'] for v in variants],frameId=r['frameId']) for i,r in enumerate(reference_frames)])
        write_artifact(directory/'player.json',json.dumps(player,separators=(',',':')))
        reports.append(report)
        write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1,objects=reports),indent=2)+'\n')
    from .show3d_player import build
    build(output,reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache'/'show3d')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'video-60')
    parser.add_argument('--height',type=int,default=360)
    args=parser.parse_args()
    if not 240<=args.height<=720:parser.error('Tracking height 240–720 required.')
    run(args.assets,args.output,args.height)
