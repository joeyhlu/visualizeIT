"""Native-resolution HOPE static-orbit RGB benchmark, separate from RGB-D study.

Research source: NVIDIA/BOP, CC BY-NC-SA 4.0 according to bundled metadata.
Onboarding labels initialize ten frames; evaluation labels never enter tracker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import urllib.error
import numpy as np
from PIL import Image
from .remote_zip import RemoteZip
from .bop_assets import cache_write
from .ply_model import read_ply, pose_camera
from .scan_mapping import unwrap
from .mapping_candidates import correct_scale
from .video_experiment import annotated_pose, projection, scorer, write_annotations, display_mesh
from .renderer import render, ray_triangle
from .storage import save_image, write_artifact, LIMIT
from . import storage
from .tracker import RigidTracker
from .vision import cv2
from .ycb_assets import ROOT

REVISION = 'ddd0a26ca3460085e93b648748160fb25e4b5566'
BASE = f'https://huggingface.co/datasets/bop-benchmark/hope/resolve/{REVISION}/'
OBJECTS = {5: 'Chocolate pudding box', 7: 'Corn can'}


def fetch(archive, member):
    for attempt in range(3):
        try: return archive.read(member)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2: raise
            time.sleep(1+attempt)


def acquire(root, count=120):
    root.mkdir(parents=True, exist_ok=True)
    paths = [root/f'object-{obj:06d}.json' for obj in OBJECTS]
    if all(p.exists() for p in paths):
        for p in paths:
            m = json.loads(p.read_text())
            if len(m['frames']) != count: raise ValueError('Cached frame count differs.')
            for asset in [m['model']]+[f[k] for f in m['frames'] for k in ('rgb','evaluationMask')]:
                if hashlib.sha256((root/asset['path']).read_bytes()).hexdigest() != asset['sha256']:
                    raise ValueError('Source hash mismatch.')
        return paths
    archive = RemoteZip(BASE+'hope_onboarding_static.zip')
    models = RemoteZip(BASE+'hope_models.zip')
    base = RemoteZip(BASE+'hope_base.zip')
    cache_write(root, root/'source-dataset_info.md', fetch(base,'hope/dataset_info.md'))
    for obj, name in OBJECTS.items():
        path = root/f'object-{obj:06d}.json'
        if path.exists(): continue
        prefix = f'onboarding_static/obj_{obj:06d}_up/'
        gt = json.loads(fetch(archive,prefix+'scene_gt.json'))
        calibration = json.loads(fetch(archive,prefix+'scene_camera.json'))
        keys = sorted(map(int,gt))[:count]
        if len(keys) != count or keys[-1]-keys[0] != count-1: raise ValueError('Need consecutive frames.')
        model_path = f'models/obj_{obj:06d}.ply'
        model_hash = cache_write(root,root/model_path,fetch(models,model_path))
        work = [(frame,kind,prefix+(f'rgb/{frame:06d}.jpg' if kind == 'rgb' else f'mask_visib/{frame:06d}_000000.png'))
                for frame in keys for kind in ('rgb','evaluationMask')]
        # Fetch concurrently, write sequentially so cache accounting cannot race.
        def read(item):
            frame,kind,member = item
            raw = (root/member).read_bytes() if (root/member).exists() else fetch(archive,member)
            return frame,kind,member,raw
        assets = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for frame,kind,member,raw in pool.map(read,work):
                digest = cache_write(root,root/member,raw)
                assets[(frame,kind)] = dict(path=member,sha256=digest)
                if kind == 'rgb' and frame%20 == 0: print(f'{name}: acquired {frame+1}/{count}',flush=True)
        records = [dict(frameId=frame, evaluationPose=gt[str(frame)][0], calibration=calibration[str(frame)],
                        rgb=assets[(frame,'rgb')],evaluationMask=assets[(frame,'evaluationMask')]) for frame in keys]
        directions = []
        for record in records:
            r,t = annotated_pose(record); direction=-r.T@t
            directions.append(direction/np.linalg.norm(direction))
        span=float(np.degrees(np.arccos(np.clip(np.min(np.array(directions)@np.array(directions).T),-1,1))))
        manifest=dict(schemaVersion=1,dataset='HOPE static onboarding',source=BASE,revision=REVISION,
            sourceLicense='CC BY-NC-SA 4.0 per bundled dataset_info.md; local research only',
            authors='NVIDIA; Stephen Tyree et al.; BOP onboarding conversion',objectId=obj,name=name,
            sequence=prefix,frames=records,viewSpanDegrees=span,hasRecordedDepth=False,
            model=dict(path=model_path,sha256=model_hash))
        cache_write(root,path,(json.dumps(manifest,indent=2)+'\n').encode())
    return paths


def image_for(root,record,key='rgb'):
    asset=record[key]; raw=(root/asset['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset['sha256']: raise ValueError('Capture hash mismatch.')
    image=Image.open(io.BytesIO(raw))
    return np.asarray(image.convert('RGB')) if key == 'rgb' else np.asarray(image)>0


def tracking_input(rgb, calibration, height=None, grayscale=False, resize_first=False):
    k=np.array(calibration['cam_K'],dtype=float).reshape(3,3)
    if grayscale and rgb.ndim==3 and not resize_first:rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if height is None or height==rgb.shape[0]:
        if grayscale and rgb.ndim==3:rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        return rgb,k
    width=round(rgb.shape[1]*height/rgb.shape[0]);sx=width/rgb.shape[1];sy=height/rgb.shape[0]
    k[0]*=sx;k[1]*=sy
    k[0,2]+=(sx-1)/2;k[1,2]+=(sy-1)/2
    result=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
    if grayscale and result.ndim==3:result=cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
    return result,k


def references(root,records,mesh,transform,height=None,grayscale=False):
    orb=cv2.ORB_create(nfeatures=2000); inverse=np.linalg.inv(transform)
    all_points=[]; all_descriptors=[]
    for record in records:
        rgb=image_for(root,record); mask=image_for(root,record,'evaluationMask')
        rgb,k=tracking_input(rgb,record['calibration'],height,grayscale)
        mask=cv2.resize(mask.astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        r,t=annotated_pose(record); camera=pose_camera(r,t,dict(cam_K=k.ravel().tolist()),transform,rgb.shape[1],rgb.shape[0])
        # Annotated mask is permitted only during these ten initialization frames.
        frame=render(mesh,camera)
        gray=rgb if rgb.ndim==2 else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        features,descriptors=orb.detectAndCompute(gray,((frame.triangle>=0)&mask).astype(np.uint8)*255)
        points=[]; pixels=[]; accepted=[]
        if descriptors is not None:
            for feature,descriptor in zip(features,descriptors):
                x,y=feature.pt; triangle=frame.triangle[int(y),int(x)]
                if triangle<0: continue
                indices=mesh.triangles[triangle]; direction=camera.rays([x,y])
                hit=ray_triangle(camera.eye,direction,mesh.positions[indices],mesh.uv[indices])
                if hit is None: continue
                p=camera.eye+direction*hit[0]
                points.append(inverse[:3,:3]@p+inverse[:3,3]);pixels.append([x,y]);accepted.append(descriptor)
        all_points.extend(points);all_descriptors.extend(accepted)
        print(f'Onboarding frame {record["frameId"]}: {len(points)} surface features',flush=True)
    return (np.asarray(all_points).reshape(-1,3),np.asarray(all_descriptors,dtype=np.uint8).reshape(-1,32),
            gray,np.asarray(pixels).reshape(-1,2),np.asarray(points).reshape(-1,3))


def unoccluded(points,vertices,triangles,eye):
    """Independent ray/triangle visibility; no measured depth is invented."""
    triangles=vertices[triangles]; edge1=triangles[:,1]-triangles[:,0];edge2=triangles[:,2]-triangles[:,0]
    origin=eye-triangles[:,0]; cross_origin=np.cross(origin,edge1)
    result=[]
    for point in points:
        vector=point-eye; distance=np.linalg.norm(vector); direction=vector/distance
        cross=np.cross(direction,edge2);det=np.sum(edge1*cross,axis=1)
        good=np.abs(det)>1e-12;inverse=np.divide(1,det,out=np.zeros_like(det),where=good)
        u=np.sum(origin*cross,axis=1)*inverse
        v=np.sum(cross_origin*direction,axis=1)*inverse
        hit=np.sum(edge2*cross_origin,axis=1)*inverse
        occluded=good&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(hit>1e-6)&(hit<distance-.001)
        result.append(not np.any(occluded))
    return np.array(result)


def annotations(record,points,normals,vertices,triangles,mask,estimate):
    k=np.array(record['calibration']['cam_K']).reshape(3,3); r,t=annotated_pose(record)
    expected,z=projection(points,r,t,k);h,w=mask.shape
    x,y=np.floor(expected).astype(int).T
    facing=np.sum((normals@r.T)*(points@r.T+t),axis=1)<0
    visible=(x>=0)&(x<w)&(y>=0)&(y<h)&(z>.03)&facing
    visible&=mask[np.clip(y,0,h-1),np.clip(x,0,w-1)]
    selected=np.flatnonzero(visible)
    visible[selected]&=unoccluded(points[selected],vertices,triangles,-r.T@t)
    actual=projection(points,estimate.rotation,estimate.translation,k)[0] if estimate.state=='tracking' else None
    return [dict(frame=record['frameId'],point=int(i),valid=int(actual is not None),
        expected_x=float(expected[i,0]*720/h),expected_y=float(expected[i,1]*720/h),
        observed_x=float(actual[i,0]*720/h) if actual is not None else '',
        observed_y=float(actual[i,1]*720/h) if actual is not None else '') for i in np.flatnonzero(visible)]


def overlay(rgb,mesh,camera):
    frame=render(mesh,camera);out=rgb.copy();mask=frame.triangle>=0;out[mask]=frame.color[mask]
    return out


def export_web(output):
    """Encode browser-playable native-size derivatives without rerunning tracking."""
    for report_path in sorted(output.glob('object-*/report.json')):
        report=json.loads(report_path.read_text())
        if report['presentation'].get('videoPending'):continue
        for kind in ('original','supplied-pose','estimated-pose'):
            source=report_path.parent/f'{kind}.mp4';destination=source.with_suffix('.webm')
            capture=cv2.VideoCapture(str(source))
            writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'VP80'),10,tuple(report['nativeResolution']))
            if not capture.isOpened() or not writer.isOpened():raise RuntimeError('VP8 video export unavailable.')
            count=0
            try:
                while True:
                    good,frame=capture.read()
                    if not good:break
                    writer.write(frame);count+=1
                    if shutil.disk_usage(output).free<storage.RESERVE:raise ValueError('Disk reserve reached.')
                    if count%10==0 and sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>LIMIT:
                        raise ValueError('HD video output budget reached.')
            finally:capture.release();writer.release()
            check=cv2.VideoCapture(str(destination))
            valid=(int(check.get(cv2.CAP_PROP_FRAME_COUNT))==count and
                [int(check.get(cv2.CAP_PROP_FRAME_WIDTH)),int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))]==report['nativeResolution'])
            check.release()
            if not valid or count==0:raise ValueError('Video export dimensions/frame count failed.')
            print(f'{report["name"]}: {kind} VP8, {count} native-size frames',flush=True)
        report['presentation']['browserCodec']='VP8/WebM derivative; MP4 originals retained'
        write_artifact(report_path,json.dumps(report,indent=2)+'\n')
    gallery(output,[json.loads(p.read_text()) for p in sorted(output.glob('object-*/report.json'))])


def render_saved(root,output):
    """Export completed accuracy traces with one native video encoder at a time."""
    for report_path in sorted(output.glob('object-*/report.json')):
        report=json.loads(report_path.read_text())
        if not report['presentation'].get('videoPending'):continue
        manifest=json.loads((root/f'object-{report["objectId"]:06d}.json').read_text())
        scan,_,transform=read_ply(root/manifest['model']['path'],report['name'])
        mesh,_,_,_,details=unwrap(scan,report['name'],details=True)
        mesh=display_mesh(mesh,correct_scale(mesh,details['metricUV'],details['chartIds']),None,'repeat',.025,0)
        records=manifest['frames'][10:];stride=report['presentation']['renderStride']
        indices=sorted(set(range(0,len(records),stride))|{len(records)-1})
        w,h=report['nativeResolution'];thumbnails=[]
        for kind in ('original','supplied-pose','estimated-pose'):
            writer=cv2.VideoWriter(str(report_path.parent/f'{kind}.webm'),cv2.VideoWriter_fourcc(*'VP80'),10,(w,h))
            if not writer.isOpened():raise RuntimeError('VP8 native encoder unavailable.')
            try:
                for index in indices:
                    record=records[index];rgb=image_for(root,record);result=report['frames'][index]
                    pixels=rgb
                    if kind=='supplied-pose':r,t=annotated_pose(record)
                    elif kind=='estimated-pose' and result['state']=='tracking':
                        matrix=np.array(result['cvCameraFromSourceObject']);r,t=matrix[:3,:3],matrix[:3,3]
                    else:r=t=None
                    if r is not None:pixels=overlay(rgb,mesh,pose_camera(r,t,record['calibration'],transform,w,h))
                    writer.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                    if index%20==0 or index==len(records)-1:
                        label=f'{record["frameId"]:06d}'
                        save_image(Image.fromarray(pixels),report_path.parent/f'{label}-{kind}.jpg',quality=97)
                        if kind=='estimated-pose':thumbnails.append(dict(frameId=record['frameId'],state=result['state'],prefix=label))
                    if shutil.disk_usage(output).free<storage.RESERVE:raise ValueError('Disk reserve reached.')
            finally:writer.release()
            print(f'{report["name"]}: exported {kind}, {len(indices)} native frames',flush=True)
        report['thumbnails']=thumbnails
        report['presentation'].update(videoPending=False,codec='VP8/WebM',browserCodec='VP8/WebM native render')
        write_artifact(report_path,json.dumps(report,indent=2)+'\n')
    gallery(output,[json.loads(p.read_text()) for p in sorted(output.glob('object-*/report.json'))])


def run(root,output,acquire_only=False,stride=2,score_only=False):
    paths=acquire(root)
    if acquire_only:return
    output.mkdir(parents=True,exist_ok=True);reports=[]
    for path in paths:
        manifest=json.loads(path.read_text()); obj=manifest['objectId'];name=manifest['name']
        directory=output/f'object-{obj:06d}';directory.mkdir(exist_ok=True)
        saved=directory/'report.json'
        if saved.exists():
            previous=json.loads(saved.read_text())
            if (previous['source']==BASE and previous['sequence']==manifest['sequence'] and
                previous['presentation']['renderStride']==stride and
                [r['frameId'] for r in previous['frames']]==[r['frameId'] for r in manifest['frames'][10:]]):
                reports.append(previous);print(f'{name}: complete saved trace reused.',flush=True);continue
        scan,vertices,transform=read_ply(root/manifest['model']['path'],name)
        mesh,_,_,_,details=unwrap(scan,name,details=True)
        metric=correct_scale(mesh,details['metricUV'],details['chartIds'])
        mesh=display_mesh(mesh,metric,None,'repeat',.025,0)
        refs=references(root,manifest['frames'][:10],mesh,transform)
        tracker=RigidTracker(refs[0],refs[1],vertices,*refs[2:])
        indices=np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int)
        points=vertices[indices];normals=scan.normals[indices]@transform[:3,:3]
        source_triangles=scan.triangles[:,[0,2,1]]
        evaluation=manifest['frames'][10:];rows=[];trace=[];timings=[];thumbnails=[]
        first_rgb=image_for(root,evaluation[0]);h,w=first_rgb.shape[:2]
        if (w,h)!=(1920,1440):raise ValueError('Expected native 1920x1440 capture; refuse upscaling.')
        writers={}
        for kind in (() if score_only else ('original','supplied-pose','estimated-pose')):
            writer=cv2.VideoWriter(str(directory/f'{kind}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
            if not writer.isOpened():raise RuntimeError('Native MP4 encoder unavailable.')
            writers[kind]=writer
        try:
            for index,record in enumerate(evaluation):
                rgb=image_for(root,record);k=np.array(record['calibration']['cam_K']).reshape(3,3)
                start=time.perf_counter();estimate=tracker.update(rgb,k,record['frameId']);timings.append(time.perf_counter()-start)
                trace.append(estimate.manifest())
                # Evaluation-only annotations/masks are consumed after tracker.update.
                mask=image_for(root,record,'evaluationMask')
                rows.extend(annotations(record,points,normals,vertices,source_triangles,mask,estimate))
                if not score_only and (index%stride==0 or index==len(evaluation)-1):
                    r,t=annotated_pose(record)
                    control=overlay(rgb,mesh,pose_camera(r,t,record['calibration'],transform,w,h))
                    tracked=overlay(rgb,mesh,pose_camera(estimate.rotation,estimate.translation,record['calibration'],transform,w,h)) if estimate.state=='tracking' else rgb.copy()
                    for kind,pixels in (('original',rgb),('supplied-pose',control),('estimated-pose',tracked)):
                        writers[kind].write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                    if index%20==0 or index==len(evaluation)-1:
                        label=f'{record["frameId"]:06d}'
                        for kind,pixels in (('original',rgb),('supplied-pose',control),('estimated-pose',tracked)):
                            save_image(Image.fromarray(pixels),directory/f'{label}-{kind}.jpg',quality=97)
                        thumbnails.append(dict(frameId=record['frameId'],state=estimate.state,prefix=label))
                    if shutil.disk_usage(output).free<storage.RESERVE:raise ValueError('Disk reserve reached.')
                    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>LIMIT:
                        raise ValueError('HD video output budget reached.')
                if index%10==0:print(f'{name}: {index+1}/{len(evaluation)} {estimate.state}, {estimate.reason}',flush=True)
        finally:
            for writer in writers.values():writer.release()
        write_annotations(directory/'annotations.csv',rows)
        score=scorer(rows);availability=sum(t['state']=='tracking' for t in trace)/len(trace)
        score.update(nativeMedianErrorPixels=score.get('medianErrorPixels')*h/720 if score.get('medianErrorPixels') is not None else None,
            nativeP95ErrorPixels=score.get('p95ErrorPixels')*h/720 if score.get('p95ErrorPixels') is not None else None,
            allFrameAvailability=availability)
        report=dict(schemaVersion=1,name=name,objectId=obj,dataset=manifest['dataset'],source=BASE,
            license=manifest['sourceLicense'],authors=manifest['authors'],sequence=manifest['sequence'],nativeResolution=[w,h],
            evaluationFrames=len(trace),onboardingFrames=10,onboarding='Annotated poses and masks for first ten frames only',
            hasRecordedDepth=False,physicalPhoneValidationPassed=False,viewSpanDegrees=manifest['viewSpanDegrees'],
            appearance='Neutral diffuse diagnostic checker; no external occlusion or PBR lighting',
            visibilityEvaluation='Evaluation mask + normals + independent mesh ray visibility; no evaluation labels enter tracker',
            presentation=dict(codec='MPEG-4 Part 2',renderStride=stride,fps=10,sourceFpsKnown=False,nativeResolution=True,videoPending=score_only,
                meaning='Sampled frames at presentation timing; no source-frame-rate or real-time claim'),
            score=score,frames=trace,thumbnails=thumbnails,trackingTimeMedianSeconds=float(np.median(timings)),
            trackingTimeP95Seconds=float(np.percentile(timings,95)),timingScope='Native-resolution CPU tracker only; rendering/loading excluded')
        write_artifact(directory/'report.json',json.dumps(report,indent=2)+'\n');reports.append(report)
        print(f'{name}: {score}, view span {manifest["viewSpanDegrees"]:.1f}',flush=True)
        gallery(output,reports)
    gallery(output,reports)


def gallery(output,reports):
    sections=[]
    for report in reports:
        folder=f'object-{report["objectId"]:06d}';score=report['score']
        suffix='webm' if 'browserCodec' in report['presentation'] else 'mp4'
        videos=''.join(f'<figure><video controls preload="metadata" src="{folder}/{kind}.{suffix}"></video><figcaption>{kind}</figcaption></figure>' for kind in ('original','supplied-pose','estimated-pose')) if not report['presentation'].get('videoPending') else '<p>All-frame accuracy run complete; video export pending.</p>'
        snapshots=' · '.join(f'<a href="{folder}/{t["prefix"]}-estimated-pose.jpg">Native JPEG frame {t["frameId"]} ({t["state"]})</a>' for t in report['thumbnails'])
        median=score.get('nativeMedianErrorPixels');p95=score.get('nativeP95ErrorPixels')
        median=f'{median:.2f}' if median is not None else 'unavailable';p95=f'{p95:.2f}' if p95 is not None else 'unavailable'
        sections.append(f'<h2>{report["name"]}</h2><p>{report["evaluationFrames"]} evaluated frames, {report["viewSpanDegrees"]:.1f}° view span. Tracking availability {score["allFrameAvailability"]:.1%}. Native pixel error during valid tracking: median {median}, P95 {p95}.</p><div>{videos}</div><p>{snapshots}</p><a href="{folder}/report.json">Results and every-frame trace</a> · <a href="{folder}/annotations.csv">Independent evaluation</a>')
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Native HD mapping tests</title><style>body{font:17px system-ui;margin:28px;background:#eff4f2}div{display:flex;gap:12px}figure{width:33%;margin:0}video{width:100%}p{line-height:1.5}@media(max-width:800px){div{display:block}figure{width:100%}}</style><h1>1920 × 1440 real object-orbit tests</h1><p>Original footage, supplied-pose control, and estimated-pose overlay. Ten annotated onboarding frames, then RGB-only tracking. All 110 held-out frames scored. Video is sampled at presentation timing (see reports); no upscaling. Lost tracking hides the overlay. No measured depth, hand occlusion, natural lighting or phone validation is claimed.</p>'+''.join(sections)+'<p>Research source: NVIDIA HOPE/BOP. Original source metadata specifies CC BY-NC-SA 4.0; authors Tyree et al. <a href="https://huggingface.co/datasets/bop-benchmark/hope">Dataset</a>. Checker renders, videos and reports are derived research outputs. <a href="report.json">Full results</a></p>'
    write_artifact(output/'index.html',html)
    write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1,objects=reports),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache'/'hope-hd')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'video-hd')
    parser.add_argument('--acquire-only',action='store_true')
    parser.add_argument('--score-only',action='store_true',help='Score without concurrent native video encoders.')
    parser.add_argument('--reserve-mib',type=int,default=512,help='Disk reserve; explicit low-space research runs may use at least 128 MiB.')
    parser.add_argument('--export-web',action='store_true',help='Convert saved videos to browser-playable VP8 without rerunning tracking.')
    parser.add_argument('--render-saved',action='store_true',help='Render accuracy traces with one native encoder at a time.')
    parser.add_argument('--render-stride',type=int,default=2)
    args=parser.parse_args()
    if args.render_stride<1:parser.error('Render stride must be positive.')
    if args.reserve_mib<128:parser.error('Retain at least 128 MiB disk reserve.')
    storage.RESERVE=args.reserve_mib*1024*1024
    if args.export_web:export_web(args.output)
    elif args.render_saved:render_saved(args.assets,args.output)
    else:
        run(args.assets,args.output,args.acquire_only,args.render_stride,args.score_only)
        if not args.acquire_only and not args.score_only:export_web(args.output)
