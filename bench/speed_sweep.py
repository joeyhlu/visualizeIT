"""Paired CPU vision speed/accuracy comparison with shared onboarding references.

Evaluation labels enter only the independent scorer, never any tracker.
"""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
from .hd_experiment import image_for, references, tracking_input, annotations
from .performance_experiment import compact_bank
from .fast_tracker import FastRigidTracker
from .ply_model import read_ply
from .scan_mapping import unwrap
from .video_experiment import scorer, projection
from .storage import write_artifact
from .ycb_assets import ROOT
from .vision import cv2

VARIANTS=(('baseline',False,12,False),('resize-first',True,12,False),('resize-first-key16',True,16,False))


def rescore_visible_rows(template,points,estimate,calibration,height):
    """Reuse independent GT visibility; replace every candidate-specific field."""
    k=np.array(calibration['cam_K']).reshape(3,3)
    actual=projection(points,estimate.rotation,estimate.translation,k)[0] if estimate.state=='tracking' else None
    return [dict(row,valid=int(actual is not None),
                 observed_x='' if actual is None else float(actual[row['point'],0]*720/height),
                 observed_y='' if actual is None else float(actual[row['point'],1]*720/height)) for row in template]


def run(root,output,height=360,features=400,roi=False,levels=False):
    variants=(('baseline',False,12,False),('object-region',False,12,True)) if roi else VARIANTS
    variants=[(*v,8) for v in variants]
    if levels:variants=[('baseline',False,12,False,8),('pyramid-4',False,12,False,4),('pyramid-3',False,12,False,3)]
    cv2.setNumThreads(1)
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in sorted(root.glob('object-*.json')):
        manifest=json.loads(path.read_text());name=manifest['name']
        scan,vertices,transform=read_ply(root/manifest['model']['path'],name)
        mesh=unwrap(scan,name)[0]
        refs=references(root,manifest['frames'][:10],mesh,transform,height,True)
        points,descriptors=compact_bank(refs[0],refs[1])
        trackers=[FastRigidTracker(points,descriptors,vertices,*refs[2:],feature_limit=features,index_key_size=key,use_roi=use_roi,feature_levels=pyramid) for _,_,key,use_roi,pyramid in variants]
        timings=[[] for _ in variants];prep_times=[[] for _ in variants]
        traces=[[] for _ in variants];rows=[[] for _ in variants]
        indices=np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int)
        landmarks=vertices[indices];normals=scan.normals[indices]@transform[:3,:3]
        triangles=scan.triangles[:,[0,2,1]]
        for frame_index,record in enumerate(manifest['frames'][10:]):
            rgb=image_for(root,record)
            estimates=[None]*len(variants)
            # Rotate evaluation order to distribute scheduling/cache effects.
            for i in np.roll(np.arange(len(variants)),frame_index%len(variants)):
                _,resize_first,_,_,_=variants[i]
                start=time.perf_counter()
                working,k=tracking_input(rgb,record['calibration'],height,True,resize_first)
                prep_end=time.perf_counter()
                estimates[i]=trackers[i].update(working,k,record['frameId'])
                timings[i].append((time.perf_counter()-start)*1000)
                prep_times[i].append((prep_end-start)*1000)
                traces[i].append(estimates[i].manifest())
            mask=image_for(root,record,'evaluationMask')
            truth=annotations(record,landmarks,normals,vertices,triangles,mask,estimates[0])
            rows[0].extend(truth)
            for i,result in enumerate(estimates[1:],start=1):
                rows[i].extend(rescore_visible_rows(truth,landmarks,result,record['calibration'],mask.shape[0]))
            if frame_index%20==0:print(f'{name}: paired replay {frame_index+1}/110',flush=True)
        for i,(variant,resize_first,key,use_roi,pyramid) in enumerate(variants):
            ms=np.asarray(timings[i][5:]);prep=np.asarray(prep_times[i][5:]);score=scorer(rows[i])
            availability=sum(f['state']=='tracking' for f in traces[i])/len(traces[i])
            score['allFrameAvailability']=availability
            score['nativeMedianErrorPixels']=None if score.get('medianErrorPixels') is None else score['medianErrorPixels']*2
            score['nativeP95ErrorPixels']=None if score.get('p95ErrorPixels') is None else score['p95ErrorPixels']*2
            result=dict(variant=variant,name=name,objectId=manifest['objectId'],height=height,features=features,
                referenceFeatures=len(points),resizeBeforeGrayscale=resize_first,lshKeySize=key,objectRegionSearch=use_roi,orbPyramidLevels=pyramid,
                medianMs=float(np.median(ms)),p95Ms=float(np.percentile(ms,95)),meanMs=float(ms.mean()),
                preparationMedianMs=float(np.median(prep)),visionOnlyMedianEquivalentFps=float(1000/np.median(ms)),
                framesAbove33ms=int(np.sum(ms>1000/30)),measuredFrames=len(ms),evaluationFrames=len(traces[i]),
                visionBudget20msPassed=bool(np.percentile(ms,95)<=20),score=score,
                qualityAndSpeedGatePassed=bool(np.percentile(ms,95)<=20 and score['attachmentGatePassed'] and availability>=.9),
                frameTimingMs=timings[i],frames=traces[i])
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('frames','frameTimingMs')}),flush=True)
        save(output,results)


def save(output,results):
    for result in results:
        result['latencyByTrackingState']={}
        pairs=list(zip(result['frameTimingMs'][5:],result['frames'][5:]))
        for state in ('tracking','limited','lost'):
            samples=[t for t,f in pairs if f['state']==state]
            result['latencyByTrackingState'][state]=dict(samples=len(samples),
                medianMs=float(np.median(samples)) if samples else None,
                p95Ms=float(np.percentile(samples,95)) if samples else None)
    report=dict(schemaVersion=1,targetLiveFps=30,phonePerformanceValidated=False,
        environment=dict(platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
                         numpy=np.__version__,opencv=cv2.__version__,openCVThreads=1),
        latencyScope='Single-thread CPU resize + grayscale + tracker; five warmup frames excluded. Decode, evaluation, rendering and camera excluded.',
        comparison='Same decoded input and common first-ten-frame annotated reference bank; independent tracker states; rotated candidate order.',
        accuracyScope='All 110 held-out frames per object, native-HD scoring; evaluation annotations never enter trackers.',
        timingLimit='Short recorded clips on a desktop, not sustained live camera or phone throughput. All-state timing includes failed tracking; inspect state-specific timings and sample counts.',objects=results)
    write_artifact(output/'report.json',json.dumps(report,indent=2)+'\n')
    def error_cell(r):
        value=r['score']['nativeP95ErrorPixels']
        return '—' if value is None else f'{value:.1f}'
    table=''.join(f'<tr><td>{r["name"]}</td><td>{r["variant"]}</td><td>{r["medianMs"]:.2f}</td><td>{r["p95Ms"]:.2f}</td><td>{r["preparationMedianMs"]:.2f}</td><td>{r["score"]["allFrameAvailability"]:.1%}</td><td>{error_cell(r)}</td><td>{r["qualityAndSpeedGatePassed"]}</td></tr>' for r in results)
    html='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Higher FPS experiment</title><style>body{font:16px system-ui;background:#101b21;color:#e6eff4;margin:28px}p{max-width:1000px;line-height:1.5}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #425966}a{color:#70ebd2}</style><h1>Higher FPS: paired vision experiment</h1><p>Same tracking resolution and feature count for all variants. Resize-first converts only the smaller image to grayscale. Key16 searches fewer LSH buckets, potentially sacrificing matches. Object-region searches near the previous tracked features, expands after loss, then returns to full-image search after three failures. All quality gates still apply.</p><p>Goal: 30 FPS live, with a provisional 20 ms P95 vision budget within the 33.3 ms total frame budget. These are desktop CPU vision timings. Camera, rendering, decoding and sustained phone performance remain untested.</p><table><tr><th>Object</th><th>Candidate</th><th>Median ms</th><th>P95 ms</th><th>Prep median ms</th><th>Availability</th><th>Native P95 error px</th><th>Quality + speed passed</th></tr>'+table+'</table><p><a href="report.json">Full traces and measurement scope</a> · <a href="../visual.html">Watch mask and mapping</a></p>'
    if any(r['variant'].startswith('pyramid-') for r in results):
        start=html.index('<p>Same tracking resolution')
        end=html.index('</p>',start)+4
        html=html[:start]+'<p>Baseline: eight feature-pyramid levels. Candidates: four or three levels, computing fewer scales during feature detection. All retain 480 × 360 images, a 400-feature cap and the same onboarding reference bank. Median timings include lost tracking frames; accepted/lost-state timings and sample counts are in the full report. Low tracking availability prevents these speed results from establishing usable live AR.</p>'+html[end:]
    write_artifact(output/'index.html',html)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache'/'hope-hd')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'video-hd'/'performance-higher-fps')
    parser.add_argument('--height',type=int,default=360)
    parser.add_argument('--features',type=int,default=400)
    parser.add_argument('--roi',action='store_true',help='Compare baseline against expanding object-region feature detection.')
    parser.add_argument('--levels',action='store_true',help='Compare baseline against fewer ORB pyramid levels at the same resolution and feature count.')
    args=parser.parse_args()
    if not 240<=args.height<=1440 or args.features<200:parser.error('Height 240–1440 and at least 200 features required.')
    if args.roi and args.levels:parser.error('Select one comparison mode.')
    run(args.assets,args.output,args.height,args.features,args.roi,args.levels)
