"""Measure tracking-input resize and reference-bank compaction against HD poses.

Target: 30 FPS eventually, reserving a provisional 20 ms P95 CPU vision budget.
This is desktop RGB replay, not a live phone FPS claim.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .hd_experiment import image_for, references, tracking_input, annotations
from .ply_model import read_ply
from .scan_mapping import unwrap
from .tracker import RigidTracker
from .storage import write_artifact
from . import storage
from .video_experiment import scorer
from .ycb_assets import ROOT
from .vision import cv2
from .fast_tracker import FastRigidTracker


def compact_bank(points,descriptors,maximum=1500):
    # Keep two appearances per 2 mm surface cell, then distribute across the bank.
    cells=np.floor(points/.002).astype(int);counts={};selected=[]
    for index,cell in enumerate(cells):
        key=tuple(cell);count=counts.get(key,0)
        if count<2:selected.append(index);counts[key]=count+1
    selected=np.asarray(selected,dtype=int)
    if len(selected)>maximum:selected=selected[np.linspace(0,len(selected)-1,maximum,dtype=int)]
    return points[selected],descriptors[selected]


def run(root,output,height=480,compact=False,threads=1,fast=False,gray=False,features=800,resize_first=False,index_key_size=12,roi=False,levels=8):
    cv2.setNumThreads(threads)
    output.mkdir(parents=True,exist_ok=True);reports=[]
    # Decode outside the measured block; keep only one native frame in memory.
    for path in sorted(root.glob('object-*.json')):
        manifest=json.loads(path.read_text());name=manifest['name']
        scan,vertices,transform=read_ply(root/manifest['model']['path'],name)
        mesh=unwrap(scan,name)[0]
        refs=references(root,manifest['frames'][:10],mesh,transform,height,gray)
        points,descriptors=compact_bank(refs[0],refs[1]) if compact else refs[:2]
        tracker=FastRigidTracker(points,descriptors,vertices,*refs[2:],feature_limit=features,index_key_size=index_key_size,use_roi=roi,feature_levels=levels) if fast else RigidTracker(points,descriptors,vertices,*refs[2:])
        component_samples={'match':[],'flow':[],'solve':[]};current_components={}
        def measured_call(function,name):
            def call(*args):
                start=time.perf_counter();value=function(*args)
                current_components[name]=time.perf_counter()-start
                return value
            return call
        for component in component_samples:setattr(tracker,component,measured_call(getattr(tracker,component),component))
        records=manifest['frames'][10:]
        indices=np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int)
        landmarks=vertices[indices];normals=scan.normals[indices]@transform[:3,:3]
        source_triangles=scan.triangles[:,[0,2,1]];times=[];rows=[];trace=[]
        for frame_index,record in enumerate(records):
            rgb=image_for(root,record)
            current_components.clear()
            start=time.perf_counter();working,k=tracking_input(rgb,record['calibration'],height,gray,resize_first)
            result=tracker.update(working,k,record['frameId']);times.append(time.perf_counter()-start)
            if frame_index>=5:
                for component in component_samples:component_samples[component].append(current_components.get(component,0)*1000)
            trace.append(result.manifest())
            rows.extend(annotations(record,landmarks,normals,vertices,source_triangles,
                                    image_for(root,record,'evaluationMask'),result))
            if frame_index%20==0:print(f'{name}: speed replay {frame_index+1}/110',flush=True)
        # Exclude five warmup frames only from latency; all 110 frames remain in accuracy/availability.
        measured=np.asarray(times[5:])*1000;score=scorer(rows)
        score['allFrameAvailability']=sum(r['state']=='tracking' for r in trace)/len(trace)
        score['nativeMedianErrorPixels']=score.get('medianErrorPixels')*2 if score.get('medianErrorPixels') is not None else None
        score['nativeP95ErrorPixels']=score.get('p95ErrorPixels')*2 if score.get('p95ErrorPixels') is not None else None
        result=dict(name=name,objectId=manifest['objectId'],trackingHeight=height,referenceFeatures=len(points),compactBank=compact,
            openCVThreads=cv2.getNumThreads(),grayscaleInput=gray,resizeBeforeGrayscale=resize_first,objectRegionSearch=roi,
            lshKeySize=index_key_size if fast else None,orbPyramidLevels=levels if fast else 8,
            matchingBackend=f'Fixed ORB/LSH index (key {index_key_size}), {features} image features, 240 flow points' if fast else 'Mutual brute-force Hamming, 2000 image features',
            componentTimingMeanMs={name:float(np.mean(values)) for name,values in component_samples.items()},
            evaluationFrames=len(trace),score=score,trackingInputAndVisionMedianMs=float(np.median(measured)),
            trackingInputAndVisionP95Ms=float(np.percentile(measured,95)),trackingInputAndVisionMeanMs=float(measured.mean()),
            visionBudget20msPassed=bool(np.percentile(measured,95)<=20),frames=trace,
            qualityAndSpeedGatePassed=bool(np.percentile(measured,95)<=20 and score.get('attachmentGatePassed') and score['allFrameAvailability']>=.9),
            latencyScope='CPU image resize + tracker, five warmup frames excluded; excludes decode, evaluation, render, camera acquisition',
            accuracyScope='Every held-out native-HD frame; independent poses/masks/mesh visibility; errors also in 720p-equivalent pixels',
            thresholdScope='Existing 3 px RANSAC and 1 px forward/backward thresholds at working resolution',
            targetLiveFps=30,phonePerformanceValidated=False)
        reports.append(result);print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)
        write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1,objects=reports),indent=2)+'\n')
    rows=''.join(f'<tr><td>{r["name"]}</td><td>{r["referenceFeatures"]}</td><td>{r["trackingInputAndVisionMedianMs"]:.1f} ms</td><td>{r["trackingInputAndVisionP95Ms"]:.1f} ms</td><td>{r["score"]["allFrameAvailability"]:.1%}</td><td>{r["score"].get("nativeP95ErrorPixels")}</td><td>{r["qualityAndSpeedGatePassed"]}</td></tr>' for r in reports)
    write_artifact(output/'index.html','<!doctype html><meta charset="utf-8"><title>30 FPS vision benchmark</title><style>body{font:17px system-ui;margin:30px}td,th{padding:12px;border-bottom:1px solid #ddd}p{max-width:900px}</style><h1>30 FPS target: CPU vision replay</h1><p>Smaller tracking inputs, full-resolution evaluation. A 20 ms P95 vision budget reserves part of the 33.3 ms frame budget for camera, GPU rendering and display. This is desktop replay timing, not live phone FPS. All 110 held-out frames count toward accuracy and availability. Both speed and attachment gates must pass.</p><table><tr><th>Object</th><th>Reference descriptors</th><th>Median</th><th>P95</th><th>Availability</th><th>Native P95 error</th><th>Quality + speed passed</th></tr>'+rows+'</table><p><a href="report.json">Settings, frame traces and full report</a></p>')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache'/'hope-hd')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts'/'video-hd'/'performance-480')
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--fast',action='store_true',help='Test bounded ORB/LSH index matching rather than exhaustive matching.')
    parser.add_argument('--gray',action='store_true',help='Convert once to luminance before resizing; a phone can supply its luminance plane directly.')
    parser.add_argument('--features',type=int,default=800)
    parser.add_argument('--resize-first',action='store_true',help='With --gray, resize RGB before converting the smaller image to grayscale.')
    parser.add_argument('--lsh-key-size',type=int,default=12,help='With --fast, LSH key size; larger keys reduce buckets searched and may lose matches.')
    parser.add_argument('--roi',action='store_true',help='With --fast, detect near previously tracked object features; expand to global search after loss.')
    parser.add_argument('--levels',type=int,default=8,help='With --fast, ORB pyramid levels, 1–8. Fewer scales reduce detection work but may lose scale robustness.')
    parser.add_argument('--reserve-mib',type=int,default=512)
    parser.add_argument('--cv-threads',type=int,default=1,help='Bound OpenCV thread count to reduce scheduling variance.')
    args=parser.parse_args()
    if not 240<=args.height<=1440:parser.error('Tracking height must be between 240 and 1440.')
    if args.cv_threads<1:parser.error('Thread count must be positive.')
    if args.features<200:parser.error('At least 200 detection features required.')
    if args.resize_first and not args.gray:parser.error('--resize-first requires --gray.')
    if not 8<=args.lsh_key_size<=20:parser.error('LSH key size must be between 8 and 20.')
    if args.lsh_key_size!=12 and not args.fast:parser.error('Non-default LSH key size requires --fast.')
    if args.roi and not args.fast:parser.error('--roi requires --fast.')
    if not 1<=args.levels<=8:parser.error('ORB pyramid levels must be between 1 and 8.')
    if args.levels!=8 and not args.fast:parser.error('Non-default pyramid levels require --fast.')
    if args.reserve_mib<128:parser.error('Retain at least 128 MiB disk reserve.')
    storage.RESERVE=args.reserve_mib*1024*1024
    run(args.assets,args.output,args.height,args.compact,args.cv_threads,args.fast,args.gray,args.features,args.resize_first,args.lsh_key_size,args.roi,args.levels)
