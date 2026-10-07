"""Full consecutive 60 FPS source windows; offline learned-mask predictions."""
import json
import time
import numpy as np
from .vision import cv2
from .retry_experiment import onboarding
from .glb_model import read_glb
from .surface_tracker import SurfaceRays
from .learned_foreground import LearnedForeground
from .hd_experiment import tracking_input
from .storage import write_artifact
from .ycb_assets import ROOT

def run():
    cv2.setNumThreads(1);root=ROOT/'.cache/show3d';output=ROOT/'artifacts/video-60/learned-mask'
    reports=[]
    for alias in ('mug','ranch','keyboard'):
        original=json.loads((ROOT/f'artifacts/video-60/{alias}/report.json').read_text())
        calibration=json.loads((root/'scenes'/original['scene']/'camera_calibration/headset0.json').read_text());calibration['cam_K']=original['cameraCalibration']
        import hashlib
        for name,digest in original['sourceHashes'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source hash mismatch.')
        mesh=read_glb(root/f'models/obj_{original["modelId"]:06d}.glb',alias)
        capture=cv2.VideoCapture(str(root/'scenes'/original['scene']/'headset0.mp4'))
        refs,pose,_=onboarding(capture,root,original,mesh,calibration,720)
        k=np.array(original['cameraCalibration']).reshape(3,3).copy();k[:2]*=720/original['nativeResolution'][1]
        geometry=SurfaceRays(mesh).silhouette(refs[2].shape,k,*pose)
        tracker=LearnedForeground(refs[2],geometry);capture.set(cv2.CAP_PROP_POS_FRAMES,original['sourceStart']+10)
        frames=[];timings=[]
        try:
            for offset in range(240):
                ok,bgr=capture.read()
                if not ok:raise ValueError('Missing evaluation frame.')
                begin=time.perf_counter();gray,_=tracking_input(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),calibration,720)
                mask,stats=tracker.update(gray);timings.append((time.perf_counter()-begin)*1000)
                frames.append(dict(frameId=original['sourceStart']+10+offset,state='mask' if mask is not None else 'unavailable',**stats))
                if offset in (0,30,60,120,239):
                    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);overlay=rgb.copy()
                    if mask is not None:overlay[mask>0]=(overlay[mask>0]*.45+np.array([180,240,20])*.55).astype(np.uint8)
                    image=cv2.resize(np.hstack((rgb,overlay)),(576,360))
                    cv2.putText(image,f'{alias} frame +{offset} | {stats["maskReason"] or "prediction"}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,120,255),1)
                    ok,encoded=cv2.imencode('.jpg',image)
                    if not ok:raise ValueError('Diagnostic image encoding failed.')
                    write_artifact(output/f'{alias}-full-{offset}.jpg',encoded.tobytes())
                if offset%40==0:print(alias,offset,frames[-1]['state'],stats['maskReason'],flush=True)
        finally:tracker.close();capture.release()
        timing=np.array(timings[5:]);report=dict(name=alias,trackingHeight=720,frames=frames,timingsMs=timings,
            maskAvailability=sum(f['state']=='mask' for f in frames)/240,medianMs=float(np.median(timing)),p95Ms=float(np.percentile(timing,95)),
            scope='Offline segmentation + hand landmarks + prompt optical flow; all 240 frames. Mask availability is not accuracy or pose availability. No evaluation poses enter this algorithm. Five timing warmups, initialization/decode excluded; grayscale conversion/resize included.',
            model='MagicTouch v1 float32 legacy + hand landmarker v1, MediaPipe 1.0.1; all inference local CPU',liveFpsValidated=False)
        write_artifact(output/f'{alias}-full.json',json.dumps(report,indent=2));reports.append(report)
        write_artifact(output/'full-report.json',json.dumps(dict(objects=reports),indent=2))
        print(alias,report['maskAvailability'],report['medianMs'],report['p95Ms'],flush=True)

if __name__=='__main__':run()
