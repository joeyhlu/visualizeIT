"""Image-only interactive segmentation probe; never consumes evaluation poses."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from .vision import cv2
from .glb_model import read_glb
from .retry_experiment import onboarding
from .surface_tracker import SurfaceRays,follow
from .hd_experiment import tracking_input
from .ycb_assets import ROOT
from .storage import write_artifact

MODEL_HASH='e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25'

def runtime():
    sys.path.insert(0,str(ROOT/'.cache'/'segmentation'))
    import mediapipe as mp
    path=ROOT/'.cache/show3d/models/magic_touch-v1.tflite'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_HASH:raise ValueError('Segmentation model hash mismatch.')
    task=mp.tasks.vision.InteractiveSegmenterLegacy.create_from_options(mp.tasks.vision.InteractiveSegmenterLegacyOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path))))
    return mp,task

def run():
    cv2.setNumThreads(1);mp,task=runtime()
    output=ROOT/'artifacts/video-60/learned-mask';output.mkdir(exist_ok=True)
    records=[]
    try:
        for alias in ('mug','ranch','keyboard'):
            report=json.loads((ROOT/f'artifacts/video-60/{alias}/report.json').read_text())
            root=ROOT/'.cache/show3d';prefix=root/'scenes'/report['scene']
            calibration=json.loads((prefix/'camera_calibration/headset0.json').read_text());calibration['cam_K']=report['cameraCalibration']
            mesh=read_glb(root/f'models/obj_{report["modelId"]:06d}.glb',alias)
            capture=cv2.VideoCapture(str(prefix/'headset0.mp4'))
            refs,pose,_=onboarding(capture,root,report,mesh,calibration,720)
            previous=refs[2]
            k=np.array(report['cameraCalibration']).reshape(3,3).copy();k[:2]*=720/report['nativeResolution'][1]
            geometry=SurfaceRays(mesh).silhouette(previous.shape,k,*pose)
            # A single auto seed for this probe, chosen only from onboarding geometry.
            # This can land on an occluder: inspect the result, do not assume success.
            distance=cv2.distanceTransform(geometry,cv2.DIST_L2,5);y,x=np.unravel_index(distance.argmax(),distance.shape)
            pixels=np.array([[x,y]],np.float32)
            capture.set(cv2.CAP_PROP_POS_FRAMES,report['sourceStart']+10)
            panels=[]
            for offset in range(121):
                ok,bgr=capture.read()
                if not ok:raise ValueError('Missing source frame.')
                gray,_=tracking_input(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),calibration,720)
                tracked,valid=follow(previous,gray,pixels,2.)
                if not valid.all():
                    records.append(dict(object=alias,offset=offset,state='seed_flow_lost'));break
                pixels=tracked;previous=gray
                if offset not in (0,30,60,120):continue
                rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                t=time.perf_counter()
                roi_type=mp.tasks.vision.InteractiveSegmenterLegacyRegionOfInterest
                from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
                keypoint=NormalizedKeypoint(float(pixels[0,0]/gray.shape[1]),float(pixels[0,1]/gray.shape[0]))
                roi=roi_type(roi_type.Format.KEYPOINT,keypoint)
                result=task.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),roi)
                probability=result.confidence_masks[-1].numpy_view().copy();elapsed=(time.perf_counter()-t)*1000
                selected=probability>.5
                if selected.ndim==3:selected=selected[:,:,0]
                mask=(selected.astype(np.uint8)*255)
                ok,encoded=cv2.imencode('.png',mask)
                if not ok:raise ValueError('Mask encoding failed.')
                write_artifact(output/f'{alias}-{offset}-mask.png',encoded.tobytes())
                overlay=rgb.copy();overlay[selected]=(overlay[selected]*.45+np.array([20,240,180])*.55).astype(np.uint8)
                cv2.circle(overlay,tuple(np.round(pixels[0]).astype(int)),5,(255,70,30),-1)
                pair=np.hstack((rgb,overlay));pair=cv2.resize(pair,(576,360))
                cv2.putText(pair,f'{alias} +{offset}/60 s | {elapsed:.0f} ms',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,100,30),1)
                panels.append(pair)
                records.append(dict(object=alias,offset=offset,seed=pixels[0].tolist(),milliseconds=elapsed,maskArea=int(selected.sum()),state='image_model_probe_not_pose_tracking'))
                print(records[-1],flush=True)
            capture.release()
            if panels:
                ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(np.vstack(panels),cv2.COLOR_RGB2BGR))
                if not ok:raise ValueError('Probe image encoding failed.')
                write_artifact(output/f'{alias}-probe.jpg',encoded.tobytes())
    finally:task.close()
    write_artifact(output/'probe.json',json.dumps(dict(modelSha256=MODEL_HASH,model='MediaPipe MagicTouch v1 float32 (legacy fallback)',sdk='1.0.1',fallbackReason='The released Windows 1.0.1 DLL lacks MpInteractiveSegmenterCreate required by the new v2 Python API.',scope='Independent images plus optical-flowed onboarding seed; no future reference poses; no manual mask labels or IoU accuracy claims.',frames=records),indent=2))

if __name__=='__main__':run()
