"""Diagnostic texture-detail evidence; no pose acceptance or benchmark labels."""
import numpy as np
from .vision import cv2


def detail_metrics(rgb, color, visible):
    image=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
    model=cv2.cvtColor(color.astype(np.float32),cv2.COLOR_RGB2GRAY)
    # Remove slow shading gradients, and keep away from silhouettes/occluders.
    support=cv2.distanceTransform(visible.astype(np.uint8),cv2.DIST_L2,5)>5
    if support.sum()<100:return dict(state='insufficient_interior',pixels=int(support.sum()))
    image_detail=image-cv2.GaussianBlur(image,(0,0),2.)
    model_detail=model-cv2.GaussianBlur(model,(0,0),2.)
    a,b=image_detail[support],model_detail[support]
    result=dict(state='measured',pixels=int(support.sum()),
        image_detail_std=float(a.std()),model_detail_std=float(b.std()))
    result['detail_correlation']=None if min(a.std(),b.std())<.005 else float(np.corrcoef(a,b)[0,1])
    local=[];h,w=image.shape
    for y in range(0,h-23,12):
        for x in range(0,w-23,12):
            keep=support[y:y+24,x:x+24]
            if keep.sum()<24*24*.7:continue
            aa=image_detail[y:y+24,x:x+24][keep];bb=model_detail[y:y+24,x:x+24][keep]
            if min(aa.std(),bb.std())<.005:continue
            local.append(float(np.corrcoef(aa,bb)[0,1]))
    result.update(local_patch_count=len(local),
        local_detail_median=float(np.median(local)) if local else None)
    return result


def axial_pose(seed,axis,center,degrees):
    """Rotate around the fixed model's long axis while preserving its camera center."""
    from .quality_contract import checked_pose
    seed=checked_pose(seed);axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    rotation=cv2.Rodrigues(axis*np.radians(degrees))[0]
    pose=seed.copy();pose[:3,:3]=seed[:3,:3]@rotation
    pose[:3,3]=seed[:3,3]+seed[:3,:3]@center-pose[:3,:3]@center
    return checked_pose(pose)
