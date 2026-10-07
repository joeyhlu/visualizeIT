"""Experimental image segmentation and conservative hand-core exclusion.

Hand landmarks are not a pixel-accurate hand mask. Seed motion is independent
2D image flow; neither a segmentation nor this flow establishes a 6D pose.
"""
import hashlib
import numpy as np
from .vision import cv2
from .surface_tracker import follow
from .learned_mask_probe import runtime
from .ycb_assets import ROOT

def paths(mask):
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    return [dict(points=cv2.approxPolyDP(c,1.,True).reshape(-1,2).tolist(),hole=bool(hierarchy[0,i,3]>=0))
            for i,c in enumerate(contours) if cv2.contourArea(c)>=3]

def hand_core(shape,hands):
    mask=np.zeros(shape,np.uint8)
    for landmarks in hands:
        p=np.array([[l.x*shape[1],l.y*shape[0]] for l in landmarks])
        width=np.linalg.norm(p[5]-p[17]);radius=max(2,int(width*.055))
        cv2.fillConvexPoly(mask,np.round(p[[0,1,5,9,13,17]]).astype(np.int32),255)
        for chain in ([1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]):
            for a,b in zip(chain,chain[1:]):cv2.line(mask,tuple(np.round(p[a]).astype(int)),tuple(np.round(p[b]).astype(int)),255,2*radius)
    return mask

class LearnedForeground:
    def __init__(self,gray,geometry):
        self.mp,self.segmenter=runtime()
        path=ROOT/'.cache/show3d/models/hand_landmarker-v1.task'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1':raise ValueError('Hand model hash mismatch.')
        options=self.mp.tasks.vision.HandLandmarkerOptions(base_options=self.mp.tasks.BaseOptions(model_asset_path=str(path)),num_hands=2)
        self.hands=self.mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.previous=gray;self.mask=geometry.copy()
        self.points=self.corners(gray,geometry);self.reason=None
        d=cv2.distanceTransform(geometry,cv2.DIST_L2,5);y,x=np.unravel_index(d.argmax(),d.shape)
        self.seed=np.array([x,y],float)

    @staticmethod
    def corners(gray,mask):
        interior=cv2.erode(mask,np.ones((3,3),np.uint8))
        points=cv2.goodFeaturesToTrack(gray,120,.005,4,mask=interior)
        return np.empty((0,2),np.float32) if points is None else points.reshape(-1,2)

    def update(self,gray):
        rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
        hands=self.hands.detect(image).hand_landmarks;excluded=hand_core(gray.shape,hands)
        current,good=follow(self.previous,gray,self.points,1.5)
        affine=None;support=0
        if good.sum()>=4:
            old=self.points[good];new=current[good]
            xy=np.round(new).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
            visible=excluded[xy[:,1],xy[:,0]]==0;old=old[visible];new=new[visible]
            if len(old)>=4:
                affine,inliers=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=3)
                support=0 if inliers is None else int(inliers.sum())
        stats=dict(handDetections=len(hands),handCoreContours=paths(excluded),promptFlowInliers=support,
                   maskScope='MagicTouch v1 image prediction minus approximate landmark hand cores; not a verified pixel hand mask or pose.')
        self.previous=gray
        if affine is None or support<4:
            self.points=current[good];stats['maskReason']='prompt_motion_unreliable'
            return None,stats
        predicted=cv2.warpAffine(self.mask,affine,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_NEAREST)
        predicted[excluded>0]=0
        old_area=max((self.mask>0).sum(),1)
        # Keep private state in the current image coordinates on rejected masks.
        # It is never returned as a visible prediction without new model evidence.
        self.mask=predicted;self.points=current[good]
        d=cv2.distanceTransform(predicted,cv2.DIST_L2,5)
        if d.max()<3:
            stats['maskReason']='no_unoccluded_prompt';return None,stats
        y,x=np.unravel_index(d.argmax(),d.shape);self.seed=np.array([x,y],float)
        from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
        roi_type=self.mp.tasks.vision.InteractiveSegmenterLegacyRegionOfInterest
        roi=roi_type(roi_type.Format.KEYPOINT,NormalizedKeypoint(x/gray.shape[1],y/gray.shape[0]))
        probability=self.segmenter.segment(image,roi).confidence_masks[-1].numpy_view().copy()
        mask=((probability.squeeze()>.5).astype(np.uint8)*255);mask[excluded>0]=0
        # Retain only the prompted component. Reject large unexplained jumps.
        _,labels,areas,_=cv2.connectedComponentsWithStats(mask)
        label=labels[y,x]
        if label==0 or areas[label,cv2.CC_STAT_AREA]<100:
            stats['maskReason']='prompt_not_in_foreground';return None,stats
        mask=(labels==label).astype(np.uint8)*255
        ratio=float((mask>0).sum()/old_area)
        if not .45<=ratio<=2.0:
            stats.update(maskReason='foreground_size_jump',maskAreaRatio=ratio);return None,stats
        self.mask=mask;self.points=self.corners(gray,mask)
        stats.update(maskReason=None,visibleMaskContours=paths(mask),seedPixels=[int(x),int(y)],maskAreaRatio=ratio)
        return mask,stats

    def close(self):
        self.segmenter.close();self.hands.close()

