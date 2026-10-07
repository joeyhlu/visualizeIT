"""Experimental image-derived foreground inside a predicted object region.

GrabCut uses grayscale appearance and geometric initialization. It is not a
semantic hand detector and can fail when hand/object appearances are similar.
"""
import numpy as np
from .vision import cv2


class VisibleMask:
    def __init__(self):
        self.foreground_hist=None;self.background_hist=None

    def estimate(self,gray,geometry):
        positions=np.argwhere(geometry>0)
        if len(positions)<100:return None,dict(maskReason='insufficient_projected_area')
        low=np.maximum(positions.min(axis=0)-22,0);high=np.minimum(positions.max(axis=0)+23,gray.shape)
        y0,x0=low;y1,x1=high
        image=gray[y0:y1,x0:x1];region=geometry[y0:y1,x0:x1]
        core=cv2.erode(region,np.ones((7,7),np.uint8))>0
        near=cv2.dilate(region,np.ones((17,17),np.uint8))>0
        outside=(region==0)&near
        if core.sum()<20 or outside.sum()<30:return None,dict(maskReason='insufficient_appearance_seeds')
        def histogram(values):
            counts=np.bincount((values//8).ravel(),minlength=32).astype(float)+1
            counts=np.convolve(counts,[.25,.5,.25],mode='same')
            return counts/counts.sum()
        if self.foreground_hist is None:
            self.foreground_hist=histogram(image[core]);self.background_hist=histogram(image[outside])
        bins=image//8
        odds=np.log(self.foreground_hist[bins]/self.background_hist[bins])
        threshold=max(.15,float(np.percentile(odds[core],70)))
        sure_foreground=core&(odds>=threshold)
        if sure_foreground.sum()<8:return None,dict(maskReason='ambiguous_object_background_appearance')
        labels=np.full(image.shape,cv2.GC_PR_BGD,np.uint8)
        labels[region>0]=cv2.GC_PR_FGD
        labels[~near]=cv2.GC_BGD
        labels[core&(odds<-.8)]=cv2.GC_BGD
        labels[sure_foreground]=cv2.GC_FGD
        try:
            cv2.grabCut(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),labels,None,
                        np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
        except cv2.error:return None,dict(maskReason='grabcut_failed')
        foreground=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.uint8)*255
        foreground[~near]=0
        fraction=float((foreground>0).sum()/max((region>0).sum(),1))
        if not .3<=fraction<=1.4:return None,dict(maskReason='foreground_area_inconsistent',maskAreaRatio=fraction)
        result=np.zeros_like(gray);result[y0:y1,x0:x1]=foreground
        contours,hierarchy=cv2.findContours(result,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        paths=[dict(points=cv2.approxPolyDP(c,1.,True).reshape(-1,2).tolist(),hole=bool(hierarchy[0,i,3]>=0))
               for i,c in enumerate(contours) if cv2.contourArea(c)>=3]
        return result,dict(maskReason=None,maskAreaRatio=fraction,visibleMaskContours=paths,
            maskScope='Image-derived GrabCut foreground, geometry-seeded; not semantic hand segmentation or a ground-truth mask.')
