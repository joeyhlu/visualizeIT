"""Experimental bounded ORB/LSH tracker; quality gates still apply."""
import numpy as np
from .tracker import RigidTracker
from .vision import cv2


class FastRigidTracker(RigidTracker):
    def __init__(self,*args,**kwargs):
        feature_limit=kwargs.pop('feature_limit',800)
        index_key_size=kwargs.pop('index_key_size',12)
        self.use_roi=kwargs.pop('use_roi',False)
        feature_levels=kwargs.pop('feature_levels',8)
        if not 1<=feature_levels<=8:raise ValueError('ORB pyramid levels must be between 1 and 8.')
        if not 8<=index_key_size<=20:raise ValueError('LSH key size must be between 8 and 20.')
        super().__init__(*args,**kwargs)
        self.search_pixels=None if self.previous_pixels is None else self.previous_pixels.copy()
        self.search_failures=0
        self.orb=cv2.ORB_create(nfeatures=feature_limit,nlevels=feature_levels)
        cv2.setRNGSeed(17)
        self.index=cv2.FlannBasedMatcher(dict(algorithm=6,table_number=6,key_size=index_key_size,multi_probe_level=1),{})
        self.index.add([self.descriptors]);self.index.train()

    def match(self,gray):
        x0,y0,x1,y1=self.search_region(gray.shape)
        features,descriptors=self.orb.detectAndCompute(gray[y0:y1,x0:x1],None)
        if descriptors is None or len(descriptors)<2:return np.empty((0,2)),np.empty((0,3))
        nearest=self.index.knnMatch(descriptors,k=2)
        # The static reference index is searched once. Require a ratio margin,
        # bounded Hamming distance and one observation per reference feature.
        owners={}
        for pair in nearest:
            if len(pair)!=2:continue
            a,b=pair
            if a.distance>=.75*b.distance or a.distance>64:continue
            if a.trainIdx not in owners or a.distance<owners[a.trainIdx].distance:owners[a.trainIdx]=a
        matches=list(owners.values())
        return (np.array([features[m.queryIdx].pt for m in matches]).reshape(-1,2)+[x0,y0],
                self.reference[[m.trainIdx for m in matches]])

    def search_region(self,shape):
        h,w=shape[:2]
        if not self.use_roi or self.search_pixels is None or len(self.search_pixels)<12 or self.search_failures>=3:
            return 0,0,w,h
        low=self.search_pixels.min(axis=0);high=self.search_pixels.max(axis=0)
        # Include a border for ORB and motion; grow after loss, then search globally.
        padding=np.maximum((high-low)*(.5+self.search_failures*.5),40)
        low=np.maximum(np.floor(low-padding).astype(int),[0,0])
        high=np.minimum(np.ceil(high+padding).astype(int)+1,[w,h])
        if (high-low<80).any():return 0,0,w,h
        return int(low[0]),int(low[1]),int(high[0]),int(high[1])

    def update(self,rgb,k,frame_id):
        result=super().update(rgb,k,frame_id)
        if result.state=='tracking':
            self.search_pixels=self.previous_pixels.copy();self.search_failures=0
        else:self.search_failures+=1
        return result

    def flow(self,gray):
        if self.previous_pixels is not None and len(self.previous_pixels)>240:
            # Keep a spatially distributed subset instead of just the earliest corners.
            order=np.lexsort((self.previous_pixels[:,1],self.previous_pixels[:,0]))
            indices=order[np.linspace(0,len(order)-1,240,dtype=int)]
            self.previous_pixels=self.previous_pixels[indices]
            self.previous_points=self.previous_points[indices]
        return super().flow(gray)
