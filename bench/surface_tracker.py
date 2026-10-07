"""Quality-first rigid tracking with bounded occlusion recovery.

Only the onboarding pose and camera images enter this tracker. Projected
geometry is a search region, not a predicted foreground/hand segmentation.
"""
import numpy as np
from .vision import cv2
from .fast_tracker import FastRigidTracker
from .tracker import TrackingResult


class SurfaceRays:
    def __init__(self,mesh):
        self.model=mesh.positions
        self.faces=mesh.triangles
        triangles=mesh.positions[mesh.triangles]
        self.a=triangles[:,0];self.e1=triangles[:,1]-self.a;self.e2=triangles[:,2]-self.a
        self.centres=triangles.mean(axis=1);self.face_normals=np.cross(self.e1,self.e2)

    def silhouette(self,shape,k,rotation,translation):
        eye=-rotation.T@translation
        pixels=projected(self.model,rotation,translation,k)
        visible=np.sum(self.face_normals*(eye-self.centres),axis=1)>0
        mask=np.zeros(shape,np.uint8)
        for face in self.faces[visible]:cv2.fillConvexPoly(mask,np.round(pixels[face]).astype(np.int32),255)
        return mask

    def intersect(self,pixels,k,rotation,translation):
        """Nearest triangle hit after calibrated image-space broad-phase culling."""
        eye=-rotation.T@translation
        rays=np.column_stack(((pixels[:,0]-k[0,2])/k[0,0],(pixels[:,1]-k[1,2])/k[1,1],np.ones(len(pixels))))@rotation
        projected_vertices=projected(self.model,rotation,translation,k)[self.faces]
        low=projected_vertices.min(axis=1)-1e-5;high=projected_vertices.max(axis=1)+1e-5
        points=np.full((len(pixels),3),np.nan)
        for i,(pixel,direction) in enumerate(zip(pixels,rays)):
            selected=np.flatnonzero((low[:,0]<=pixel[0])&(high[:,0]>=pixel[0])&(low[:,1]<=pixel[1])&(high[:,1]>=pixel[1]))
            if not len(selected):continue
            e1=self.e1[selected];e2=self.e2[selected];offset=eye-self.a[selected]
            p=np.cross(direction,e2);det=np.sum(e1*p,axis=1)
            inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-10)
            q=np.cross(offset,e1)
            u=np.sum(offset*p,axis=1)*inv
            v=np.sum(direction*q,axis=1)*inv
            distance=np.sum(e2*q,axis=1)*inv
            valid=(np.abs(det)>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(distance>.03)
            hit=np.where(valid,distance,np.inf).min()
            if np.isfinite(hit):points[i]=eye+direction*hit
        return points


def follow(previous,current,pixels,limit=1.):
    if pixels is None or not len(pixels):return np.empty((0,2)),np.zeros(0,bool)
    initial=np.asarray(pixels,np.float32).reshape(-1,1,2)
    observed,forward,_=cv2.calcOpticalFlowPyrLK(previous,current,initial,None,winSize=(21,21),maxLevel=3)
    if observed is None:return np.empty((0,2)),np.zeros(len(initial),bool)
    returned,backward,_=cv2.calcOpticalFlowPyrLK(current,previous,observed,None,winSize=(21,21),maxLevel=3)
    if returned is None:return observed.reshape(-1,2),np.zeros(len(initial),bool)
    observed=observed.reshape(-1,2)
    good=(forward.ravel()==1)&(backward.ravel()==1)&(np.linalg.norm(returned.reshape(-1,2)-initial.reshape(-1,2),axis=1)<=limit)
    good&=(observed[:,0]>=0)&(observed[:,1]>=0)&(observed[:,0]<current.shape[1])&(observed[:,1]<current.shape[0])
    return observed,good


def projected(points,rotation,translation,k):
    camera=points@rotation.T+translation
    return camera[:,:2]/camera[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]


def geometry_corners(gray,rays,k,pose,maximum=180):
    rotation,translation=pose
    pixels=projected(rays.model,rotation,translation,k)
    mask=np.zeros_like(gray)
    hull=cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask,hull,255)
    mask=cv2.erode(mask,np.ones((5,5),np.uint8))
    corners=cv2.goodFeaturesToTrack(gray,maximum,.012,4,mask=mask,blockSize=5)
    if corners is None:return np.empty((0,2)),np.empty((0,3))
    pixels=corners.reshape(-1,2);points=rays.intersect(pixels,k,rotation,translation)
    valid=np.isfinite(points).all(axis=1)
    return pixels[valid],points[valid]


class SurfaceTracker(FastRigidTracker):
    def __init__(self,*args,mesh,initial_pose,**kwargs):
        self.use_visible_mask=kwargs.pop('visible_mask',False)
        self.external_foreground=kwargs.pop('external_foreground',None)
        self.replenish=kwargs.pop('replenish',True)
        kwargs.update(feature_limit=1000,feature_levels=8,use_roi=True)
        super().__init__(*args,**kwargs)
        self.rays=SurfaceRays(mesh)
        self.pose=(initial_pose[0].copy(),initial_pose[1].copy())
        self.failures=0;self.pending=None;self.pending_age=0
        self.search_pixels=self.previous_pixels.copy()
        self.pixel_threshold=2.5
        if self.use_visible_mask:
            from .visible_mask import VisibleMask
            self.masker=VisibleMask()

    def estimate(self,pixels,points,k,frame_id):
        stats=dict(correspondences=len(points),inliers=0,inlierRatio=0,featureCoverage=0,
                   medianReprojectionErrorPixels=None,trackingConfidence=None,solver='robust_pose_continuity')
        def fail(reason):return TrackingResult(frame_id,'lost',reason,None,None,stats),None
        if len(points)<6:return fail('insufficient_correspondences')
        spread=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)
        if spread[1]<1e-5:return fail('collinear_correspondences')
        proposals=[];cv2.setRNGSeed(17)
        try:
            ok,r,t,inliers=cv2.solvePnPRansac(points,pixels.astype(float),k,None,iterationsCount=120,
                reprojectionError=self.pixel_threshold,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
            if ok and inliers is not None and len(inliers)>=6:proposals.append((r,t,inliers.ravel()))
            if self.pose is not None and self.failures<=12:
                r=cv2.Rodrigues(self.pose[0])[0];t=self.pose[1].reshape(3,1).copy()
                # Trim observations iteratively; partial occlusion cannot dominate the fit.
                selected=np.arange(len(points))
                for _ in range(4):
                    if len(selected)<6:break
                    ok,r,t=cv2.solvePnP(points[selected],pixels[selected].astype(float),k,None,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
                    if not ok:break
                    errors=np.linalg.norm(cv2.projectPoints(points,r,t,k,None)[0].reshape(-1,2)-pixels,axis=1)
                    selected=np.flatnonzero(errors<=max(self.pixel_threshold,float(np.percentile(errors,65))))
                if ok and len(selected)>=6:proposals.append((r,t,selected))
        except cv2.error:return fail('opencv_pose_error')
        candidates=[]
        for r,t,selected in proposals:
            try:
                r,t=cv2.solvePnPRefineLM(points[selected],pixels[selected].astype(float),k,None,r,t)
            except cv2.error:continue
            rotation=cv2.Rodrigues(r)[0];translation=t.ravel()
            errors=np.linalg.norm(projected(points,rotation,translation,k)-pixels,axis=1)
            inliers=np.flatnonzero(errors<=self.pixel_threshold)
            if len(inliers)<6:continue
            if self.pose is not None and self.failures<=12:
                angle=np.arccos(np.clip((np.trace(rotation@self.pose[0].T)-1)/2,-1,1))
                distance=np.linalg.norm(translation-self.pose[1])
                if angle>np.deg2rad(18+2*self.failures) or distance>.06+.01*self.failures:continue
            camera=self.model@rotation.T+translation
            if not np.isfinite(camera).all() or (camera[:,2]<=.03).any():continue
            hull=cv2.convexHull(projected(self.model,rotation,translation,k).astype(np.float32))
            support=cv2.convexHull(pixels[inliers].astype(np.float32))
            coverage=cv2.contourArea(support)/max(cv2.contourArea(hull),1)
            ratio=len(inliers)/len(points);median=float(np.median(errors[inliers]))
            candidates.append((len(inliers),coverage,-median,rotation,translation,inliers,ratio,errors))
        if not candidates:return fail('pose_or_motion_rejected')
        _,coverage,_,rotation,translation,inliers,ratio,errors=max(candidates,key=lambda p:p[:3])
        stats.update(inliers=len(inliers),inlierRatio=ratio,featureCoverage=coverage,
                     medianReprojectionErrorPixels=float(np.median(errors[inliers])),
                     supportPixels=pixels[inliers].round(2).tolist(),
                     rejectedPixels=pixels[errors>self.pixel_threshold].round(2).tolist())
        if ratio<.5:return fail('insufficient_inlier_ratio')
        # Weak frames retain useful flow privately, but never display a stale pose.
        accepted=len(inliers)>=10 and coverage>=.10
        state='tracking' if accepted else 'limited'
        stats['trackingConfidence']='geometric_thresholds_met_not_a_probability' if accepted else None
        result=TrackingResult(frame_id,state,None if accepted else 'partial_surface_support',
                              rotation if accepted else None,translation if accepted else None,stats)
        return result,(rotation,translation,pixels[inliers],points[inliers])

    def update(self,rgb,k,frame_id):
        gray=rgb if rgb.ndim==2 else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.pixel_threshold=2.5*gray.shape[0]/360
        observed,good=follow(self.previous_gray,gray,self.previous_pixels)
        pixels=observed[good];points=self.previous_points[good] if self.previous_points is not None else np.empty((0,3))
        if self.iteration%5==0 or len(points)<30:
            matched,reference=self.match(gray)
            pixels=np.concatenate((pixels,matched));points=np.concatenate((points,reference))
        if len(points):
            _,unique=np.unique(np.round(points/.0005).astype(np.int64),axis=0,return_index=True)
            unique.sort();pixels=pixels[unique];points=points[unique]
            _,unique=np.unique(np.round(pixels/2).astype(np.int64),axis=0,return_index=True)
            unique.sort();pixels=pixels[unique];points=points[unique]
        external_stats={};external_mask=None
        if self.external_foreground is not None:
            external_mask,external_stats=self.external_foreground(gray,frame_id)
            if external_mask is not None and len(pixels):
                xy=np.round(pixels).astype(int)
                xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
                keep=external_mask[xy[:,1],xy[:,0]]>0
                external_stats['featuresExcludedByMask']=int((~keep).sum())
                pixels=pixels[keep];points=points[keep]
        result,retained=self.estimate(pixels,points,k,frame_id)
        result.statistics.update(external_stats)
        if self.use_visible_mask and retained is not None:
            geometry=self.rays.silhouette(gray.shape,k,*retained[:2])
            mask,mask_statistics=self.masker.estimate(gray,geometry)
            if mask is not None and len(pixels):
                coordinates=np.round(pixels).astype(int)
                in_image=(coordinates[:,0]>=0)&(coordinates[:,0]<gray.shape[1])&(coordinates[:,1]>=0)&(coordinates[:,1]<gray.shape[0])
                selected=np.zeros(len(points),bool)
                selected[in_image]=mask[coordinates[in_image,1],coordinates[in_image,0]]>0
                if selected.sum()>=6:
                    refined,new_retained=self.estimate(pixels[selected],points[selected],k,frame_id)
                    if new_retained is not None:result,retained=refined,new_retained
                mask_statistics['featuresExcludedByMask']=int((~selected).sum())
            result.statistics.update(mask_statistics)
        if retained is not None:
            rotation,translation,self.previous_pixels,self.previous_points=retained
            self.pose=(rotation,translation)
            if result.state=='tracking':self.failures=0
            else:self.failures+=1
            self.search_pixels=self.previous_pixels.copy();self.search_failures=0
            if self.pending is not None and result.state=='tracking':
                old_pixels,old_points=self.pending;current,good=follow(self.previous_gray,gray,old_pixels)
                good&=np.linalg.norm(projected(old_points,rotation,translation,k)-current,axis=1)<=self.pixel_threshold*.7
                self.pending=(current[good],old_points[good]);self.pending_age+=1
                if self.pending_age>=2:
                    self.previous_pixels=np.concatenate((self.previous_pixels,self.pending[0]))
                    self.previous_points=np.concatenate((self.previous_points,self.pending[1]))
                    self.pending=None
            else:self.pending=None
            if self.replenish and result.state=='tracking' and self.pending is None and (self.iteration%5==0 or len(self.previous_points)<60):
                new_pixels,new_points=geometry_corners(gray,self.rays,k,self.pose,100)
                if len(self.previous_pixels) and len(new_pixels):
                    distant=np.linalg.norm(new_pixels[:,None]-self.previous_pixels[None],axis=2).min(axis=1)>5
                    new_pixels=new_pixels[distant];new_points=new_points[distant]
                self.pending=(new_pixels,new_points);self.pending_age=0
        else:
            self.failures+=1;self.search_failures+=1;self.pending=None
            # Continue image flow briefly even when pose validation fails; no overlay.
            self.previous_pixels=pixels;self.previous_points=points
        if self.failures>12:
            self.previous_pixels=np.empty((0,2));self.previous_points=np.empty((0,3));self.pose=None
        if len(self.previous_pixels)>320:
            indices=np.linspace(0,len(self.previous_pixels)-1,320,dtype=int)
            self.previous_pixels=self.previous_pixels[indices];self.previous_points=self.previous_points[indices]
        self.previous_gray=gray;self.iteration+=1
        result.statistics.update(recoveryFrames=self.failures,workingHeight=gray.shape[0])
        if self.external_foreground is not None and external_mask is None and result.state=='tracking':
            result.state='limited';result.reason='foreground_unavailable'
            result.rotation=None;result.translation=None
        return result
