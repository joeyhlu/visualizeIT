"""Image-edge evidence supplements texture tracking for known rigid geometry.

No evaluation poses/masks enter this tracker. A silhouette is a geometric
prediction; hand occlusion is handled as missing/outlying image-edge evidence.
"""
import numpy as np
from .vision import cv2
from .surface_tracker import SurfaceTracker,projected
from .tracker import TrackingResult


class ContourTracker(SurfaceTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        triangles=kwargs['mesh'].positions[kwargs['mesh'].triangles]
        self.faces=kwargs['mesh'].triangles
        self.centres=triangles.mean(axis=1)
        self.face_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        self.contour_pose=(self.pose[0].copy(),self.pose[1].copy())
        self.contour_failures=0

    def silhouette(self,shape,k,pose):
        rotation,translation=pose;eye=-rotation.T@translation
        pixels=projected(self.model,rotation,translation,k)
        visible=np.sum(self.face_normals*(eye-self.centres),axis=1)>0
        mask=np.zeros(shape,np.uint8)
        for face in self.faces[visible]:
            cv2.fillConvexPoly(mask,np.round(pixels[face]).astype(np.int32),255)
        return mask

    def contour_fit(self,gray,k,pose):
        mask=self.silhouette(gray.shape,k,pose)
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contours=[c.reshape(-1,2) for c in contours if len(c)>=12]
        if not contours:return None,dict(contourReason='no_projected_boundary')
        pixels=np.concatenate(contours).astype(float)
        pixels=pixels[np.linspace(0,len(pixels)-1,min(160,len(pixels)),dtype=int)]
        gx=cv2.Sobel(mask,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(mask,cv2.CV_32F,0,1,ksize=3)
        lookup=pixels.astype(int);normals=-np.column_stack((gx[lookup[:,1],lookup[:,0]],gy[lookup[:,1],lookup[:,0]]))
        magnitude=np.linalg.norm(normals,axis=1);valid=magnitude>0
        pixels=pixels[valid];normals=normals[valid]/magnitude[valid,None]
        surface_pixels=pixels-normals*1.2
        points=self.rays.intersect(surface_pixels,k,*pose)
        good=np.isfinite(points).all(axis=1);points=points[good];pixels=pixels[good];normals=normals[good]
        if len(points)<25:return None,dict(contourReason='insufficient_boundary_geometry')
        smooth=cv2.GaussianBlur(gray,(5,5),.9)
        ix=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);iy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
        edges=cv2.Canny(smooth,40,100)
        steps=np.arange(-12,13)
        samples=pixels[:,None]+normals[:,None]*steps[None,:,None]
        samples=np.round(samples).astype(int)
        sx=np.clip(samples[:,:,0],0,gray.shape[1]-1);sy=np.clip(samples[:,:,1],0,gray.shape[0]-1)
        gradient=np.stack((ix[sy,sx],iy[sy,sx]),axis=2)
        magnitude=np.linalg.norm(gradient,axis=2)
        alignment=np.abs(np.sum(gradient*normals[:,None],axis=2))/np.maximum(magnitude,1)
        # Nearby, similarly oriented edges provide evidence; internal texture and
        # distant background edges do not automatically support the contour.
        evidence=np.where((edges[sy,sx]>0)&(alignment>.65),-np.abs(steps)+alignment*.3,-np.inf)
        best=np.argmax(evidence,axis=1);supported=np.isfinite(evidence[np.arange(len(points)),best])
        targets=samples[np.arange(len(points)),best].astype(float)-normals*1.2
        support=float(supported.mean())
        if supported.sum()<25 or support<.50:return None,dict(contourReason='insufficient_image_edge_support',contourSupport=support)
        points=points[supported];targets=targets[supported];normals=normals[supported]
        r=cv2.Rodrigues(pose[0])[0].ravel();t=pose[1].copy()
        initial_r=r.copy();initial_t=t.copy()
        for _ in range(8):
            observation,jacobian=cv2.projectPoints(points,r,t,k,None)
            residual=np.sum((observation.reshape(-1,2)-targets)*normals,axis=1)
            j=np.sum(jacobian[:,:6].reshape(-1,2,6)*normals[:,:,None],axis=1)
            weights=np.minimum(1,2.5/np.maximum(np.abs(residual),1e-6))
            h=j.T@(weights[:,None]*j)
            prior=np.diag([20,20,20,5000,5000,5000])
            delta=np.linalg.solve(h+prior+np.diag(np.maximum(np.diag(h)*.002,1e-3)),
                -j.T@(weights*residual)-prior@np.r_[r-initial_r,t-initial_t])
            if np.linalg.norm(delta[:3])>.12:delta*=.12/np.linalg.norm(delta[:3])
            r+=delta[:3];t+=delta[3:]
            if np.linalg.norm(delta)<1e-5:break
        rotation=cv2.Rodrigues(r)[0];observed=projected(points,rotation,t,k)
        residual=np.abs(np.sum((observed-targets)*normals,axis=1))
        angle=np.arccos(np.clip((np.trace(rotation@pose[0].T)-1)/2,-1,1))
        valid=(residual<3);coverage=float(valid.mean())
        statistics=dict(contourSupport=support,contourInlierFraction=coverage,contourMedianErrorPixels=float(np.median(residual)),
            contourPoints=int(len(points)),contourReason=None,
            contourPixels=observed[valid].round(2).tolist(),contourRejectedPixels=observed[~valid].round(2).tolist())
        if coverage<.65 or np.median(residual)>2 or angle>np.deg2rad(14) or np.linalg.norm(t-pose[1])>.04 or (self.model@rotation.T+t)[:,2].min()<=.03:
            statistics['contourReason']='contour_pose_rejected';return None,statistics
        # Require support across image sectors, not just a thumb-sized edge.
        centre=pixels.mean(axis=0);sector=(targets[:,0]>centre[0]).astype(int)+2*(targets[:,1]>centre[1]).astype(int)
        if len(np.unique(sector[valid]))<3:
            statistics['contourReason']='localized_boundary_support';return None,statistics
        return (rotation,t),statistics

    def update(self,rgb,k,frame_id):
        gray=rgb if rgb.ndim==2 else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        previous=self.contour_pose
        result=super().update(gray,k,frame_id)
        # Feature evidence proposes a pose when available. Otherwise image edges
        # must establish a fresh pose; no stale-pose rendering fallback exists.
        seed=(result.rotation,result.translation) if result.state=='tracking' else previous
        fitted,statistics=self.contour_fit(gray,k,seed) if seed is not None and self.contour_failures<12 else (None,dict(contourReason='no_recent_pose'))
        if fitted is not None:
            self.contour_pose=fitted;self.contour_failures=0
            self.pose=fitted;self.failures=0
            result=TrackingResult(frame_id,'tracking',None,*fitted,dict(result.statistics,**statistics,poseEvidence='image_edges_and_available_features'))
        elif result.state=='tracking':
            self.contour_pose=(result.rotation,result.translation);self.contour_failures=0
            result.statistics.update(statistics,poseEvidence='features_only')
        else:
            self.contour_failures+=1;result.statistics.update(statistics)
            if self.contour_failures>=12:self.contour_pose=None
        return result
