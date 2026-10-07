"""Experimental convex-object contour proposal; never an accepted tracking pose by itself.

Uses a fixed 3D asset, current visible mask and a supplied automatic pose.
No annotation or estimated reference pose is read. Concave/deformable objects
and permanent integration are outside this keyboard diagnostic's scope.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .quality_contract import checked_pose,project
from .vision import cv2


@dataclass(frozen=True)
class ContourSettings:
    samples: int=256
    iterations: int=5
    maximum_distance_720: float=25.
    minimum_normal_agreement: float=.75
    minimum_support: float=.15
    rotation_bound_degrees: float=8.
    translation_bound_m: float=.04
    prior_residual_pixels: float=4.


def boundary(mask):
    contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea)[:,0].astype(float)
    if len(contour)<12:return None
    tangent=np.roll(contour,-3,axis=0)-np.roll(contour,3,axis=0)
    sign=1 if cv2.contourArea(contour.astype(np.float32),oriented=True)>0 else -1
    normals=sign*np.column_stack([tangent[:,1],-tangent[:,0]])
    lengths=np.linalg.norm(normals,axis=1);valid=lengths>1e-6
    return contour[valid],normals[valid]/lengths[valid,None]


def projected_hull(vertices,pose,k,count):
    pixels,z=project(vertices,pose,k)
    if not np.isfinite(pixels).all() or np.any(z<=.01):return None
    ids=cv2.convexHull(pixels.astype(np.float32),returnPoints=False).ravel()
    xyz=vertices[ids];pixels=pixels[ids]
    edge=np.roll(pixels,-1,axis=0)-pixels
    lengths=np.linalg.norm(edge,axis=1);total=lengths.sum()
    if total<1:return None
    cumulative=np.cumsum(lengths);distances=(np.arange(count)+.5)/count*total
    segments=np.searchsorted(cumulative,distances)
    fraction=(distances-np.r_[0,cumulative[:-1]][segments])/lengths[segments]
    points=xyz[segments]*(1-fraction[:,None])+np.roll(xyz,-1,axis=0)[segments]*fraction[:,None]
    # Interpolation is in 3D; project again rather than linearly warping artwork.
    coordinates,_=project(points,pose,k)
    sign=1 if cv2.contourArea(pixels.astype(np.float32),oriented=True)>0 else -1
    normals=sign*np.column_stack([edge[:,1],-edge[:,0]])/lengths[:,None]
    return points,coordinates,normals[segments],pixels


def correspondences(vertices,pose,k,observed,settings,scale):
    hull=projected_hull(vertices,pose,k,settings.samples)
    if hull is None:return None
    xyz,uv,normals,hull_pixels=hull;target,target_normals=observed
    tree=cKDTree(target);distances,indices=tree.query(uv,k=min(24,len(target)))
    alignment=(normals[:,None,:]*target_normals[indices]).sum(2)
    permitted=(alignment>=settings.minimum_normal_agreement)&(distances<=settings.maximum_distance_720/scale)
    ranked=np.where(permitted,distances,np.inf);choices=ranked.argmin(1)
    best=ranked[np.arange(len(uv)),choices];keep=np.isfinite(best)
    if keep.sum()<24:return None
    indices=indices[np.arange(len(uv)),choices]
    spread=cv2.contourArea(cv2.convexHull(uv[keep].astype(np.float32)))
    total=cv2.contourArea(hull_pixels.astype(np.float32))
    support=spread/max(total,1)
    if support<settings.minimum_support:return None
    return xyz[keep],target[indices[keep]],dict(samples=int(keep.sum()),support=float(support),
        median_boundary_distance_720=float(np.median(best[keep])*scale),
        p95_boundary_distance_720=float(np.percentile(best[keep],95)*scale))


def refine(vertices,k,mask,seed,settings=ContourSettings()):
    seed=checked_pose(seed);vertices=np.asarray(vertices,float);scale=720/mask.shape[0]
    observed=boundary(mask)
    if observed is None:return None,dict(reason='no_current_mask_boundary')
    initial=correspondences(vertices,seed,k,observed,settings,scale)
    if initial is None:return None,dict(reason='insufficient_visible_boundary_support')
    bounds=np.r_[np.full(3,np.radians(settings.rotation_bound_degrees)),np.full(3,settings.translation_bound_m)]
    def pose_at(delta):
        pose=seed.copy();pose[:3,:3]=cv2.Rodrigues(delta[:3])[0]@seed[:3,:3];pose[:3,3]+=delta[3:]
        return pose
    delta=np.zeros(6);current=seed
    for _ in range(settings.iterations):
        match=correspondences(vertices,current,k,observed,settings,scale)
        if match is None:return None,dict(reason='boundary_support_lost_during_proposal')
        xyz,target,stats=match
        def residual(value):
            pixels,_=project(xyz,pose_at(value),k)
            return np.r_[(pixels-target).ravel()*scale,value/bounds*settings.prior_residual_pixels]
        optimized=least_squares(residual,delta,bounds=(-bounds,bounds),loss='soft_l1',f_scale=2.,max_nfev=50)
        if not optimized.success:return None,dict(reason='contour_optimizer_did_not_converge')
        delta=optimized.x;current=pose_at(delta)
    final=correspondences(vertices,current,k,observed,settings,scale)
    if final is None:return None,dict(reason='final_boundary_support_unavailable')
    before=initial[2];after=final[2]
    if after['median_boundary_distance_720']>=before['median_boundary_distance_720']:
        return None,dict(reason='boundary_fit_did_not_improve',before=before,after=after)
    return checked_pose(current),dict(reason=None,before=before,after=after,
        delta_camera_rotation_degrees=np.degrees(delta[:3]).tolist(),delta_camera_translation_m=delta[3:].tolist(),
        accepted_tracking_pose=False,visibility_or_attachment_verified=False)
