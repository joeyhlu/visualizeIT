"""Experimental bounded NumPy raster; the legacy raster remains the fallback."""
import math
import numpy as np
from bench import workbench_geometry as wg

FRAGMENT_CAP = 65536

def rasterize(vertices, faces, K, pose, width, height, *, near=1e-3, roi=None):
    K = wg.check_K(K)
    if near <= 0 or not math.isfinite(near):
        raise wg.GeometryError('near plane must be positive')
    if K[0, 1] != 0:
        raise wg.GeometryError('rasterizer requires zero skew')
    # The fast path targets ordinary finite geometry under NumPy's default
    # error policy. Exceptional arithmetic retains the reference behavior.
    pcheck = wg.check_pose(pose)
    vcheck = np.asarray(vertices, dtype=np.float64)
    if (np.geterr() != {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
            or not np.isfinite(vcheck).all() or not np.isfinite(pcheck).all()
            or np.any(np.abs(vcheck) > 1e100) or np.any(np.abs(pcheck) > 1e100)
            or np.any(np.abs(K) > 1e100) or near < 1e-100):
        return wg.rasterize(vertices, faces, K, pose, width, height, near=near, roi=roi)
    cam = wg.to_camera(vertices, pose)
    f = np.asarray(faces, dtype=np.int64)
    window = (0, 0, width-1, height-1) if roi is None else (
        max(0, int(roi[0])), max(0, int(roi[1])), min(width-1, int(roi[2])), min(height-1, int(roi[3])))
    rx0, ry0, rx1, ry1 = window
    def fallback():
        return wg.rasterize(vertices, faces, K, pose, width, height, near=near, roi=roi)
    # Keep exceptional-input warnings, clipping order and failure behavior in
    # the reference implementation. The measured finite-front case qualifies.
    if rx0 > rx1 or ry0 > ry1 or not np.isfinite(cam).all():
        return fallback()
    zall = cam[:, 2]
    if np.any(zall < near):
        return fallback()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = fx * cam[:, 0] / zall + cx
        sy = fy * cam[:, 1] / zall + cy
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        return fallback()
    fsx, fsy = sx[f], sy[f]
    outside = ((fsx.max(axis=1) < rx0-1) | (fsx.min(axis=1) > rx1+1) |
               (fsy.max(axis=1) < ry0-1) | (fsy.min(axis=1) > ry1+1))
    ids = np.flatnonzero(~outside)
    stats = {'faces':int(len(f)), 'clipped':0, 'culled_behind':0, 'guard_band_skipped':0,
             'degenerate_screen':0, 'culled_outside':int(outside.sum())}
    depth = np.full((height, width), np.inf)
    face_id = np.full((height, width), -1, dtype=np.int32)
    bary = np.zeros((height, width, 3))
    def result(): return wg.RasterResult(depth, face_id, bary, stats)
    if not len(ids): return result()
    guard = ((np.abs(fsx[ids]) > wg.GUARD_BAND_PX).any(axis=1) |
             (np.abs(fsy[ids]) > wg.GUARD_BAND_PX).any(axis=1))
    stats['guard_band_skipped'] = int(guard.sum()); ids = ids[~guard]
    if not len(ids): return result()
    X = np.round(fsx[ids] * wg.SUBPIXEL).astype(np.int64)
    Y = np.round(fsy[ids] * wg.SUBPIXEL).astype(np.int64)
    area = (X[:,1]-X[:,0])*(Y[:,2]-Y[:,0]) - (Y[:,1]-Y[:,0])*(X[:,2]-X[:,0])
    nonzero = area != 0; stats['degenerate_screen'] = int((~nonzero).sum())
    ids, X, Y, area = ids[nonzero], X[nonzero], Y[nonzero], area[nonzero]
    if not len(ids): return result()
    z = zall[f[ids]].copy()
    B = np.broadcast_to(np.eye(3), (len(ids),3,3)).copy()
    negative = area < 0
    X[negative] = X[negative][:,[0,2,1]]
    Y[negative] = Y[negative][:,[0,2,1]]
    z[negative] = z[negative][:,[0,2,1]]
    B[negative] = B[negative][:,[0,2,1]]
    area = np.abs(area)
    x0 = np.maximum(np.ceil(X.min(axis=1)/wg.SUBPIXEL).astype(np.int64),rx0)
    x1 = np.minimum(np.floor(X.max(axis=1)/wg.SUBPIXEL).astype(np.int64),rx1)
    y0 = np.maximum(np.ceil(Y.min(axis=1)/wg.SUBPIXEL).astype(np.int64),ry0)
    y1 = np.minimum(np.floor(Y.max(axis=1)/wg.SUBPIXEL).astype(np.int64),ry1)
    valid = (x0<=x1)&(y0<=y1)
    ids,X,Y,z,B,area,x0,x1,y0,y1 = [a[valid] for a in (ids,X,Y,z,B,area,x0,x1,y0,y1)]
    widths=x1-x0+1; counts=widths*(y1-y0+1)
    if np.any(counts > FRAGMENT_CAP): return fallback()
    cumulative=np.cumsum(counts); start=0
    depth_flat=depth.reshape(-1); id_flat=face_id.reshape(-1); bary_flat=bary.reshape(-1,3)
    while start < len(ids):
        prior=int(cumulative[start-1]) if start else 0
        end=int(np.searchsorted(cumulative,prior+FRAGMENT_CAP,side='right'))
        end=max(start+1,end)
        cs=counts[start:end]; total=int(cs.sum())
        tri=np.repeat(np.arange(start,end),cs)
        offsets=np.arange(total)-np.repeat(np.cumsum(cs)-cs,cs)
        px=(x0[tri]+offsets%widths[tri])*wg.SUBPIXEL
        py=(y0[tri]+offsets//widths[tri])*wg.SUBPIXEL
        weights=[]; inside=np.ones(total,dtype=bool)
        for a,b in ((1,2),(2,0),(0,1)):
            dx=X[tri,b]-X[tri,a];dy=Y[tri,b]-Y[tri,a]
            e=dx*(py-Y[tri,a])-dy*(px-X[tri,a])
            owned=(dy>0)|((dy==0)&(dx<0))
            inside &= (e>0)|((e==0)&owned);weights.append(e)
        if inside.any():
            tri=tri[inside];px=px[inside]//wg.SUBPIXEL;py=py[inside]//wg.SUBPIXEL
            lam=np.stack(weights,axis=-1)[inside].astype(np.float64)/area[tri,None].astype(np.float64)
            w=lam/z[tri];inv_z=w.sum(axis=-1)
            with np.errstate(divide='ignore',invalid='ignore'):
                zz=1.0/inv_z
                pixel=py*width+px
                closer=(zz>0)&(zz<depth_flat[pixel])
                if closer.any():
                    tri,pixel,zz,w,inv_z=[a[closer] for a in (tri,pixel,zz,w,inv_z)]
                    order=np.lexsort((tri,zz,pixel))
                    sorted_pixel=pixel[order]
                    unique=np.r_[True,sorted_pixel[1:]!=sorted_pixel[:-1]]
                    win=order[unique];target=pixel[win]
                    orig=(w[win,:,None]*B[tri[win]]).sum(axis=-2)/inv_z[win,None]
                    depth_flat[target]=zz[win];id_flat[target]=ids[tri[win]];bary_flat[target]=orig
        start=end
    return result()
