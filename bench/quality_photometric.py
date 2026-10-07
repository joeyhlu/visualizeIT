"""Diagnostic local SE(3) alignment against fixed model texture.

Inputs are the current image/mask and model samples, never evaluation poses.
This proposes poses; it does not establish tracking confidence or replace
GoTrack's geometric validation. Flat texture cannot constrain this objective.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .vision import cv2
from .quality_contract import canonical_pose


@dataclass(frozen=True)
class PhotometricSettings:
    maximum_points: int = 3000
    minimum_points: int = 200
    rotation_bound_degrees: float = 10.
    translation_bound_m: float = .012
    maximum_evaluations: int = 60
    minimum_relative_improvement: float = .05


def bilinear(image, pixels):
    """Double precision interpolation, without cv2 remap's 1/32 px steps."""
    image = np.asarray(image, np.float64)
    pixels = np.asarray(pixels, np.float64)
    h, w = image.shape
    valid = np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w-1)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < h-1)
    p = np.clip(np.nan_to_num(pixels), [0, 0], [w-1.000001, h-1.000001])
    x, y = np.floor(p).astype(int).T; a, b = (p-np.floor(p)).T
    values = ((1-a)*(1-b)*image[y, x] + a*(1-b)*image[y, x+1]
              + (1-a)*b*image[y+1, x] + a*b*image[y+1, x+1])
    return values, valid


def perturbed_pose(seed, parameters):
    pose = np.array(seed, dtype=np.float64, copy=True)
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(parameters[:3]).as_matrix()
    pose[:3, 3] += parameters[3:6]
    return pose


def project_samples(points_m, pose, intrinsics):
    xyz = np.asarray(points_m) @ pose[:3, :3].T + pose[:3, 3]
    projection = xyz @ np.asarray(intrinsics).T
    # Render depths use (x+.5,y+.5); numpy image samples use center index x,y.
    pixels = projection[:, :2]/np.maximum(projection[:, 2:3], 1e-8)-.5
    return pixels, xyz[:, 2] > 0


def align_samples(rgb, mask, intrinsics, seed, points_m, intensities, settings=None):
    settings = settings or PhotometricSettings(); seed = canonical_pose(seed)
    points_m = np.asarray(points_m, np.float64); intensities = np.asarray(intensities, np.float64)
    if len(points_m) != len(intensities): raise ValueError('Model samples and intensities must agree')
    if len(points_m) < settings.minimum_points or np.std(intensities) < .025:
        return seed, dict(state='unobservable_texture', points=len(points_m))
    gray = cv2.cvtColor(np.asarray(rgb, np.float32)/255, cv2.COLOR_RGB2GRAY)
    mask = np.asarray(mask, np.float64)
    angles = np.radians(settings.rotation_bound_degrees)
    bounds = np.r_[np.full(3, angles), np.full(3, settings.translation_bound_m), 1., .5]
    low = -bounds; high = bounds; low[6] = .25; high[6] = 2.
    parameters = np.r_[np.zeros(6), 1., 0.]
    objectives = []; jacobian = None
    for sigma in (2., .75):
        image = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float64)
        def residual(p):
            xy, front = project_samples(points_m, perturbed_pose(seed, p), intrinsics)
            values, inside = bilinear(image, xy); visibility, _ = bilinear(mask, xy)
            valid = inside & front & (visibility > .95)
            error = values - (p[6]*intensities+p[7])
            # Keep fixed residual dimensions and penalize escaping the object.
            return np.where(valid, error, .5)
        seed_values, _ = bilinear(image, project_samples(points_m, seed, intrinsics)[0])
        gain, offset = np.linalg.lstsq(np.column_stack((intensities, np.ones(len(intensities)))), seed_values, rcond=None)[0]
        seed_parameters = np.r_[np.zeros(6), np.clip(gain, .25, 2), np.clip(offset, -.5, .5)]
        initial = residual(seed_parameters)
        result = least_squares(residual, parameters, bounds=(low, high), loss='soft_l1',
            f_scale=.05, x_scale=np.r_[np.full(3, .03), np.full(3, .002), .2, .1],
            max_nfev=settings.maximum_evaluations, diff_step=None)
        def objective(error): return float(np.mean(np.sqrt(1+(error/.05)**2)-1))
        before = objective(initial)
        after = objective(residual(result.x))
        objectives.append(dict(sigma=sigma, before=before, after=after, evaluations=result.nfev))
        parameters = result.x; jacobian = result.jac[:, :6]
    xy, front = project_samples(points_m, perturbed_pose(seed, parameters), intrinsics)
    visible, inside = bilinear(mask, xy)
    retained = float(np.mean(inside & front & (visible > .95)))
    improvement = (objectives[-1]['before']-objectives[-1]['after'])/max(objectives[-1]['before'], 1e-12)
    # Diagnostic observability: scale Jacobian columns to physical perturbations.
    sv = np.linalg.svd(jacobian*np.r_[np.full(3, .03), np.full(3, .002)], compute_uv=False)
    constrained = bool(sv[-1] > 1e-4 and sv[0]/sv[-1] < 1e4)
    usable = retained >= .95 and constrained and improvement >= settings.minimum_relative_improvement
    candidate = perturbed_pose(seed, parameters) if usable else seed
    return candidate, dict(state='proposal' if usable else 'not_improved', points=len(points_m),
        retained_visible_fraction=retained, relative_improvement=improvement,
        parameter_delta=parameters.tolist(), jacobian_singular_values=sv.tolist(),
        stages=objectives, settings=asdict(settings), tracking_validated=False)


def render_samples(color, depth_mm, mask, seed, intrinsics, settings=None):
    settings = settings or PhotometricSettings()
    color = np.asarray(color, np.float32)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    visible = cv2.erode((np.asarray(mask)>0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    visible &= np.asarray(depth_mm) > 0
    y, x = np.nonzero(visible)
    if len(x) > settings.maximum_points:
        selected = np.linspace(0, len(x)-1, settings.maximum_points, dtype=int); x=x[selected]; y=y[selected]
    z = np.asarray(depth_mm)[y, x]*.001
    rays = np.column_stack((x+.5, y+.5, np.ones(len(x)))) @ np.linalg.inv(intrinsics).T
    xyz_camera = rays*z[:, None]
    pose = canonical_pose(seed)
    points = (xyz_camera-pose[:3, 3]) @ pose[:3, :3]
    return points, gray[y, x].astype(np.float64)
