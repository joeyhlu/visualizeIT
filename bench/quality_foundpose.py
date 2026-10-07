"""Upstream FoundPose TF-IDF + cyclic buddies + PnP, returning five hypotheses.

The template representation is generated from the fixed textured model alone.
No video annotations or accumulated surface estimates enter its feature bank.
"""
import numpy as np
from .quality_assets import CACHE, BUDGET
from .quality_contract import Frame, pose_units
from .quality_gotrack import upstream_path
from .quality_time import LEGACY, PHYSICAL, FrameKey, TimePolicy, LEGACY_POLICY
from .vision import cv2
from .quality_trace import trace


def build_bank(renderer, backbone, device, path):
    upstream_path()
    from .quality_neighbors import install
    install()
    import torch
    from utils import misc, structs, data_util, feature_util, cluster_util, template_util, pca_util
    sphere = misc.sample_views(min_n_views=57, radius=923.075, mode='fibonacci')[0]
    camera = structs.PinholePlaneCameraModel(width=640, height=480, f=(572.4114, 573.57043), c=(325.2611, 242.04899), T_world_from_eye=np.eye(4))
    xyzs = []; features = []; tids = []; cameras = []
    dummy = torch.zeros(1, 3, 480, 640)
    with torch.inference_mode():
        for view in sphere:
            for roll in range(14):
                angle = 2*np.pi*roll/14
                r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
                pose = np.eye(4); pose[:3, :3] = r@view['R']; pose[:3, 3] = (r@view['t']).ravel()
                data, crops, delta = data_util.compute_gotrack_inputs_from_init_poses(input_rgbs=dummy,
                    input_cameras=[camera], init_poses_cam_from_model=torch.from_numpy(pose.astype(np.float32))[None],
                    renderer=renderer, obj_ids=[renderer.obj_id], object_vertices=[renderer.vertices_m*1000],
                    crop_size=(280, 280), crop_rel_pad=.05, cropping_type='perspective_2d_box', ssaa_factor=4., background_type='gray')
                template = data['templates']; crop_pose = delta[0].numpy()@pose
                _, f, _, xyz = feature_util.get_visual_features_registered_in_3d(image_chw=template.rgbs[0].to(device),
                    depth_image_hw=template.depths[0].to(device), object_mask=template.masks[0].to(device), camera=crops[0],
                    T_model_from_camera=torch.from_numpy(np.linalg.inv(crop_pose).astype(np.float32)), extractor=backbone,
                    grid_cell_size=14., debug=False)
                i = len(cameras); xyzs.append(xyz.cpu()); features.append(f.cpu()); tids.append(torch.full((len(f),), i, dtype=torch.int64))
                cameras.append(crops[0])
                if i % 100 == 0: print('FoundPose templates', i+1, '/ 798', flush=True)
    vectors = torch.cat(features); ids = torch.cat(tids)
    projector = pca_util.PCAProjector(n_components=256, whiten=False)
    np.random.seed(0); torch.manual_seed(0)
    projector.fit(vectors, max_samples=100000)
    reduced = projector.transform(vectors).contiguous().cpu()
    centroids, words, _ = cluster_util.kmeans(reduced, num_centroids=2048, verbose=False)
    descs, idfs = template_util.calc_tfidf_descriptors(feat_vectors=reduced, feat_to_word_ids=words,
        feat_to_template_ids=ids, feat_words=centroids, num_templates=len(cameras), tfidf_knn_k=3,
        tfidf_soft_assign=False, tfidf_soft_sigma_squared=10.)
    state = dict(vertices=torch.cat(xyzs), feat_vectors=reduced, feat_to_template_ids=ids,
                 centroids=centroids.cpu(), descs=descs.cpu(), idfs=idfs.cpu(), projector=projector,
                 template_count=len(cameras), asset_sha256=renderer.asset_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part'); torch.save(state, temporary)
    used = sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    if used > BUDGET:
        temporary.unlink(); raise ValueError('8 GiB model-cache budget exceeded by template bank')
    temporary.replace(path)
    return state


class FoundPoseRecovery:
    def __init__(self, refiner, bank, model_memory=False, *, time_policy=LEGACY_POLICY):
        if not isinstance(time_policy, TimePolicy):
            raise ValueError('time_policy must be a TimePolicy')
        upstream_path()
        from .quality_neighbors import install
        install()
        from utils import repre_util, config, knn_util
        self.refiner = refiner; self.backbone = refiner.network.backbone
        self.time_policy = time_policy
        self.clock_modes_supported = frozenset((time_policy.mode,))
        self.memory = None
        if model_memory:
            from .quality_memory import ModelPointMemory
            self.memory = ModelPointMemory(self.render_depth, time_policy=time_policy)
        if bank['asset_sha256'] != refiner.renderer.asset_sha256: raise ValueError('Wrong object template bank')
        self.projector = bank['projector']
        self.repre = repre_util.FeatureBasedObjectRepre(vertices=bank['vertices'], feat_vectors=bank['feat_vectors'],
            feat_to_template_ids=bank['feat_to_template_ids'], feat_cluster_centroids=bank['centroids'],
            feat_cluster_idfs=bank['idfs'], template_descs=bank['descs'], template_desc_opts=config.TemplateDescOpts())
        self.visual_words = knn_util.KNN(k=3, metric='l2'); self.visual_words.fit(bank['centroids'])
        self.indices = []
        for i in range(bank['template_count']):
            index = knn_util.KNN(k=1, metric='l2'); index.fit(bank['feat_vectors'][bank['feat_to_template_ids'] == i]); self.indices.append(index)

    def refine(self, frame, mask, seed): return self.refiner.refine(frame, mask, seed)

    def render_depth(self, pose, k, width, height):
        from utils import structs, renderer_base
        camera_pose = np.linalg.inv(pose); camera_pose[:3, 3] *= 1000
        camera = structs.PinholePlaneCameraModel(width=width, height=height,
            f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=camera_pose)
        return self.refiner.renderer.render_object_model(self.refiner.obj_id, camera)[renderer_base.RenderType.DEPTH]*.001

    def observe(self, frame, mask):
        if self.time_policy.mode == PHYSICAL and (
                not isinstance(frame, Frame) or frame.clock_mode != PHYSICAL):
            raise ValueError('Physical motion observation requires a physical Frame')
        if self.memory is not None: self.memory.observe(frame, mask)

    def motion_seed(self, frame_or_legacy_id):
        if self.time_policy.mode == PHYSICAL:
            if not isinstance(frame_or_legacy_id, Frame) or frame_or_legacy_id.clock_mode != PHYSICAL:
                raise ValueError('Physical motion seeds require a physical Frame')
        elif isinstance(frame_or_legacy_id, Frame):
            raise ValueError('Legacy motion seeds require an integer frame ID')
        return None if self.memory is None else self.memory.seed(frame_or_legacy_id)

    def validate_motion(self, candidate, frame_or_legacy_id):
        if self.time_policy.mode == PHYSICAL:
            if not isinstance(frame_or_legacy_id, Frame) or frame_or_legacy_id.clock_mode != PHYSICAL:
                raise ValueError('Physical motion validation requires a physical Frame')
        elif isinstance(frame_or_legacy_id, Frame):
            raise ValueError('Legacy motion validation requires an integer frame ID')
        return ((True, None, {'state': 'disabled'}) if self.memory is None else
                self.memory.validate(candidate, frame_or_legacy_id))

    def clear_motion(self, key: FrameKey, *, reason: str):
        if not isinstance(key, FrameKey) or key.clock_mode != self.time_policy.mode:
            raise ValueError('FoundPose motion cleanup clock mode mismatch')
        if not isinstance(reason, str) or not reason:
            raise ValueError('FoundPose motion cleanup requires a reason')
        if self.memory is not None:
            self.memory.clear_motion(key, reason=reason)

    def commit(self, frame, mask, pose):
        if self.time_policy.mode == PHYSICAL and (
                not isinstance(frame, Frame) or frame.clock_mode != PHYSICAL):
            raise ValueError('Physical model-memory commit requires a physical Frame')
        if self.memory is not None: self.memory.commit(frame, mask, pose)

    def recover(self, frame, mask, top_k=5):
        import torch
        from utils import crop_generation, structs, feature_util, corresp_util, pnp_util
        h, w = frame.rgb.shape[:2]; k = frame.intrinsics
        yy, xx = np.nonzero(mask)
        if len(xx) < 24: return []
        camera = structs.PinholePlaneCameraModel(width=w, height=h, f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=np.eye(4))
        crops = crop_generation.batch_cropping_from_bbox(source_images=[frame.rgb], source_cameras=[camera],
            source_xyxy_bboxes=np.array([[xx.min(), yy.min(), xx.max()+1, yy.max()+1]]), source_masks=(mask > 0)[None, ..., None],
            crop_size=(280, 280), crop_rel_pad=.05)
        crop_mask = crops.masks[0].to(self.refiner.device)
        with torch.inference_mode():
            feature = self.backbone(crops.rgbs.to(self.refiner.device))['feature_maps'][0]
            points = feature_util.filter_points_by_mask(feature_util.generate_grid_points((280, 280), 14.).to(self.refiner.device), crop_mask)
            if len(points) < 6: return []
            vectors = feature_util.sample_feature_map_at_points(feature, points, (280, 280))
            vectors = self.projector.transform(vectors.cpu()).contiguous()
            trace(f'recover/{frame.frame_id}', image=frame.rgb, vectors=vectors, points=points)
            corresps = corresp_util.establish_correspondences(query_points=points.cpu(), query_features=vectors,
                object_repre=self.repre, template_matching_type='tfidf', feat_matching_type='cyclic_buddies',
                top_n_templates=min(top_k, len(self.indices)), top_k_buddies=300, visual_words_knn_index=self.visual_words,
                template_knn_indices=self.indices, debug=False)
        hypotheses = []
        for i, c in enumerate(corresps):
            cv2.setRNGSeed((frame.frame_id*29+i) & 0x7fffffff)
            ok, r, t, inliers, quality = pnp_util.estimate_pose(c, crops.cameras[0], pnp_type='opencv',
                pnp_ransac_iter=400, pnp_inlier_thresh=10., pnp_required_ransac_conf=.99, pnp_refine_lm=True)
            if not ok: continue
            pose = np.eye(4); pose[:3, :3] = r; pose[:3, 3] = np.asarray(t).ravel()
            original = crops.cameras[0].T_world_from_eye@pose
            hypotheses.append((float(quality), pose_units(original, .001)))
        hypotheses.sort(key=lambda x: x[0], reverse=True)
        return [pose for _, pose in hypotheses[:top_k]]


# Capture-only adapter. The legacy build_bank and FoundPoseRecovery bodies above remain unchanged.
import math as _capture_math
import json as _capture_json
import shutil as _capture_shutil
import time as _capture_time
import uuid as _capture_uuid
from collections.abc import Mapping as _CaptureMapping
from datetime import datetime as _capture_datetime, timezone as _capture_timezone
from pathlib import Path as _capture_Path

_BANK_TEMPLATE_COUNT = 798
_BANK_MAX_FEATURE_ROWS = _BANK_TEMPLATE_COUNT * 400
_BANK_ARTIFACT_CEILING_BYTES = 512 * 1024**2
_BANK_FINALIZATION_COPY_COUNT = 2
_BANK_RECEIPT_ALLOWANCE_BYTES = 16 * 1024**2
_BANK_PROSPECTIVE_RESERVE_BYTES = (
    _BANK_FINALIZATION_COPY_COUNT * _BANK_ARTIFACT_CEILING_BYTES +
    _BANK_RECEIPT_ALLOWANCE_BYTES)
_BANK_POST_BUILD_RESERVE_BYTES = (
    (_BANK_FINALIZATION_COPY_COUNT - 1) * _BANK_ARTIFACT_CEILING_BYTES +
    _BANK_RECEIPT_ALLOWANCE_BYTES)
_BANK_MAX_EVIDENCE_BYTES = _BANK_RECEIPT_ALLOWANCE_BYTES


def _capture_bank_target(output, quality_assets):
    if not isinstance(output, (str, _capture_Path)):
        raise TypeError('Capture bank output must be a string or pathlib.Path')
    requested = _capture_Path(output)
    if not requested.is_absolute():
        raise ValueError('Capture bank output must be an absolute private trial path')
    cache_root = quality_assets.CAPTURE_CACHE.resolve(strict=True)
    trials_entry = cache_root / 'capture-bank-trials'
    if trials_entry.is_symlink():
        raise ValueError('Capture bank trial root cannot be a symlink')
    trials_root = trials_entry.resolve(strict=False)
    if trials_root.parent != cache_root or trials_root.name != 'capture-bank-trials':
        raise ValueError('Capture bank trial root escapes CAPTURE_CACHE')
    if trials_entry.exists() and not trials_entry.is_dir():
        raise ValueError('Capture bank trial root is not a directory')
    if requested.parent.is_symlink():
        raise ValueError('Capture bank trial directory cannot be a symlink')
    target = requested.resolve(strict=False)
    try:
        relative = target.relative_to(trials_root)
    except ValueError as exc:
        raise ValueError('Capture bank output must stay under CAPTURE_CACHE/capture-bank-trials') from exc
    if len(relative.parts) != 2 or target.name != 'artifact.pt':
        raise ValueError('Capture bank output must be <fresh-trial-id>/artifact.pt')
    trial_id = relative.parts[0]
    if (not trial_id or trial_id in ('.', '..') or
            any(not (character.isascii() and
                     (character.isalnum() or character in '._-'))
                for character in trial_id)):
        raise ValueError('Capture bank trial id must be one safe path component')
    if (target.exists() or target.is_symlink() or target.parent.exists() or
            target.parent.is_symlink()):
        raise FileExistsError('Capture bank trial output and its parent must be fresh')
    temporary_candidates = (target.with_suffix('.part'), target.with_name(target.name + '.part'))
    if any(path.exists() or path.is_symlink() for path in temporary_candidates):
        raise FileExistsError('Capture bank trial contains a stale partial artifact')
    return target


def _capture_bank_budget_reserve(cache_bytes, free_bytes, budget):
    """Pure prospective disk/cache gate derived from the fixed 798-template layout."""
    if any(type(value) is not int or value < 0
           for value in (cache_bytes, free_bytes, budget)):
        raise ValueError('Capture bank budget observations must be nonnegative exact integers')
    if cache_bytes + _BANK_PROSPECTIVE_RESERVE_BYTES > budget:
        raise ValueError('FoundPose capture bank prospective 1040 MiB reserve exceeds the 8 GiB cache budget')
    remaining_free = free_bytes - _BANK_PROSPECTIVE_RESERVE_BYTES
    if remaining_free <= 20 * 1024**3:
        raise ValueError('FoundPose capture bank requires more than 20 GiB after its prospective reserve')
    return {
        'max_feature_rows': _BANK_MAX_FEATURE_ROWS,
        'artifact_ceiling_bytes': _BANK_ARTIFACT_CEILING_BYTES,
        'finalization_copy_count': _BANK_FINALIZATION_COPY_COUNT,
        'receipt_allowance_bytes': _BANK_RECEIPT_ALLOWANCE_BYTES,
        'reserve_bytes': _BANK_PROSPECTIVE_RESERVE_BYTES,
        'cache_bytes_before': cache_bytes,
        'cache_budget_bytes': budget,
        'cache_bytes_with_reserve': cache_bytes + _BANK_PROSPECTIVE_RESERVE_BYTES,
        'free_bytes_before': free_bytes,
        'free_bytes_after_reserve': remaining_free,
        'minimum_free_bytes_exclusive': 20 * 1024**3,
        'estimate_formula': '2*512MiB artifact ceiling + 16MiB receipt/runtime allowance',
        'payload_bound_formula': '319200*(3*4+256*4+8)+2048*256*4+798*2048*4+2048*4+<2MiB PCA',
    }


def _capture_bank_preflight(verified_plan, resources, output, device):
    from . import quality_assets

    if type(device) is not str or device not in ('cpu', 'cuda'):
        raise ValueError('Capture bank device must be cpu or cuda')
    if type(resources) is not quality_assets.CaptureStageResources:
        raise TypeError('Capture bank requires an exact issued CaptureStageResources record')

    provenance = quality_assets.inference_provenance(
        verified_plan.bundle, capture_plan=verified_plan, capture_resources=resources)
    capture = provenance.get('capture_provenance')
    if type(capture) is not dict:
        raise ValueError('Capture bank requires public issued capture provenance')
    if capture.get('stage') != 'banks' or resources.stage != 'banks':
        raise ValueError('FoundPose capture bank requires issued banks-stage resources')
    if device != resources.device or capture.get('render_flags') is None:
        raise ValueError('Capture bank device differs from the issued stage resources')
    if resources.output is None or type(resources.output) is not quality_assets.CaptureResourceUse:
        raise ValueError('FoundPose capture bank is missing its deterministic issued output resource')
    if resources.output.descriptor is not None or resources.output.purpose != 'foundpose-bank':
        raise ValueError('FoundPose output must be the unproduced issued foundpose-bank resource')
    if (len(resources.required) != 1 or resources.required[0].purpose != 'smoke' or
            resources.required[0].descriptor is None):
        raise ValueError('FoundPose capture bank requires exactly one verified model-smoke prerequisite')

    flags = capture['render_flags']
    expected_flag_names = {'unlit_templates', 'disable_multisampling', 'grayscale'}
    if (type(flags) is not dict or set(flags) != expected_flag_names or
            any(type(value) is not bool for value in flags.values()) or
            flags['disable_multisampling'] is not False or flags['grayscale'] is not False):
        raise ValueError('FoundPose capture bank requires the exact issued lit/unlit flags and multisampling policy')
    if dict(resources.render_flags) != flags:
        raise ValueError('Capture bank flags differ from the exact issued resource record')

    if (len(capture['used_resources']) != 1 or
            capture['used_resources'][0].get('purpose') != 'smoke' or
            capture['used_resources'][0].get('resource_kind') !=
            'quality-capture-model-smoke-v1'):
        raise ValueError('FoundPose capture bank requires the semantically verified model-smoke artifact')

    output_resource = capture.get('output_resource')
    if type(output_resource) is not dict:
        raise ValueError('FoundPose capture bank output provenance is malformed')
    recipe = output_resource.get('recipe')
    if type(recipe) is not dict:
        raise ValueError('FoundPose capture bank recipe is malformed')
    expected_recipe_fields = {
        'schema', 'resource', 'resource_kind', 'object_id', 'object_name', 'asset_sha256',
        'asset_receipt_sha256', 'unit_receipt_sha256', 'renderer_proof_sha256',
        'output_camera_sha256', 'coordinate_mode', 'rendering_contract', 'render_flags',
        'producer_device', 'settings',
    }
    if set(recipe) != expected_recipe_fields:
        raise ValueError('FoundPose capture bank recipe has an unexpected shape')
    expected_settings = {
        'template_count': 798, 'view_count': 57, 'rolls_per_view': 14,
        'pca_components': 256, 'kmeans_clusters': 2048,
        'color_ssaa_factor': 4, 'depth_mask_ssaa_factor': 1,
    }
    if (recipe.get('schema') != 'quality-capture-resource-recipe-v1' or
            recipe.get('resource') != 'foundpose-bank' or
            recipe.get('resource_kind') != 'quality-capture-foundpose-bank-v1' or
            recipe.get('settings') != expected_settings or
            recipe.get('asset_sha256') != resources.asset_receipt.asset_sha256 or
            recipe.get('asset_receipt_sha256') != resources.asset_receipt.receipt_sha256 or
            recipe.get('unit_receipt_sha256') != resources.asset_receipt.unit_receipt_sha256 or
            recipe.get('renderer_proof_sha256') != resources.asset_receipt.renderer_proof_sha256 or
            recipe.get('object_id') != resources.asset_receipt.object_id or
            recipe.get('object_name') != resources.asset_receipt.object_name or
            recipe.get('producer_device') != device or
            recipe.get('coordinate_mode') != 'integer_centers_v1' or
            recipe.get('render_flags') != {
                'unlit_templates': flags['unlit_templates'],
                'disable_multisampling': False,
            }):
        raise ValueError('FoundPose bank recipe differs from the authenticated issued policy')
    contract = recipe.get('rendering_contract')
    expected_contract = {
        'render_policy': 'capture_zero_sample_v1',
        'coordinate_mode': 'integer_centers_v1',
        'frame_dimensions': (720, 720),
        'color_storage': 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage',
        'depth_storage': 'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage',
        'mask_source': 'same-frame-renderer-depth-greater-than-zero',
        'renderer_depth_units': 'millimetres',
        'renderer_depth_to_metres_conversion_count': 1,
        'pose_translation_input_units': 'millimetres',
        'crop_dimensions': (280, 280),
        'crop_coordinate_mode': 'integer_centers_v1',
        'foundpose_color_native_dimensions': (1120, 1120),
        'foundpose_color_resize_dimensions': (280, 280),
        'foundpose_color_interpolation': 'area',
        'foundpose_resized_intrinsics_rule': 's*fx,s*fy,s*(c+0.5)-0.5',
        'low_resolution_depth_mask_rendered_separately': True,
        'disable_multisampling': False,
        'textured_renderer_unlit': flags['unlit_templates'],
        'cnos_grayscale_postprocess': False,
    }
    if type(contract) is not dict or set(contract) != set(expected_contract):
        raise ValueError('FoundPose rendering contract has an unexpected shape')
    for name, expected in expected_contract.items():
        actual = contract[name]
        if name in ('frame_dimensions', 'crop_dimensions', 'foundpose_color_native_dimensions',
                    'foundpose_color_resize_dimensions'):
            actual = tuple(actual) if isinstance(actual, (tuple, list)) else actual
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'FoundPose rendering contract field {name!r} differs from policy')

    manifest = verified_plan.manifest
    bundle = _capture_Path(verified_plan.bundle).resolve(strict=True)
    asset_name = manifest.get('asset')
    if type(asset_name) is not str or not asset_name:
        raise ValueError('Issued capture manifest has no normalized GLB path')
    relative_asset = _capture_Path(asset_name.replace(chr(92), '/'))
    if relative_asset.is_absolute() or any(part in ('', '.', '..') for part in relative_asset.parts):
        raise ValueError('Issued capture asset path is malformed')
    asset_path = bundle.joinpath(*relative_asset.parts).resolve(strict=True)
    try:
        asset_path.relative_to(bundle)
    except ValueError as exc:
        raise ValueError('Issued capture asset escapes its verified bundle') from exc
    if (not asset_path.is_file() or
            quality_assets.digest(asset_path).upper() != resources.asset_receipt.asset_sha256.upper()):
        raise ValueError('Current normalized capture GLB differs from its issued asset receipt')

    checkpoint_name = 'gotrack_checkpoint.pt'
    checkpoint_spec = quality_assets.MODELS.get(checkpoint_name)
    if type(checkpoint_spec) is not dict or type(checkpoint_spec.get('sha256')) is not str:
        raise ValueError('Pinned GoTrack checkpoint SHA-256 is unavailable')
    checkpoint_sha = checkpoint_spec['sha256'].upper()
    if provenance.get('checkpoint_sha256', {}).get(checkpoint_name, '').upper() != checkpoint_sha:
        raise ValueError('FoundPose GoTrack checkpoint differs from its issued source closure')
    checkpoint_path = (quality_assets.CACHE / 'checkpoints' / checkpoint_name).resolve(strict=True)
    try:
        checkpoint_path.relative_to(quality_assets.CACHE.resolve(strict=True))
    except ValueError as exc:
        raise ValueError('Pinned GoTrack checkpoint escapes the bounded model cache') from exc
    if quality_assets.digest(checkpoint_path).upper() != checkpoint_sha:
        raise ValueError('Current GoTrack checkpoint differs from the frozen pin')

    target = _capture_bank_target(output, quality_assets)
    cache_root = quality_assets.CACHE.resolve(strict=True)
    cache_bytes = sum(path.stat().st_size for path in cache_root.rglob('*') if path.is_file())
    free_bytes = _capture_shutil.disk_usage(cache_root).free
    reserve = _capture_bank_budget_reserve(cache_bytes, free_bytes, quality_assets.BUDGET)

    return (provenance, capture, recipe, flags, manifest, asset_path, checkpoint_sha,
            target, cache_bytes, free_bytes, reserve)


def _capture_bank_camera_pair(upstream_camera):
    """Return corrected 1120 COLOR and native 280 geometry cameras without mutation."""
    width = getattr(upstream_camera, 'width', None)
    height = getattr(upstream_camera, 'height', None)
    if type(width) is not int or type(height) is not int or (width, height) != (1120, 1120):
        raise ValueError('FoundPose capture adapter requires the upstream 1120x1120 SSAA camera')
    focal = np.asarray(getattr(upstream_camera, 'f', None), dtype=np.float64)
    principal = np.asarray(getattr(upstream_camera, 'c', None), dtype=np.float64)
    pose = np.asarray(getattr(upstream_camera, 'T_world_from_eye', None), dtype=np.float64)
    if (focal.shape != (2,) or principal.shape != (2,) or pose.shape != (4, 4) or
            not np.isfinite(focal).all() or not np.isfinite(principal).all() or
            not np.isfinite(pose).all() or np.any(focal <= 0)):
        raise ValueError('FoundPose capture camera has malformed or nonfinite intrinsics/extrinsics')
    if not np.isclose(focal[0], focal[1], rtol=1e-7, atol=1e-6):
        raise ValueError('FoundPose capture geometry requires the pinned isotropic focal length')
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-5) or
            not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-5) or
            not np.allclose(pose[3], (0., 0., 0., 1.), rtol=0., atol=2e-7)):
        raise ValueError('FoundPose capture camera transform must be a proper rigid transform')

    camera_type = type(upstream_camera)
    high = camera_type(
        width=1120, height=1120,
        f=(float(focal[0]), float(focal[1])),
        c=(float(principal[0] + 1.5), float(principal[1] + 1.5)),
        T_world_from_eye=np.array(pose, copy=True))
    low = camera_type(
        width=280, height=280,
        f=(float(focal[0] / 4.), float(focal[1] / 4.)),
        c=(float(principal[0] / 4.), float(principal[1] / 4.)),
        T_world_from_eye=np.array(pose, copy=True))
    return high, low


def _capture_bank_camera_record(camera):
    return {
        'dimensions': [int(camera.width), int(camera.height)],
        'f': [float(camera.f[0]), float(camera.f[1])],
        'c': [float(camera.c[0]), float(camera.c[1])],
        'T_world_from_eye': np.asarray(camera.T_world_from_eye, dtype=np.float64).tolist(),
    }


def _capture_bank_detach(value, *, depth=0):
    """Copy bounded renderer metadata to JSON-safe builtins, rejecting opaque values."""
    if depth > 12:
        raise ValueError('Capture render metadata nesting is excessive')
    if isinstance(value, _CaptureMapping):
        if len(value) > 128:
            raise ValueError('Capture render metadata mapping is too large')
        copied = {}
        for key, item in value.items():
            if type(key) is not str or len(key) > 128:
                raise ValueError('Capture render metadata keys must be bounded strings')
            copied[key] = _capture_bank_detach(item, depth=depth + 1)
        return copied
    if isinstance(value, (tuple, list)):
        if len(value) > 4096:
            raise ValueError('Capture render metadata sequence is too large')
        return [_capture_bank_detach(item, depth=depth + 1) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError('Capture render metadata text is too large')
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('Capture render metadata cannot contain nonfinite numbers')
        return number
    raise ValueError(f'Unsupported capture render metadata value: {type(value).__name__}')


def _capture_bank_identity_matches(source_identities, source_closure):
    if not isinstance(source_identities, dict) or not isinstance(source_closure, _CaptureMapping):
        raise ValueError('Capture renderer source identities or issued closure are malformed')
    expected = {
        'quality_gotrack': ('bench/quality_gotrack.py', 'quality_gotrack', 'bench/quality_gotrack.py'),
        'quality_render_stability': (
            'bench/quality_render_stability.py', 'quality_render_stability',
            'bench/quality_render_stability.py'),
        'pyrender_renderer': (
            'runtime/pyrender/renderer.py', 'renderer', '/pyrender/renderer.py'),
        'pyrender_offscreen': (
            'runtime/pyrender/offscreen.py', 'offscreen', '/pyrender/offscreen.py'),
    }
    if set(source_identities) != set(expected):
        raise ValueError('Capture renderer source identity set differs from the public renderer contract')
    for identity_name, (closure_path, module_suffix, file_suffix) in expected.items():
        identity = source_identities[identity_name]
        if not isinstance(identity, dict) or set(identity) != {'module', 'file', 'sha256'}:
            raise ValueError(f'Capture renderer source identity {identity_name!r} is malformed')
        module_name = identity['module']
        file_name = identity['file']
        digest = identity['sha256']
        if (type(module_name) is not str or not module_name.endswith(module_suffix) or
                type(file_name) is not str or not _capture_Path(file_name).is_absolute() or
                not file_name.replace('\\', '/').lower().endswith(file_suffix.lower()) or
                type(digest) is not str or len(digest) != 64 or
                any(character not in '0123456789abcdefABCDEF' for character in digest)):
            raise ValueError(f'Capture renderer source identity {identity_name!r} is malformed')
        expected_digest = source_closure.get(closure_path)
        if (type(expected_digest) is not str or
                digest.upper() != expected_digest.upper()):
            raise ValueError(f'Capture renderer source identity {identity_name!r} differs from its issued closure')


def _validate_zero_metadata(metadata, expected_dimensions, previous=None, *,
                             source_closure, allocation_pair_validator):
    """Validate one immediately observed public capture allocation and return its detached copy."""
    metadata = _capture_bank_detach(metadata)
    if not isinstance(metadata, dict):
        raise ValueError('Capture renderer metadata must be a mapping')
    width, height = expected_dimensions
    if (type(width) is not int or type(height) is not int or
            (width, height) not in ((1120, 1120), (280, 280))):
        raise ValueError('Capture renderer metadata expected dimensions are unsupported')
    required = {
        'schema_version', 'render_policy', 'coordinate_mode', 'depth_units', 'dimensions',
        'allocation_generation', 'offscreen_identity', 'framebuffer_fields', 'allocation_pair',
        'allocation_calls', 'framebuffer_complete', 'gl_samples', 'gl_sample_buffers',
        'framebuffer_bindings_restored', 'current_context_released', 'dimension_match',
        'color_storage_policy', 'depth_storage_policy', 'source_identities',
    }
    if not required.issubset(metadata):
        raise ValueError('Capture renderer metadata is missing required public policy fields')
    if (type(metadata['schema_version']) is not int or metadata['schema_version'] != 1 or
            metadata['render_policy'] != 'capture_zero_sample_v1' or
            metadata['coordinate_mode'] != 'integer_centers_v1' or
            metadata['depth_units'] != 'millimetres' or
            metadata['dimensions'] != [width, height] or
            type(metadata['allocation_generation']) is not int or
            type(metadata['allocation_generation']) is bool or
            type(metadata['offscreen_identity']) is not int or metadata['offscreen_identity'] <= 0):
        raise ValueError('Capture renderer metadata differs from the issued zero-sample policy or dimensions')
    for name in ('framebuffer_complete', 'framebuffer_bindings_restored',
                 'current_context_released', 'dimension_match'):
        if type(metadata[name]) is not bool or metadata[name] is not True:
            raise ValueError(f'Capture renderer metadata {name!r} did not pass')
    for name in ('gl_samples', 'gl_sample_buffers'):
        if type(metadata[name]) is not int or metadata[name] != 0:
            raise ValueError(f'Capture renderer metadata {name!r} is not zero')
    if (metadata['color_storage_policy'] !=
            'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage' or
            metadata['depth_storage_policy'] !=
            'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage'):
        raise ValueError('Capture renderer storage policy differs from the issued ordinary-renderbuffer policy')
    fields = metadata['framebuffer_fields']
    if not isinstance(fields, dict) or set(fields) != {
            'multisample_draw_fbo', 'single_sample_read_fbo', 'multisample_dimensions'}:
        raise ValueError('Capture renderer framebuffer metadata is malformed')
    if (any(type(fields[name]) is not int or fields[name] <= 0
            for name in ('multisample_draw_fbo', 'single_sample_read_fbo')) or
            fields['multisample_dimensions'] != [width, height]):
        raise ValueError('Capture renderer framebuffer identities/dimensions are malformed')
    generation = metadata['allocation_generation']
    if previous is None:
        if generation != 1:
            raise ValueError('Fresh capture renderer allocation generation must begin at one')
    elif (type(previous) is not dict or type(previous.get('allocation_generation')) is not int or
          type(previous.get('dimensions')) is not list or
          previous.get('dimensions') == [width, height] or
          generation != previous['allocation_generation'] + 1):
        raise ValueError('Capture renderer resize generation did not advance across the actual dimension transition')
    calls = metadata['allocation_calls']
    if not isinstance(calls, list) or not calls or not callable(allocation_pair_validator):
        raise ValueError('Capture renderer has no independently verifiable allocation-call pair')
    validated_pair = _capture_bank_detach(
        allocation_pair_validator(calls, dimensions=(width, height)))
    if (not isinstance(validated_pair, dict) or validated_pair.get('passed') is not True or
            validated_pair.get('dimensions') != [width, height] or
            metadata['allocation_pair'] != validated_pair):
        raise ValueError('Capture renderer allocation pair does not match its actual allocation-call rows')
    _capture_bank_identity_matches(metadata['source_identities'], source_closure)
    return metadata


def _capture_bank_tensor_array(value, expected_shape, expected_dtype, label):
    device = getattr(value, 'device', None)
    if device is not None and getattr(device, 'type', None) != 'cpu':
        raise ValueError(f'Capture render {label} must be returned on CPU')
    if device is None and not isinstance(value, np.ndarray):
        raise ValueError(f'Capture render {label} must expose an explicit CPU tensor device')
    shape = tuple(int(dimension) for dimension in getattr(value, 'shape', ()))
    if shape != tuple(expected_shape):
        raise ValueError(f'Capture render {label} shape {shape!r} differs from {tuple(expected_shape)!r}')
    raw = value.detach().numpy() if callable(getattr(value, 'detach', None)) else np.asarray(value)
    if raw.dtype != expected_dtype:
        raise ValueError(f'Capture render {label} dtype {raw.dtype} differs from {expected_dtype}')
    return raw


def _capture_bank_render_arrays(result, dimensions, render_type):
    width, height = dimensions
    if not isinstance(result, _CaptureMapping):
        raise ValueError('Capture public renderer result must be a mapping')
    expected_keys = {render_type.COLOR, render_type.DEPTH, render_type.MASK}
    if set(result) != expected_keys:
        raise ValueError('Capture public renderer did not return exactly COLOR, DEPTH and MASK')
    color = _capture_bank_tensor_array(result[render_type.COLOR], (height, width, 3), np.dtype('float32'), 'COLOR')
    depth = _capture_bank_tensor_array(result[render_type.DEPTH], (height, width), np.dtype('float32'), 'DEPTH')
    mask = _capture_bank_tensor_array(result[render_type.MASK], (height, width), np.dtype('bool'), 'MASK')
    if (not np.isfinite(color).all() or np.any(color < 0.) or np.any(color > 1.) or
            not np.isfinite(depth).all() or np.any(depth < 0.) or
            not np.array_equal(mask, depth > 0.)):
        raise ValueError('Capture renderer COLOR/DEPTH/MASK values violate the public return contract')
    summaries = {
        'COLOR': {'shape': list(color.shape), 'dtype': str(color.dtype), 'byte_count': int(color.nbytes)},
        'DEPTH': {'shape': list(depth.shape), 'dtype': str(depth.dtype), 'byte_count': int(depth.nbytes)},
        'MASK': {'shape': list(mask.shape), 'dtype': str(mask.dtype), 'byte_count': int(mask.nbytes)},
    }
    return color, depth, mask, summaries


class _CaptureBankRenderProxy:
    """Capture-only paired COLOR/geometry adapter around the unchanged upstream builder."""

    def __init__(self, renderer, *, expected_obj_id, expected_asset_sha256,
                 render_type, torch_module, source_closure, allocation_pair_validator):
        self._renderer = renderer
        self._render_type = render_type
        self._torch = torch_module
        self._source_closure = source_closure
        self._allocation_pair_validator = allocation_pair_validator
        self.expected_obj_id = expected_obj_id
        self.expected_asset_sha256 = expected_asset_sha256.upper()
        self.logical_attempted = 0
        self.logical_returned = 0
        self.completed_pairs = 0
        self.native_attempted = 0
        self.native_returned = 0
        self.native_verified = 0
        self.pairs = []
        self.events = []
        self._previous_metadata = None

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def _native_call(self, *, role, render_types, obj_id, camera, background):
        self.native_attempted += 1
        event = {
            'native_call_index': self.native_attempted,
            'role': role,
            'dimensions': [int(camera.width), int(camera.height)],
            'camera': _capture_bank_camera_record(camera),
            'object_id': obj_id,
            'asset_sha256': self.expected_asset_sha256,
            'render_types': [name for name in ('COLOR', 'DEPTH', 'MASK')
                             if any(value == getattr(self._render_type, name)
                                    for value in render_types)],
            'return_tensors': True,
            'background': [float(value) for value in background],
            'attempted': True,
            'returned': False,
            'verified': False,
            'duration_ms': None,
            'array_summaries': None,
            'render_policy_metadata': None,
            'error': None,
        }
        self.events.append(event)
        started = _capture_time.perf_counter_ns()
        try:
            result = self._renderer.render_object_model(
                obj_id=obj_id, camera_model_c2w=camera,
                render_types=list(render_types), return_tensors=True,
                background=background)
        except BaseException as exc:
            event['duration_ms'] = (_capture_time.perf_counter_ns() - started) / 1e6
            event['error'] = _capture_bank_error(exc, f'native_{role}')
            raise
        self.native_returned += 1
        event['returned'] = True
        event['duration_ms'] = (_capture_time.perf_counter_ns() - started) / 1e6
        try:
            raw_metadata = self._renderer.render_policy_metadata
        except BaseException as exc:
            event['error'] = _capture_bank_error(exc, f'{role}_metadata')
            raise
        if raw_metadata is None:
            event['error'] = _capture_bank_error(
                ValueError('Public renderer returned no current render-policy metadata'),
                f'{role}_metadata')
            raise ValueError(f'Capture {role} render returned without current public metadata')
        try:
            metadata = _capture_bank_detach(raw_metadata)
            if len(_capture_json.dumps(
                    metadata, separators=(',', ':'), allow_nan=False).encode('utf-8')) > 8192:
                raise ValueError('Capture render metadata exceeds the bounded 8 KiB per-call receipt size')
            event['render_policy_metadata'] = metadata
            metadata = _validate_zero_metadata(
                metadata, (int(camera.width), int(camera.height)), self._previous_metadata,
                source_closure=self._source_closure,
                allocation_pair_validator=self._allocation_pair_validator)
            arrays = _capture_bank_render_arrays(
                result, (int(camera.width), int(camera.height)), self._render_type)
            event['array_summaries'] = arrays[3]
            event['verified'] = True
            self.native_verified += 1
            self._previous_metadata = metadata
            return arrays
        except BaseException as exc:
            event['error'] = _capture_bank_error(exc, f'{role}_verification')
            raise

    def render_object_model(self, obj_id, camera_model_c2w, render_types=None,
                            return_tensors=False, background=None, **kwargs):
        self.logical_attempted += 1
        logical_index = self.logical_attempted - 1
        pair = {
            'logical_index': logical_index,
            'status': 'attempted',
            'input_camera': None,
            'native_call_indices': [],
            'error': None,
        }
        self.pairs.append(pair)
        try:
            if logical_index >= _BANK_TEMPLATE_COUNT:
                raise ValueError('FoundPose capture builder made more than 798 logical template requests')
            if obj_id != self.expected_obj_id:
                raise ValueError('FoundPose capture builder requested a different object id')
            if (type(return_tensors) is not bool or return_tensors is not True or kwargs or
                    not isinstance(render_types, (tuple, list)) or len(render_types) != 3 or
                    set(render_types) != {self._render_type.COLOR, self._render_type.DEPTH,
                                          self._render_type.MASK}):
                raise ValueError('FoundPose upstream render request differs from its pinned tensor/type contract')
            background_array = np.asarray(background, dtype=np.float64)
            if (background_array.shape != (3,) or not np.isfinite(background_array).all() or
                    not np.array_equal(background_array, np.array([.5, .5, .5]))):
                raise ValueError('FoundPose capture rendering requires the exact gray RGB background')
            high_camera, low_camera = _capture_bank_camera_pair(camera_model_c2w)
            pair['input_camera'] = _capture_bank_camera_record(camera_model_c2w)
            pair['high_color_camera'] = _capture_bank_camera_record(high_camera)
            pair['low_geometry_camera'] = _capture_bank_camera_record(low_camera)
            high_index = len(self.events) + 1
            pair['native_call_indices'].append(high_index)
            high_color, _, _, _ = self._native_call(
                role='color_ssaa4', render_types=[self._render_type.COLOR],
                obj_id=obj_id, camera=high_camera, background=background_array)
            low_index = len(self.events) + 1
            pair['native_call_indices'].append(low_index)
            _, low_depth, low_mask, _ = self._native_call(
                role='depth_mask_ssaa1',
                render_types=[self._render_type.DEPTH, self._render_type.MASK],
                obj_id=obj_id, camera=low_camera, background=background_array)
            depth_carrier = self._torch.repeat_interleave(
                self._torch.repeat_interleave(
                    self._renderer_tensor(low_depth), 4, dim=0), 4, dim=1)
            mask_carrier = self._torch.repeat_interleave(
                self._torch.repeat_interleave(
                    self._renderer_tensor(low_mask), 4, dim=0), 4, dim=1)
            if (tuple(depth_carrier.shape) != (1120, 1120) or
                    tuple(mask_carrier.shape) != (1120, 1120)):
                raise ValueError('Capture low-resolution geometry carrier did not expand by exact 4x block transport')
            expanded_depth = np.repeat(np.repeat(low_depth, 4, axis=0), 4, axis=1)
            expanded_mask = np.repeat(np.repeat(low_mask, 4, axis=0), 4, axis=1)
            actual_depth = _capture_bank_tensor_array(
                depth_carrier, (1120, 1120), np.dtype('float32'), 'expanded DEPTH')
            actual_mask = _capture_bank_tensor_array(
                mask_carrier, (1120, 1120), np.dtype('bool'), 'expanded MASK')
            if (not np.array_equal(actual_depth, expanded_depth) or
                    not np.array_equal(actual_mask, expanded_mask)):
                raise ValueError('Capture geometry carrier differs from exact 4x nearest block transport')
            pair['carrier'] = {
                'transport': 'repeat_interleave_4_rows_then_4_columns',
                'depth_dimensions': list(depth_carrier.shape),
                'mask_dimensions': list(mask_carrier.shape),
                'depth_dtype': str(getattr(depth_carrier, 'dtype', '')),
                'mask_dtype': str(getattr(mask_carrier, 'dtype', '')),
            }
            pair['status'] = 'completed'
            self.logical_returned += 1
            self.completed_pairs += 1
            return {
                self._render_type.COLOR: self._renderer_tensor(high_color),
                self._render_type.DEPTH: depth_carrier,
                self._render_type.MASK: mask_carrier,
            }
        except BaseException as exc:
            pair['status'] = 'failed'
            pair['error'] = _capture_bank_error(exc, f'logical_template_{logical_index}')
            raise

    def _renderer_tensor(self, value):
        """Keep public Torch tensors unchanged; allow NumPy-only independent fakes."""
        if isinstance(value, np.ndarray):
            converter = getattr(self._torch, 'as_tensor', None)
            return converter(value) if callable(converter) else value
        return value

    def assert_complete(self):
        expected = _BANK_TEMPLATE_COUNT
        actual = (self.logical_attempted, self.logical_returned, self.completed_pairs,
                  self.native_attempted, self.native_returned, self.native_verified)
        if actual != (expected, expected, expected, expected * 2, expected * 2, expected * 2):
            raise ValueError(f'FoundPose paired-render evidence is incomplete: {actual!r}')
        if len(self.pairs) != expected or len(self.events) != expected * 2:
            raise ValueError('FoundPose paired-render evidence row counts differ from the fixed recipe')
        for index, pair in enumerate(self.pairs):
            high, low = self.events[2 * index:2 * index + 2]
            if (pair.get('logical_index') != index or pair.get('status') != 'completed' or
                    pair.get('native_call_indices') != [2 * index + 1, 2 * index + 2] or
                    high.get('native_call_index') != 2 * index + 1 or
                    high.get('role') != 'color_ssaa4' or high.get('dimensions') != [1120, 1120] or
                    high.get('render_types') != ['COLOR'] or
                    low.get('native_call_index') != 2 * index + 2 or
                    low.get('role') != 'depth_mask_ssaa1' or low.get('dimensions') != [280, 280] or
                    low.get('render_types') != ['DEPTH', 'MASK'] or
                    any(event.get('attempted') is not True or event.get('returned') is not True or
                        event.get('verified') is not True or
                        event.get('render_policy_metadata', {}).get('dimensions') != event.get('dimensions')
                        for event in (high, low))):
                raise ValueError(f'FoundPose paired-render evidence failed its actual call join at template {index}')
        evidence_bytes = len(_capture_json.dumps(
            {'pairs': self.pairs, 'events': self.events},
            separators=(',', ':'), allow_nan=False).encode('utf-8'))
        if evidence_bytes > _BANK_MAX_EVIDENCE_BYTES:
            raise ValueError('FoundPose paired-render evidence exceeds the bounded 16 MiB receipt allowance')


def _capture_bank_error(exc, location):
    return {
        'type': type(exc).__name__,
        'message': str(exc)[:1000],
        'failure_location': location,
    }


def _capture_bank_constructor_record(renderer, asset_receipt, flags):
    arguments = {
        'unlit': flags['unlit_templates'],
        'disable_multisampling': False,
        'coordinate_mode': 'integer_centers_v1',
        'render_policy': 'capture_zero_sample_v1',
    }
    observed = {
        'unlit': renderer.unlit,
        'disable_multisampling': renderer.disable_multisampling,
        'coordinate_mode': renderer.coordinate_mode,
        'render_policy': renderer.render_policy,
    }
    if (any(type(observed[name]) is not type(arguments[name]) or
            observed[name] != arguments[name] for name in arguments) or
            renderer.obj_id != asset_receipt.object_id or
            renderer.asset_sha256.upper() != asset_receipt.asset_sha256.upper()):
        raise ValueError('FoundPose renderer differs from its authenticated asset and recipe')
    background = np.asarray(renderer.scene.bg_color, dtype=np.float64)
    ambient = np.asarray(renderer.scene.ambient_light, dtype=np.float64)
    if (background.shape != (4,) or ambient.shape != (4,) or
            not np.isfinite(background).all() or not np.isfinite(ambient).all() or
            not np.allclose(background, (.5, .5, .5, 0.), rtol=0., atol=1e-7) or
            not np.allclose(ambient, (.02, .02, .02, 1.), rtol=0., atol=1e-7)):
        raise ValueError('FoundPose renderer scene metadata is malformed')
    return {
        'api': 'bench.quality_gotrack.TexturedRenderer',
        'object_id': renderer.obj_id,
        'asset_sha256': renderer.asset_sha256,
        'arguments': arguments,
        'observed_configuration': {
            **observed,
            'scene_bg_rgba': background.tolist(),
            'ambient_light_rgba': ambient.tolist(),
            'spotlight_intensity': 2.4,
            'spotlight_inner_cone': _capture_math.pi / 16.,
            'spotlight_outer_cone': _capture_math.pi / 6.,
        },
    }


def _capture_bank_payload_shapes(state, asset_receipt):
    expected_fields = {
        'vertices', 'feat_vectors', 'feat_to_template_ids', 'centroids',
        'descs', 'idfs', 'projector', 'template_count', 'asset_sha256',
    }
    if type(state) is not dict or set(state) != expected_fields:
        raise ValueError('FoundPose bank returned an unexpected payload field set')
    if type(state['template_count']) is not int or state['template_count'] != _BANK_TEMPLATE_COUNT:
        raise ValueError('FoundPose bank did not produce exactly 798 templates')
    if (type(state['asset_sha256']) is not str or
            state['asset_sha256'].upper() != asset_receipt.asset_sha256.upper()):
        raise ValueError('FoundPose bank payload asset identity differs from its issued GLB')

    tensors = {}
    shapes = {}
    for name in ('vertices', 'feat_vectors', 'feat_to_template_ids', 'centroids', 'descs', 'idfs'):
        value = state[name]
        raw_shape = getattr(value, 'shape', None)
        if raw_shape is None:
            raise ValueError(f'FoundPose bank payload field {name!r} has no measured shape')
        shape = tuple(int(dimension) for dimension in raw_shape)
        dtype = str(getattr(value, 'dtype', ''))
        shapes[name] = shape
        count = getattr(value, 'numel', None)
        element_size = getattr(value, 'element_size', None)
        if callable(count) and callable(element_size):
            byte_count = int(count()) * int(element_size())
        elif isinstance(value, np.ndarray):
            byte_count = int(value.nbytes)
        else:
            raise ValueError(f'FoundPose bank payload field {name!r} has no measured byte count')
        tensors[name] = {'shape': list(shape), 'dtype': dtype, 'byte_count': byte_count}
    vertices = shapes['vertices']
    features = shapes['feat_vectors']
    template_ids = shapes['feat_to_template_ids']
    centroids = shapes['centroids']
    if (len(vertices) != 2 or vertices[1] != 3 or vertices[0] <= 0 or
            vertices[0] > _BANK_MAX_FEATURE_ROWS or
            len(features) != 2 or features[1] != 256 or
            len(template_ids) != 1 or
            vertices[0] != features[0] or features[0] != template_ids[0] or
            len(centroids) != 2 or centroids != (2048, 256) or
            shapes['descs'] != (798, 2048) or shapes['idfs'] != (2048,)):
        raise ValueError('FoundPose bank tensor layouts differ from the fixed 256-D/2048-word recipe')
    expected_dtypes = {
        'vertices': 'float32', 'feat_vectors': 'float32',
        'feat_to_template_ids': 'int64', 'centroids': 'float32',
        'descs': 'float32', 'idfs': 'float32',
    }
    for name, expected in expected_dtypes.items():
        if not tensors[name]['dtype'].lower().endswith(expected):
            raise ValueError(f'FoundPose bank tensor {name!r} dtype differs from {expected}')
    projector = state['projector']
    fitted = getattr(projector, 'pca', None)
    if (fitted is None or getattr(projector, 'whiten', None) is not False or
            getattr(projector, 'n_components', None) != 256 or
            getattr(fitted, 'n_components_', None) != 256 or
            getattr(fitted, 'n_features_in_', None) != 384 or
            getattr(fitted, 'whiten', None) is not False):
        raise ValueError('FoundPose bank PCA projector is not the pinned non-whitened fitted projector')
    projector_fields = {}
    for name, expected_shape in (
            ('components_', (256, 384)), ('mean_', (384,)),
            ('explained_variance_', (256,)), ('explained_variance_ratio_', (256,)),
            ('singular_values_', (256,))):
        value = getattr(fitted, name, None)
        shape = tuple(int(dimension) for dimension in getattr(value, 'shape', ()))
        if shape != expected_shape:
            raise ValueError(f'FoundPose PCA fitted field {name!r} has an unexpected shape')
        projector_fields[name] = {'shape': list(shape), 'dtype': str(value.dtype),
                                  'byte_count': int(value.nbytes if isinstance(value, np.ndarray)
                                                    else value.numel() * value.element_size())}
    noise_variance = float(fitted.noise_variance_)
    if not np.isfinite(noise_variance) or noise_variance < 0.:
        raise ValueError('FoundPose PCA noise variance is not a finite nonnegative scalar')
    projector_fields['noise_variance_'] = {
        'shape': [], 'dtype': str(type(fitted.noise_variance_).__name__),
        'value': noise_variance, 'byte_count': 8,
    }
    total_payload_bytes = sum(row['byte_count'] for row in tensors.values()) + sum(
        row['byte_count'] for row in projector_fields.values())
    if total_payload_bytes > _BANK_ARTIFACT_CEILING_BYTES:
        raise ValueError('FoundPose bank payload tensor/PCA bytes exceed the conservative artifact ceiling')
    return {
        'template_count': state['template_count'],
        'asset_sha256': state['asset_sha256'],
        'tensor_payloads': tensors,
        'projector_type': f'{type(projector).__module__}.{type(projector).__qualname__}',
        'projector_whiten': False,
        'projector_fitted_fields': projector_fields,
        'measured_tensor_and_projector_bytes': total_payload_bytes,
        'maximum_feature_rows': _BANK_MAX_FEATURE_ROWS,
    }


def build_capture_bank(verified_plan, resources, output, device):
    """Build a pending FoundPose bank from current public issued capture resources."""
    (provenance, capture, recipe, flags, manifest, asset_path, checkpoint_sha,
     target, cache_bytes_before, free_bytes_before, reserve_record) = _capture_bank_preflight(
        verified_plan, resources, output, device)
    from . import quality_assets

    started_ns = _capture_time.perf_counter_ns()
    started_at = _capture_datetime.now(_capture_timezone.utc).isoformat()
    pending = {
        'schema_version': 1,
        'kind': 'quality-capture-foundpose-bank-pending-v1',
        'status': 'pending',
        'outcome': 'pending',
        'execution_id': _capture_uuid.uuid4().hex.upper(),
        'stage': 'banks',
        'resource_kind': capture['output_resource']['resource_kind'],
        'resource_key': capture['output_resource']['resource_key'],
        'asset_receipt_sha256': resources.asset_receipt.receipt_sha256,
        'asset_sha256': resources.asset_receipt.asset_sha256,
        'unit_receipt_sha256': resources.asset_receipt.unit_receipt_sha256,
        'renderer_proof_sha256': resources.asset_receipt.renderer_proof_sha256,
        'object_id': resources.asset_receipt.object_id,
        'object_name': resources.asset_receipt.object_name,
        'input_manifest_sha256': capture['input_manifest_sha256'],
        'capture_table_sha256': capture['capture_table_sha256'],
        'output_camera_sha256': capture['output_camera_sha256'],
        'device': device,
        'checkpoint_sha256': checkpoint_sha,
        'recipe': recipe,
        'source_closure': capture['output_resource']['producer_source_closure'],
        'used_resources': capture['used_resources'],
        'render_implementation': 'capture_foundpose_paired_ssaa4_v1',
        'registration_policy': 'pinned_integer_grid_isotropic_lift_v1',
        'feature_sampling_convention': 'pinned_uv_2p_over_size_minus1_align_corners_false',
        'output_path': target.relative_to(quality_assets.CAPTURE_CACHE.resolve(strict=True)).as_posix(),
        'started_at_utc': started_at,
        'ended_at_utc': None,
        'timings_ms': {
            'worker_total': None,
            'network_load': None,
            'renderer_construct': None,
            'bank_build_and_write': None,
            'renderer_close': None,
        },
        'render_evidence': {
            'logical_attempted': 0,
            'logical_returned': 0,
            'completed_pairs': 0,
            'native_attempted': 0,
            'native_returned': 0,
            'native_verified': 0,
            'pairs': [],
            'native_events': [],
        },
        'constructor_record': None,
        'payload': None,
        'cache_bytes_before': cache_bytes_before,
        'free_bytes_before': free_bytes_before,
        'prospective_budget_reserve': reserve_record,
        'post_build_budget_observation': None,
        'cache_bytes_after': None,
        'artifact_sha256': None,
        'artifact_byte_count': None,
        'renderer_created': False,
        'renderer_close_attempted': False,
        'renderer_closed': False,
        'render_policy_metadata_cleared': None,
        'render_policy_metadata_clear_error': None,
        'failure_location': None,
        'error': None,
        'close_error': None,
    }
    renderer = None
    observer = None
    network = None
    torch = None
    location = 'network_load'
    try:
        from .quality_gotrack import TexturedRenderer, load_network

        phase_started = _capture_time.perf_counter_ns()
        try:
            network = load_network(device)
            import torch as torch_module
            torch = torch_module
            if device == 'cuda':
                torch.cuda.synchronize()
        finally:
            pending['timings_ms']['network_load'] = (
                _capture_time.perf_counter_ns() - phase_started) / 1e6

        location = 'renderer_construct'
        phase_started = _capture_time.perf_counter_ns()
        try:
            renderer = TexturedRenderer(
                asset_path,
                resources.asset_receipt.object_id,
                unlit=flags['unlit_templates'],
                disable_multisampling=False,
                coordinate_mode='integer_centers_v1',
                render_policy='capture_zero_sample_v1',
            )
        finally:
            pending['timings_ms']['renderer_construct'] = (
                _capture_time.perf_counter_ns() - phase_started) / 1e6
        pending['renderer_created'] = True
        pending['constructor_record'] = _capture_bank_constructor_record(
            renderer, resources.asset_receipt, flags)
        if renderer.asset_sha256.upper() != resources.asset_receipt.asset_sha256.upper():
            raise ValueError('Loaded renderer asset bytes changed after capture preflight')

        upstream_path()
        from utils.renderer_base import RenderType
        from .quality_render_stability import validate_capture_allocation_pairs
        observer = _CaptureBankRenderProxy(
            renderer,
            expected_obj_id=resources.asset_receipt.object_id,
            expected_asset_sha256=resources.asset_receipt.asset_sha256,
            render_type=RenderType,
            torch_module=torch,
            source_closure=capture['output_resource']['producer_source_closure'],
            allocation_pair_validator=validate_capture_allocation_pairs)
        location = 'bank_build_and_write'
        phase_started = _capture_time.perf_counter_ns()
        try:
            state = build_bank(observer, network.backbone, device, target)
            observer.assert_complete()
            if device == 'cuda':
                torch.cuda.synchronize()
        finally:
            pending['timings_ms']['bank_build_and_write'] = (
                _capture_time.perf_counter_ns() - phase_started) / 1e6
        location = 'payload_validation'
        pending['payload'] = _capture_bank_payload_shapes(state, resources.asset_receipt)
        if not target.is_file() or target.is_symlink():
            raise ValueError('FoundPose bank builder did not leave a regular trial artifact')
        pending['artifact_sha256'] = quality_assets.digest(target).upper()
        pending['artifact_byte_count'] = target.stat().st_size
        if (pending['artifact_byte_count'] <= 0 or
                pending['artifact_byte_count'] > _BANK_ARTIFACT_CEILING_BYTES):
            raise ValueError('FoundPose bank artifact is empty or exceeds its 512 MiB ceiling')
    except Exception as exc:
        pending['error'] = _capture_bank_error(exc, location)
        pending['failure_location'] = location
    finally:
        network = None
        if renderer is not None:
            pending['renderer_close_attempted'] = True
            location = 'renderer_close'
            phase_started = _capture_time.perf_counter_ns()
            try:
                renderer.close()
                pending['renderer_closed'] = True
            except Exception as exc:
                pending['close_error'] = _capture_bank_error(exc, location)
                if pending['error'] is None:
                    pending['failure_location'] = location
            finally:
                pending['timings_ms']['renderer_close'] = (
                    _capture_time.perf_counter_ns() - phase_started) / 1e6
                try:
                    pending['render_policy_metadata_cleared'] = (
                        renderer.render_policy_metadata is None)
                    if pending['render_policy_metadata_cleared'] is not True:
                        raise ValueError('Capture renderer retained policy metadata after close')
                except Exception as exc:
                    pending['render_policy_metadata_cleared'] = False
                    pending['render_policy_metadata_clear_error'] = _capture_bank_error(
                        exc, 'renderer_close_metadata_validation')
                    if pending['error'] is None:
                        pending['failure_location'] = 'renderer_close_metadata_validation'
                        pending['error'] = pending['render_policy_metadata_clear_error']
        if observer is not None:
            pending['render_evidence'] = {
                'logical_attempted': observer.logical_attempted,
                'logical_returned': observer.logical_returned,
                'completed_pairs': observer.completed_pairs,
                'native_attempted': observer.native_attempted,
                'native_returned': observer.native_returned,
                'native_verified': observer.native_verified,
                'pairs': observer.pairs,
                'native_events': observer.events,
            }

    try:
        pending['cache_bytes_after'] = sum(
            path.stat().st_size
            for path in quality_assets.CACHE.resolve(strict=True).rglob('*')
            if path.is_file())
        free_bytes_after = _capture_shutil.disk_usage(
            quality_assets.CACHE.resolve(strict=True)).free
        pending['post_build_budget_observation'] = {
            'cache_bytes_after': pending['cache_bytes_after'],
            'free_bytes_after': free_bytes_after,
            'remaining_reserve_bytes': _BANK_POST_BUILD_RESERVE_BYTES,
            'cache_bytes_with_remaining_reserve': (
                pending['cache_bytes_after'] + _BANK_POST_BUILD_RESERVE_BYTES),
            'free_bytes_after_remaining_reserve': (
                free_bytes_after - _BANK_POST_BUILD_RESERVE_BYTES),
        }
        if (pending['cache_bytes_after'] + _BANK_POST_BUILD_RESERVE_BYTES > quality_assets.BUDGET or
                free_bytes_after - _BANK_POST_BUILD_RESERVE_BYTES <= 20 * 1024**3):
            pending['failure_location'] = pending['failure_location'] or 'cache_budget_validation'
            pending['error'] = pending['error'] or _capture_bank_error(
                ValueError('FoundPose bank plus remaining finalization reserve exceeds cache/free-space bounds'),
                'cache_budget_validation')
        evidence_bytes = len(_capture_json.dumps(
            pending['render_evidence'], separators=(',', ':'), allow_nan=False).encode('utf-8'))
        if evidence_bytes > _BANK_MAX_EVIDENCE_BYTES:
            pending['failure_location'] = pending['failure_location'] or 'render_evidence_budget'
            pending['error'] = pending['error'] or _capture_bank_error(
                ValueError('FoundPose paired-render evidence exceeds the 16 MiB receipt allowance'),
                'render_evidence_budget')
    except Exception as exc:
        pending['failure_location'] = pending['failure_location'] or 'cache_budget_validation'
        pending['error'] = pending['error'] or _capture_bank_error(
            exc, 'cache_budget_validation')

    pending['ended_at_utc'] = _capture_datetime.now(_capture_timezone.utc).isoformat()
    pending['timings_ms']['worker_total'] = (
        _capture_time.perf_counter_ns() - started_ns) / 1e6
    render_evidence = pending['render_evidence']
    full_pair_counts = (
        render_evidence['logical_attempted'] == _BANK_TEMPLATE_COUNT and
        render_evidence['logical_returned'] == _BANK_TEMPLATE_COUNT and
        render_evidence['completed_pairs'] == _BANK_TEMPLATE_COUNT and
        render_evidence['native_attempted'] == 2 * _BANK_TEMPLATE_COUNT and
        render_evidence['native_returned'] == 2 * _BANK_TEMPLATE_COUNT and
        render_evidence['native_verified'] == 2 * _BANK_TEMPLATE_COUNT)
    pending['outcome'] = (
        'measured_success' if pending['error'] is None and
        pending['close_error'] is None and pending['renderer_closed'] and
        pending['render_policy_metadata_cleared'] is True and full_pair_counts and
        pending['payload'] is not None and pending['artifact_sha256'] is not None and
        type(pending['artifact_byte_count']) is int and
        0 < pending['artifact_byte_count'] <= _BANK_ARTIFACT_CEILING_BYTES
        else 'failed')
    return pending
