"""Full-image CNOS detection: FastSAM proposals + DINOv2-L template matching.

Uses the repository's proposal/crop/matching mathematics with batch size one.
Runs in a process separate from SAM2 and GoTrack; no projected-mask fallback.
"""
import json
import numpy as np
from .quality_assets import CACHE, digest
from .quality_gotrack import upstream_path, TexturedRenderer


def load_descriptor(device):
    upstream_path()
    import torch
    import dinov2.hub.backbones as backbones
    path = CACHE/'checkpoints/dinov2_vitl14_pretrain.pth'
    lock = json.loads((CACHE/'lock.json').read_text())
    if digest(path) != lock['models'][path.name]['sha256']: raise ValueError('CNOS DINO checkpoint changed')
    model = backbones.dinov2_vitl14(pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.eval().to(device)


def camera_for(manifest):
    from utils import structs
    k = np.array(manifest['intrinsics']); w, h = manifest['native_resolution']
    return structs.PinholePlaneCameraModel(width=w, height=h, f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=np.eye(4))


def build_descriptors(bundle, device='cuda', unlit=False, grayscale=False, output=None):
    upstream_path()
    import torch
    from utils import data_util, misc, crop_generation, im_util
    manifest = json.loads((bundle/'input.json').read_text())
    if (unlit or grayscale) and output is None: raise ValueError('Experimental banks require a separate output path')
    renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'],unlit=unlit,disable_multisampling=unlit)
    model = load_descriptor(device); camera = camera_for(manifest)
    dummy = torch.zeros(1, 3, camera.height, camera.width)
    descriptors = []
    try:
        with torch.inference_mode():
            for view in misc.sample_views(min_n_views=57, radius=923.075, mode='fibonacci')[0]:
                pose = np.eye(4); pose[:3, :3] = view['R']; pose[:3, 3] = np.asarray(view['t']).ravel()
                data, crops, _ = data_util.compute_gotrack_inputs_from_init_poses(input_rgbs=dummy, input_cameras=[camera],
                    init_poses_cam_from_model=torch.from_numpy(pose.astype(np.float32))[None], renderer=renderer,
                    obj_ids=[manifest['object_id']], object_vertices=[renderer.vertices_m*1000], crop_size=(280, 280),
                    crop_rel_pad=.05, cropping_type='perspective_2d_box', ssaa_factor=1., background_type='gray')
                t = data['templates']; boxes = im_util.masks_to_xyxy_boxes(t.masks)
                tight = crop_generation.batch_cropping_from_bbox(source_images=(t.rgbs.permute(0, 2, 3, 1)*255).numpy(),
                    source_cameras=crops, source_xyxy_bboxes=boxes.numpy(), source_masks=t.masks[..., None].numpy(),
                    crop_size=(280, 280), crop_rel_pad=.05)
                rgb = tight.rgbs*tight.masks[:, None]
                if grayscale:
                    gray=(rgb[:,0]*.299+rgb[:,1]*.587+rgb[:,2]*.114)[:,None]
                    rgb=gray.repeat(1,3,1,1)
                desc = model.forward_features(im_util.im_normalize(rgb).to(device))['x_norm_clstoken']
                descriptors.append(desc[0].cpu().numpy())
    finally: renderer.close()
    target = output or CACHE/'banks'/f'{manifest["object"]}-cnos.npz'; target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, descriptors=np.stack(descriptors), asset_sha256=renderer.asset_sha256,
        unlit=unlit,grayscale=grayscale)
    return target


def detect(bundle, rgb, device='cuda', prior_mask=None, prior_age=None):
    upstream_path()
    import torch
    from model.config import FastSAMOpts
    from utils import fastsam_util, crop_generation, im_util
    manifest = json.loads((bundle/'input.json').read_text())
    path = CACHE/'checkpoints/FastSAM-x.pt'; lock = json.loads((CACHE/'lock.json').read_text())
    if digest(path) != lock['models'][path.name]['sha256']: raise ValueError('FastSAM checkpoint changed')
    bank_path = CACHE/'banks'/f'{manifest["object"]}-cnos.npz'
    if not bank_path.exists(): raise RuntimeError('CNOS template descriptors must be built before segmentation')
    with np.load(bank_path, allow_pickle=False) as bank:
        if str(bank['asset_sha256']) != digest(bundle/manifest['asset']): raise ValueError('Wrong CNOS object bank')
        references = torch.from_numpy(bank['descriptors'].copy()).to(device)
    segmentor = fastsam_util.FastSAM(FastSAMOpts(model_path=str(path), conf_threshold=.25, iou_threshold=.9,
                                                max_det=200, im_width_size=640, verbose=False))
    segmentor.set_device(torch.device(device))
    # Ultralytics accepts BGR ndarray; inference boundary remains RGB.
    proposals = segmentor.generate_detection_proposals(rgb[..., ::-1].copy())
    if not len(proposals): return None, {'reason': 'cnos_no_proposals'}
    keep = im_util.filter_noisy_detections(proposals.boxes, proposals.masks, min_box_size=.05,
                                          min_mask_size=3e-4, width=rgb.shape[1], height=rgb.shape[0])
    proposals = proposals[keep]
    if not len(proposals): return None, {'reason': 'cnos_no_supported_proposals'}
    crops = crop_generation.batch_cropping_from_bbox(source_images=[rgb]*len(proposals),
        source_cameras=[camera_for(manifest)]*len(proposals), source_xyxy_bboxes=proposals.boxes.cpu().numpy(),
        source_masks=proposals.masks[..., None].cpu().numpy(), crop_size=(280, 280), crop_rel_pad=.05)
    # Do not keep FastSAM weights on GPU while loading the large descriptor net.
    proposals_masks = proposals.masks.cpu().numpy(); del segmentor, proposals
    if device == 'cuda': torch.cuda.empty_cache()
    model = load_descriptor(device); scores = []
    with torch.inference_mode():
        refs = torch.nn.functional.normalize(references, dim=1)
        for rgb_crop, mask in zip(crops.rgbs, crops.masks):
            q = model.forward_features(im_util.im_normalize((rgb_crop*mask)[None]).to(device))['x_norm_clstoken']
            similarities = torch.nn.functional.normalize(q, dim=1)@refs.T
            scores.append(float(similarities.topk(5, dim=1).values.mean()))
    ranked = np.argsort(scores)[::-1]; best = ranked[0]
    diagnostics = dict(score=scores[best], proposals=len(scores), reason=None,bank_sha256=digest(bank_path))
    if prior_mask is not None:
        from .quality_detection_association import associate
        selected,association=associate(scores,proposals_masks>.5,prior_mask,prior_age)
        if selected is None:return None,dict(diagnostics,reason=association['reason'],association=association)
        return (proposals_masks[selected]>.5).astype(np.uint8)*255,dict(diagnostics,
            score=scores[selected],selected_proposal=int(selected),association=association)
    if scores[best] < .25:
        return None, dict(diagnostics, reason='cnos_low_object_identity_score')
    if len(ranked) > 1 and scores[ranked[1]] >= scores[best]*.95:
        a, b = proposals_masks[best] > .5, proposals_masks[ranked[1]] > .5
        if np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b)) < .5:
            return None, dict(diagnostics, reason='cnos_ambiguous_object_identity')
    return (proposals_masks[best] > .5).astype(np.uint8)*255, diagnostics
