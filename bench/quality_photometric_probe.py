"""Image-only pose proposals on saved automatic tracking outputs.

No reference controls or annotations are loaded. Each frame is independent;
this is not a replacement automatic benchmark or a validated tracking loop.
"""
import argparse
import json
import time
import numpy as np
from .quality_assets import CACHE, digest, save_result
from .quality_gotrack import TexturedRenderer, upstream_path
from .quality_contract import canonical_pose
from .quality_photometric import align_samples, render_samples
from .vision import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, nargs='+', default=[10, 50, 100, 150, 193, 235])
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args(); upstream_path()
    from utils import structs, renderer_base
    cv2.setNumThreads(1)
    root = CACHE/'inputs/ranch'; source = CACHE/'results/ranch/appearance/complete.json'
    manifest = json.loads((root/'input.json').read_text()); saved = json.loads(source.read_text())
    chosen = set(manifest['frame_ids'] if args.all else args.frames)
    k = np.array(manifest['intrinsics']); k[:2] *= .5
    renderer = TexturedRenderer(root/manifest['asset'], manifest['object_id'], unlit=True)
    cap = cv2.VideoCapture(str(root/manifest['video'])); rows = []
    try:
        for record in saved['frames']:
            if record['frameId'] not in chosen: continue
            fid = record['frameId']; seed = record['cameraFromObject']
            entry = dict(frame_id=fid, original_pose_state=record['pose_state'], seed_pose=seed)
            if seed is None: entry['state']='pose_unavailable'; rows.append(entry); continue
            start = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid); ok, bgr = cap.read()
            if not ok: raise ValueError('Source frame missing')
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (512, 640), interpolation=cv2.INTER_AREA)
            mask = cv2.imread(str(CACHE/'results/ranch/segmentation'/record['mask_path']), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 640), interpolation=cv2.INTER_NEAREST) > 0
            camera_pose = np.linalg.inv(canonical_pose(seed)); camera_pose[:3, 3] *= 1000
            camera = structs.PinholePlaneCameraModel(width=512, height=640, f=(k[0,0],k[1,1]),
                c=(k[0,2],k[1,2]), T_world_from_eye=camera_pose)
            output = renderer.render_object_model(manifest['object_id'], camera)
            samples, intensity = render_samples(output[renderer_base.RenderType.COLOR],
                output[renderer_base.RenderType.DEPTH], mask, seed, k)
            pose, stats = align_samples(rgb, mask, k, seed, samples, intensity)
            entry.update(stats, proposed_pose=pose.tolist(), elapsed_seconds=time.perf_counter()-start)
            rows.append(entry); print(fid, stats['state'], round(entry['elapsed_seconds'], 3), flush=True)
    finally: cap.release(); renderer.close()
    name = 'photometric-all.json' if args.all else 'photometric-selected.json'
    save_result(CACHE/'diagnostics'/name, dict(scope=__doc__, source_sha256=digest(source),
        optimizer_sha256=digest(CACHE.parents[1]/'bench/quality_photometric.py'), frames=rows,
        independent_accuracy_verified=False, tracking_validated=False))


if __name__ == '__main__': main()
