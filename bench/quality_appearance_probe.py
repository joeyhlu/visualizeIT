"""Evaluator-only textured-render appearance diagnostic on selected frames.

Compares saved predictions with estimated reference controls, not human truth.
Photometric mismatch from lighting/texture capture can invalidate both; these
scores must not be promoted to tracking thresholds without broader validation.
"""
import json
import argparse
import numpy as np
from .quality_assets import CACHE, save_result
from .quality_gotrack import TexturedRenderer, upstream_path
from .quality_contract import canonical_pose
from .vision import cv2
from .quality_appearance import appearance_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-bottle', action='store_true')
    parser.add_argument('--unlit-render', action='store_true')
    args = parser.parse_args()
    upstream_path()
    from utils import structs, renderer_base
    cv2.setNumThreads(1); rows = []
    selections = [('ranch', list(range(10, 250)))] if args.full_bottle else [('ranch', [10, 50, 100, 150, 193, 235]), ('keyboard', [1149, 1229, 1309, 1383])]
    for alias, ids in selections:
        root = CACHE/'inputs'/alias; manifest = json.loads((root/'input.json').read_text())
        predicted = {f['frameId']: f for f in json.loads((CACHE/'results'/alias/'complete.json').read_text())['frames']}
        # Evaluation labels are intentionally confined to this diagnostic.
        from .quality_assets import ROOT
        controls = {f['frameId']: f['reference'] for f in json.loads((ROOT/'artifacts/video-60'/alias/'player.json').read_text())['frames']}
        renderer = TexturedRenderer(root/manifest['asset'], 1, unlit=args.unlit_render)
        cap = cv2.VideoCapture(str(root/manifest['video'])); k = np.array(manifest['intrinsics']); k[:2] *= .5
        try:
            for fid in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid); ok, bgr = cap.read()
                if not ok: raise ValueError('Missing source frame')
                rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (512, 640), interpolation=cv2.INTER_AREA)
                record = predicted[fid]
                mask = cv2.imread(str(CACHE/'results'/alias/'segmentation'/record['mask_path']), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (512, 640), interpolation=cv2.INTER_NEAREST) > 0
                entry = dict(object=alias, frame_id=fid, saved_pose_state=record['pose_state'])
                for mode, pose in (('prediction', record['cameraFromObject']), ('estimated_reference_control', controls[fid])):
                    if pose is None: entry[mode] = dict(state='pose_unavailable'); continue
                    camera_pose = np.linalg.inv(canonical_pose(pose)); camera_pose[:3, 3] *= 1000
                    camera = structs.PinholePlaneCameraModel(width=512, height=640,
                        f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]), T_world_from_eye=camera_pose)
                    output = renderer.render_object_model(1, camera)
                    entry[mode] = appearance_metrics(rgb, output[renderer_base.RenderType.COLOR],
                        mask & (output[renderer_base.RenderType.DEPTH] > 0))
                rows.append(entry); print(entry, flush=True)
        finally: cap.release(); renderer.close()
    name = 'appearance-bottle-full.json' if args.full_bottle else 'appearance-probe.json'
    if args.unlit_render: name = name.replace('.json', '-unlit.json')
    save_result(CACHE/'diagnostics'/name, dict(scope=__doc__, rows=rows, unlit_render=args.unlit_render, independent_accuracy_verified=False))


if __name__ == '__main__': main()
