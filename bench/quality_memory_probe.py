"""Replay private model-memory checks against saved automatic poses.

Diagnostic only: no GoTrack rerun, no replacement benchmark, no evaluation
annotations or dataset poses. In particular, rejection counts are not accuracy.
"""
import json
import numpy as np
from .quality_assets import CACHE, save_result
from .quality_runner import read_input
from .quality_contract import Frame, PoseCandidate, canonical_pose
from .quality_memory import ModelPointMemory
from .glb_model import read_glb
from .show3d_experiment import camera_for
from .renderer import render
from .vision import cv2


def main():
    cv2.setNumThreads(1)
    bundle = CACHE/'inputs/keyboard'; manifest = read_input(bundle)
    saved = json.loads((CACHE/'results/keyboard/complete.json').read_text())
    if [f['frameId'] for f in saved['frames']] != manifest['frame_ids']:
        raise ValueError('Expected all original keyboard frames')
    mesh = read_glb(bundle/manifest['asset'], 'keyboard')
    def depth(pose, k, width, height):
        return render(mesh, camera_for((pose[:3, :3], pose[:3, 3]), k, width, height), shade=False).depth
    memory = ModelPointMemory(depth)
    cap = cv2.VideoCapture(str(bundle/manifest['video']))
    cap.set(cv2.CAP_PROP_POS_FRAMES, manifest['frame_ids'][0])
    k = np.array(manifest['intrinsics']); rows = []
    path = CACHE/'results/keyboard/model-memory/probe.json'
    try:
        for index, record in enumerate(saved['frames']):
            ok, bgr = cap.read()
            if not ok: raise ValueError('Missing chronological source frame')
            frame = Frame(record['frameId'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), k)
            mask = cv2.imread(str(CACHE/'results/keyboard/segmentation'/record['mask_path']), cv2.IMREAD_GRAYSCALE)
            before_points = 0 if memory.points is None else len(memory.points)
            anchor = memory.anchor_id
            memory.observe(frame, mask if record['mask_state'] == 'available' else None)
            pose = record.get('cameraFromObject')
            entry = dict(frame_id=frame.frame_id, saved_pose_state=record['pose_state'],
                         motion_seed_available=memory.seed(frame.frame_id) is not None,
                         anchor_frame=anchor, points_before_observation=before_points,
                         points_after_observation=0 if memory.points is None else len(memory.points),
                         image_std=float(memory.gray.std()))
            if pose is not None:
                pose = canonical_pose(pose)
                candidate = PoseCandidate(pose, np.empty((0, 3)), np.empty((0, 2)), np.empty(0))
                valid, reason, stats = memory.validate(candidate, frame.frame_id)
                entry.update(motion_check=stats, would_reject=not valid, reason=reason)
                if valid: memory.commit(frame, mask, pose)
            rows.append(entry)
            save_result(path, dict(scope=__doc__, complete=len(rows) == 240, frames=rows,
                full_model_inference=False, independent_accuracy_verified=False))
            if index % 40 == 0: print('Model memory probe', index+1, '/', 240, flush=True)
    finally: cap.release()
    print('Motion evidence', sum(r['motion_seed_available'] for r in rows),
          'saved accepted poses contradicted', sum(r.get('would_reject', False) for r in rows), flush=True)
    print('Late flip check', next(r for r in rows if r['frame_id'] == 1383), flush=True)


if __name__ == '__main__': main()
