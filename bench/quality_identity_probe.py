"""Chronological surface-identity diagnostic over saved automatic model poses.

This does not rerun or replace GoTrack. Rejections identify contradictions,
not proven tracking errors. No dataset poses or annotations are read.
"""
import argparse
import json
import numpy as np
from .quality_assets import CACHE, save_result
from .quality_runner import read_input
from .quality_contract import Frame, PoseCandidate, canonical_pose
from .quality_identity_memory import SurfaceIdentityMemory
from .glb_model import read_glb
from .show3d_experiment import camera_for
from .renderer import render
from .vision import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['keyboard', 'mug', 'ranch'], required=True)
    a = parser.parse_args(); cv2.setNumThreads(1)
    bundle = CACHE/'inputs'/a.object; manifest = read_input(bundle)
    saved = json.loads((CACHE/'results'/a.object/'complete.json').read_text())
    if [f['frameId'] for f in saved['frames']] != manifest['frame_ids']: raise ValueError('Original full window required')
    mesh = read_glb(bundle/manifest['asset'], a.object)
    def depth(pose, k, width, height):
        return render(mesh, camera_for((pose[:3, :3], pose[:3, 3]), k, width, height), shade=False).depth
    memory = SurfaceIdentityMemory(depth)
    cap = cv2.VideoCapture(str(bundle/manifest['video'])); cap.set(cv2.CAP_PROP_POS_FRAMES, manifest['frame_ids'][0])
    k = np.array(manifest['intrinsics']); rows = []
    path = CACHE/'results'/a.object/'model-memory/identity-probe.json'
    try:
        for index, record in enumerate(saved['frames']):
            ok, bgr = cap.read()
            if not ok: raise ValueError('Missing source image')
            frame = Frame(record['frameId'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), k)
            mask = cv2.imread(str(CACHE/'results'/a.object/'segmentation'/record['mask_path']), cv2.IMREAD_GRAYSCALE)
            if record['mask_state'] != 'available': mask = None
            pose = record.get('cameraFromObject')
            # Check chronology even while no display pose was returned. The
            # placeholder is never stored as a model reference or rendered.
            candidate_pose = canonical_pose(pose) if pose is not None else np.eye(4)
            candidate = PoseCandidate(candidate_pose, np.empty((0, 3)), np.empty((0, 2)), np.empty(0))
            valid, reason, stats = memory.check(frame, mask, candidate)
            entry = dict(frame_id=frame.frame_id, saved_pose_state=record['pose_state'], identity_check=stats,
                would_reject=(not valid) if pose is not None else None, reason=reason if pose is not None else None)
            if pose is not None and valid and (not memory.views or stats['state'] == 'available'):
                entry['keyframe_added'] = memory.commit(frame, mask, candidate_pose)
            rows.append(entry)
            save_result(path, dict(scope=__doc__, complete=len(rows) == 240, frames=rows,
                full_model_inference=False, independent_accuracy_verified=False))
            if index % 40 == 0: print(a.object, 'identity probe', index+1, '/', 240, flush=True)
    finally: cap.release()
    print('Identity evidence', sum(r['identity_check']['state'] == 'available' for r in rows),
          'saved pose contradictions', sum(bool(r['would_reject']) for r in rows), flush=True)
    target = 1383 if a.object == 'keyboard' else 235 if a.object == 'ranch' else manifest['frame_ids'][-1]
    print('Diagnostic target', next(r for r in rows if r['frame_id'] == target), flush=True)


if __name__ == '__main__': main()
