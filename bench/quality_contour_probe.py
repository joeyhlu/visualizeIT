"""Image-only saved-pose contour diagnostic; no evaluation labels are opened here."""
import json
import time
from dataclasses import asdict
from .quality_assets import ROOT,CACHE,digest,save_result
from .quality_contour_pose import refine,ContourSettings
from .glb_model import read_glb
from .vision import cv2
import numpy as np


FRAME_IDS=[1149,1190,1231,1256,1281,1306,1314,1330,1355,1388]


def main():
    bundle=CACHE/'inputs/keyboard';manifest=json.loads((bundle/'input.json').read_text())
    seed_path=CACHE/'results/keyboard/render-stability/complete.json'
    seed=json.loads(seed_path.read_text());frames={f['frameId']:f for f in seed['frames']}
    mesh=read_glb(bundle/'object.glb','keyboard');settings=ContourSettings();output=[]
    for fid in FRAME_IDS:
        record=frames[fid];path=CACHE/'results/keyboard/segmentation'/record['mask_path']
        mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if mask is None:raise ValueError('Current inference mask missing')
        if record['pose_state']!='tracking':raise ValueError('Accepted automatic seed required for this proposal diagnostic')
        begin=time.perf_counter();pose,stats=refine(mesh.positions,np.asarray(manifest['intrinsics']),mask,record['cameraFromObject'],settings)
        output.append(dict(frame_id=fid,camera_from_object_proposal=None if pose is None else pose.tolist(),
            seed_camera_from_object=record['cameraFromObject'],diagnostics=stats,elapsed_ms=(time.perf_counter()-begin)*1000,
            native_mask_sha256=digest(path),seed_record_pose_state=record['pose_state'],accepted_tracking_pose=False))
    save_result(CACHE/'diagnostics/keyboard-contour-proposals.json',dict(scope=__doc__,
        selected_after_candidate_results=True,full_window_benchmark=False,settings=asdict(settings),
        asset_sha256=digest(bundle/'object.glb'),seed_results_sha256=digest(seed_path),
        algorithm_sha256=digest(ROOT/'bench/quality_contour_pose.py'),reference_or_annotations_loaded=False,frames=output))
    print('Contour proposals:',sum(f['camera_from_object_proposal'] is not None for f in output),'/',len(output))


if __name__=='__main__':main()
