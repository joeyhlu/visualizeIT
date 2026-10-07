"""Paired diagnostic: five GoTrack iterations with/without an image-only contour seed."""
import json
import time
import os
import random
import numpy as np
from .quality_assets import CACHE,ROOT,save_result,digest,inference_provenance
from .quality_contract import Frame,validate
from .quality_gotrack import TexturedRenderer,load_network,GoTrackRefiner
from .quality_runner import read_rgb
from .vision import cv2


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);cv2.setNumThreads(1)
    bundle=CACHE/'inputs/keyboard';manifest=json.loads((bundle/'input.json').read_text())
    proposal_path=CACHE/'diagnostics/keyboard-contour-proposals.json'
    proposals=json.loads(proposal_path.read_text());network=load_network('cuda')
    renderer=TexturedRenderer(bundle/'object.glb',28,unlit=True,disable_multisampling=True)
    refiner=GoTrackRefiner(network,renderer,28,'cuda');output=[]
    try:
        for item in proposals['frames']:
            fid=item['frame_id'];branches={}
            rgb=read_rgb(bundle,manifest,fid);frame=Frame(fid,rgb,np.asarray(manifest['intrinsics']))
            mask=cv2.imread(str(CACHE/f'results/keyboard/segmentation/masks/{fid}.png'),cv2.IMREAD_GRAYSCALE)
            for name,key in [('control','seed_camera_from_object'),('contour_seed','camera_from_object_proposal')]:
                seed=item[key]
                if seed is None:branches[name]=dict(current_image_validated=False,pose=None,reason='contour_proposal_unavailable');continue
                torch.cuda.synchronize();begin=time.perf_counter();candidate=refiner.refine(frame,mask,np.asarray(seed))
                torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
                if candidate is None:valid=False;reason='refinement_unavailable';stats=None
                else:valid,reason,stats=validate(candidate,frame,mask)
                branches[name]=dict(current_image_validated=valid,pose=None if candidate is None else candidate.pose.tolist(),
                    reason=reason,validation=stats,elapsed_ms=elapsed)
            output.append(dict(frame_id=fid,branches=branches));print(fid,{k:v['current_image_validated'] for k,v in branches.items()},flush=True)
            save_result(CACHE/'diagnostics/keyboard-contour-neural.json',dict(scope=__doc__,frames=output,
                provenance=inference_provenance(bundle),proposal_sha256=digest(proposal_path),
                probe_code_sha256=digest(ROOT/'bench/quality_contour_neural_probe.py'),
                selected_after_candidate_results=True,full_window_benchmark=False,sequential_recovery_test=False,
                reference_or_annotation_inputs=False,rendering_success_not_claimed=True))
    finally:renderer.close()


if __name__=='__main__':main()
