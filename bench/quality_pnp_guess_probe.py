"""Paired five-iteration fitting diagnostic; no reference/annotation inputs."""
import argparse
import json
import os
import time
import random
from contextlib import nullcontext
import numpy as np
from .quality_assets import CACHE,ROOT,save_result,digest,inference_provenance
from .quality_contract import Frame,validate
from .quality_gotrack import TexturedRenderer,load_network,GoTrackRefiner
from .quality_runner import read_rgb
from .quality_pnp_guess import without_extrinsic_guess
from .vision import cv2


CASES={
    'ranch':('render-stability-appearance/complete.json',[10,50,100,150,185,209,217,225,230,249]),
    'keyboard':('render-stability/complete.json',[1149,1176,1202,1229,1256,1281,1306,1330,1355,1388]),
    'mug':('complete.json',[827,853,880,906,933,959,986,1012,1039,1066]),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--object',choices=CASES,required=True);args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);cv2.setNumThreads(1)
    alias=args.object;bundle=CACHE/'inputs'/alias;manifest=json.loads((bundle/'input.json').read_text())
    path,ids=CASES[alias];seed_path=CACHE/'results'/alias/path
    predicted={f['frameId']:f for f in json.loads(seed_path.read_text())['frames']}
    out=CACHE/'diagnostics/pnp-guess'/alias;out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'bench/quality_pnp_guess_probe.py').read_bytes();(out/'probe-source.py').write_bytes(source)
    network=load_network('cuda');renderer=TexturedRenderer(bundle/'object.glb',manifest['object_id'],unlit=True,disable_multisampling=True)
    refiner=GoTrackRefiner(network,renderer,manifest['object_id'],'cuda');output=[]
    try:
        for fid in ids:
            frame=Frame(fid,read_rgb(bundle,manifest,fid),np.asarray(manifest['intrinsics']));branches={}
            mask=cv2.imread(str(CACHE/'results'/alias/'segmentation'/predicted[fid]['mask_path']),0)
            if mask is None:raise ValueError('Actual mask required')
            for name,context in [('control',nullcontext),('no_guess',without_extrinsic_guess)]:
                seed=predicted[fid]['cameraFromObject']
                if seed is None:branches[name]=dict(current_image_validated=False,pose=None,reason='automatic_seed_unavailable');continue
                torch.cuda.synchronize();begin=time.perf_counter()
                with context():candidate=refiner.refine(frame,mask,np.asarray(seed))
                torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000
                if candidate is None:valid=False;reason='refinement_unavailable';stats=None
                else:valid,reason,stats=validate(candidate,frame,mask)
                branches[name]=dict(current_image_validated=valid,pose=None if candidate is None else candidate.pose.tolist(),
                    reason=reason,validation=stats,elapsed_ms=elapsed)
            output.append(dict(frame_id=fid,branches=branches));print(alias,fid,{k:v['current_image_validated'] for k,v in branches.items()},flush=True)
            save_result(out/'neural.json',dict(scope=__doc__,object=alias,frames=output,requested_frame_ids=ids,
                provenance=inference_provenance(bundle),seed_result_sha256=digest(seed_path),
                probe_code_sha256=digest(ROOT/'bench/quality_pnp_guess_probe.py'),
                fitting_patch_sha256=digest(ROOT/'bench/quality_pnp_guess.py'),
                settings=dict(unlit=True,disable_multisampling=True,iterations=5,thresholds_changed=False),
                selected_after_prior_benchmark=True,full_window_benchmark=False,sequential_recovery_test=False,
                reference_or_annotation_inputs=False,rendering_success_not_claimed=True))
    finally:renderer.close()


if __name__=='__main__':main()
