"""Image-only CNOS proposal/descriptor diagnostic, without evaluation labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from .quality_assets import CACHE,digest,save_result
from .quality_gotrack import upstream_path
from .quality_cnos import load_descriptor,camera_for,build_descriptors
from .quality_runner import read_rgb


def extract(bundle,frame,output):
    upstream_path()
    import torch
    from model.config import FastSAMOpts
    from utils import fastsam_util,crop_generation,im_util
    manifest=json.loads((bundle/'input.json').read_text());rgb=read_rgb(bundle,manifest,frame)
    checkpoint=CACHE/'checkpoints/FastSAM-x.pt';lock=json.loads((CACHE/'lock.json').read_text())
    if digest(checkpoint)!=lock['models'][checkpoint.name]['sha256']:raise ValueError('FastSAM checkpoint changed')
    segmentor=fastsam_util.FastSAM(FastSAMOpts(model_path=str(checkpoint),conf_threshold=.25,iou_threshold=.9,
        max_det=200,im_width_size=640,verbose=False));segmentor.set_device(torch.device('cuda'))
    proposals=segmentor.generate_detection_proposals(rgb[...,::-1].copy())
    keep=im_util.filter_noisy_detections(proposals.boxes,proposals.masks,min_box_size=.05,
        min_mask_size=3e-4,width=rgb.shape[1],height=rgb.shape[0]);proposals=proposals[keep]
    boxes=proposals.boxes.cpu().numpy();masks=proposals.masks.cpu().numpy()>.5
    crops=crop_generation.batch_cropping_from_bbox(source_images=[rgb]*len(proposals),
        source_cameras=[camera_for(manifest)]*len(proposals),source_xyxy_bboxes=boxes,
        source_masks=proposals.masks[...,None].cpu().numpy(),crop_size=(280,280),crop_rel_pad=.05)
    del segmentor,proposals;torch.cuda.empty_cache()
    model=load_descriptor('cuda');descriptors=[]
    with torch.inference_mode():
        for image,mask in zip(crops.rgbs,crops.masks):
            q=model.forward_features(im_util.im_normalize((image*mask)[None]).to('cuda'))['x_norm_clstoken']
            descriptors.append(q[0].cpu().numpy())
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,descriptors=np.stack(descriptors),boxes=boxes,masks=masks,frame_id=frame,
        asset_sha256=digest(bundle/manifest['asset']))
    print('Extracted',len(boxes),'current-image candidates',flush=True)


def rank(query_path,bank_path,output):
    with np.load(query_path,allow_pickle=False) as z:
        q=z['descriptors'].astype(np.float64);boxes=z['boxes'];masks=z['masks'];asset=str(z['asset_sha256']);fid=int(z['frame_id'])
    with np.load(bank_path,allow_pickle=False) as z:
        if str(z['asset_sha256'])!=asset:raise ValueError('Object assets disagree')
        references=z['descriptors'].astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True);references/=np.linalg.norm(references,axis=1,keepdims=True)
    similarities=q@references.T;scores=np.sort(similarities,axis=1)[:,-5:].mean(1);order=np.argsort(scores)[::-1]
    rows=[]
    for i in order:
        yy,xx=np.nonzero(masks[i]);rows.append(dict(proposal_id=int(i),identity_score=float(scores[i]),
            box=boxes[i].tolist(),mask_pixels=int(masks[i].sum()),centroid=[float(xx.mean()),float(yy.mean())]))
    report=dict(scope=__doc__,frame_id=fid,bank_sha256=digest(bank_path),query_sha256=digest(query_path),ranked=rows,
        independent_identity_verified=False)
    save_result(output,report);print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['extract','bank','rank']);parser.add_argument('--bundle',type=Path,default=CACHE/'inputs/keyboard')
    parser.add_argument('--frame',type=int,default=1254);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query',type=Path);parser.add_argument('--bank',type=Path)
    parser.add_argument('--unlit',action='store_true');parser.add_argument('--grayscale',action='store_true')
    args=parser.parse_args()
    if args.stage=='rank':rank(args.query,args.bank,args.output);return
    import torch,os,random
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.stage=='extract':extract(args.bundle,args.frame,args.output)
    else:print(build_descriptors(args.bundle,'cuda',args.unlit,args.grayscale,args.output),flush=True)


if __name__=='__main__':main()
