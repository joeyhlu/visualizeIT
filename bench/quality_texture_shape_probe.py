"""Current-image silhouette ranking of preserved axial proposals; diagnostic only."""
import json
import numpy as np
from .quality_assets import CACHE,save_result,digest
from .quality_gotrack import TexturedRenderer
from .quality_contract import canonical_pose
from .vision import cv2


def main():
    previous=CACHE/'diagnostics/bottle-texture-detail/probe.json';data=json.loads(previous.read_text())
    bundle=CACHE/'inputs/ranch';manifest=json.loads((bundle/'input.json').read_text())
    out=CACHE/'diagnostics/bottle-texture-shape';out.mkdir(parents=True,exist_ok=True)
    renderer=TexturedRenderer(bundle/'object.glb',manifest['object_id'],unlit=True,disable_multisampling=True)
    from utils import structs,renderer_base
    k=np.array(manifest['intrinsics']);k[:2]*=.5;rows=[]
    try:
        for item in data['rows']:
            fid=item['frame_id']
            if 'candidates' not in item:rows.append(item);continue
            mask=cv2.imread(str(CACHE/f'results/ranch/segmentation/masks/{fid}.png'),0)
            if mask is None:raise ValueError('Actual mask required')
            mask=cv2.resize(mask,(512,640),interpolation=cv2.INTER_NEAREST)>0
            scored=[]
            for candidate in item['candidates']:
                camera_pose=np.linalg.inv(canonical_pose(candidate['camera_from_object']));camera_pose[:3,3]*=1000
                camera=structs.PinholePlaneCameraModel(width=512,height=640,f=(k[0,0],k[1,1]),c=(k[0,2],k[1,2]),T_world_from_eye=camera_pose)
                output=renderer.render_object_model(manifest['object_id'],camera);silhouette=output[renderer_base.RenderType.DEPTH]>0
                iou=float((silhouette&mask).sum()/max(1,(silhouette|mask).sum()))
                scored.append({**candidate,'visible_mask_silhouette_iou':iou})
            ranked=sorted(scored,key=lambda c:c['visible_mask_silhouette_iou'],reverse=True)
            # An exploratory seed only. Hand occlusion can bias this ranking;
            # it supplies neither correspondences nor acceptance confidence.
            best=ranked[0]
            rows.append({**item,'candidates':scored,'best_degrees':best['degrees'],
                'runner_up_margin':best['visible_mask_silhouette_iou']-ranked[1]['visible_mask_silhouette_iou']})
            print(fid,'shape yaw',best['degrees'],'IoU',best['visible_mask_silhouette_iou'],flush=True)
    finally:renderer.close()
    save_result(out/'probe.json',{**data,'scope':__doc__,'rows':rows,'proposal_label':'shape_seed',
        'previous_probe_sha256':digest(previous),
        'caveat':'Visible-object silhouette IoU penalizes real occluders; this diagnostic is not a foreground prediction or pose validation.'})


if __name__=='__main__':main()
