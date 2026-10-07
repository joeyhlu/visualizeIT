"""Selected current-image axial hypotheses; no reference poses or manual masks loaded."""
import json
import numpy as np
from .quality_assets import CACHE,digest,save_result
from .quality_gotrack import TexturedRenderer
from .quality_contract import canonical_pose
from .quality_appearance import appearance_metrics
from .quality_texture_detail import detail_metrics,axial_pose
from .vision import cv2


def main():
    bundle=CACHE/'inputs/ranch';manifest=json.loads((bundle/'input.json').read_text())
    predicted_path=CACHE/'results/ranch/render-stability-appearance/complete.json'
    predicted={f['frameId']:f for f in json.loads(predicted_path.read_text())['frames']}
    out=CACHE/'diagnostics/bottle-texture-detail';out.mkdir(parents=True,exist_ok=True)
    renderer=TexturedRenderer(bundle/'object.glb',manifest['object_id'],unlit=True,disable_multisampling=True)
    vertices=renderer.vertices_m;center=(vertices.min(0)+vertices.max(0))/2
    _,_,vt=np.linalg.svd(vertices-vertices.mean(0),full_matrices=False);axis=vt[0]
    axis*=1 if axis[np.argmax(np.abs(axis))]>=0 else -1
    from utils import structs,renderer_base
    cap=cv2.VideoCapture(str(bundle/'source.mp4'));k=np.array(manifest['intrinsics']);k[:2]*=.5
    rows=[];cv2.setNumThreads(1)
    try:
        for fid in [10,50,100,150,185,209,217,225,230,249]:
            record=predicted[fid];seed=record['cameraFromObject']
            if seed is None:rows.append(dict(frame_id=fid,state='pose_unavailable'));continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,bgr=cap.read()
            if not ok:raise ValueError('Actual source frame missing')
            rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(512,640),interpolation=cv2.INTER_AREA)
            mask=cv2.imread(str(CACHE/'results/ranch/segmentation'/record['mask_path']),0)
            if mask is None:raise ValueError('Actual cached mask missing')
            mask=cv2.resize(mask,(512,640),interpolation=cv2.INTER_NEAREST)>0
            candidates=[];images={}
            for degrees in range(-180,180,30):
                pose=axial_pose(seed,axis,center,degrees)
                camera_pose=np.linalg.inv(canonical_pose(pose));camera_pose[:3,3]*=1000
                camera=structs.PinholePlaneCameraModel(width=512,height=640,f=(k[0,0],k[1,1]),c=(k[0,2],k[1,2]),T_world_from_eye=camera_pose)
                rendered=renderer.render_object_model(manifest['object_id'],camera)
                color=rendered[renderer_base.RenderType.COLOR];visible=mask&(rendered[renderer_base.RenderType.DEPTH]>0)
                details=detail_metrics(rgb,color,visible);broad=appearance_metrics(rgb,color,visible)
                overlap=float(visible.sum()/max(1,mask.sum()))
                candidates.append(dict(degrees=degrees,camera_from_object=pose.tolist(),detail=details,broad=broad,visible_overlap=overlap))
                images[degrees]=np.rint(color*255).astype(np.uint8)
            supported=[c for c in candidates if c['visible_overlap']>=.7 and c['detail'].get('detail_correlation') is not None]
            supported.sort(key=lambda c:c['detail']['detail_correlation'],reverse=True)
            winner=supported[0] if supported else None
            rows.append(dict(frame_id=fid,state='diagnostic_only',candidates=candidates,
                best_degrees=None if winner is None else winner['degrees'],
                runner_up_margin=None if len(supported)<2 else winner['detail']['detail_correlation']-supported[1]['detail']['detail_correlation']))
            np.savez_compressed(out/f'{fid}.npz',rgb=rgb,mask=mask,seed_render=images[0],
                proposal_render=images[0 if winner is None else winner['degrees']])
            print(fid,'best yaw',None if winner is None else winner['degrees'],'seed detail',next(c['detail'] for c in candidates if c['degrees']==0),flush=True)
    finally:cap.release();renderer.close()
    save_result(out/'probe.json',dict(scope=__doc__,rows=rows,axis_object=axis.tolist(),center_object_m=center.tolist(),
        prediction_sha256=digest(predicted_path),asset_sha256=digest(bundle/'object.glb'),
        selection='Ten exploratory original-window frames; selected after earlier candidate review, not a benchmark',
        reference_or_annotations_loaded=False,new_accepted_tracking_poses=False))


if __name__=='__main__':main()
