"""Current-image-only appearance veto diagnostic on saved stable bottle poses."""
import json
import numpy as np
from dataclasses import asdict
from .quality_assets import CACHE,ROOT,digest,save_result
from .quality_gotrack import TexturedRenderer
from .quality_appearance import appearance_metrics,appearance_decision,AppearanceSettings
from .quality_contract import canonical_pose
from .vision import cv2


def main():
    bundle=CACHE/'inputs/ranch';manifest=json.loads((bundle/'input.json').read_text())
    prediction_path=CACHE/'results/ranch/render-stability/complete.json'
    predicted=json.loads(prediction_path.read_text());settings=AppearanceSettings()
    renderer=TexturedRenderer(bundle/'object.glb',manifest['object_id'],unlit=True,disable_multisampling=True)
    from utils import structs,renderer_base
    cap=cv2.VideoCapture(str(bundle/'source.mp4'));k=np.array(manifest['intrinsics']);k[:2]*=.5;rows=[]
    try:
        for record in predicted['frames']:
            fid=record['frameId'];entry=dict(frame_id=fid,pose_state=record['pose_state'])
            if record['cameraFromObject'] is None:entry.update(appearance={'state':'pose_unavailable'},vetoed=False)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,bgr=cap.read()
                if not ok:raise ValueError('Actual source frame required')
                rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(512,640),interpolation=cv2.INTER_AREA)
                mask=cv2.imread(str(CACHE/'results/ranch/segmentation'/record['mask_path']),0)
                mask=cv2.resize(mask,(512,640),interpolation=cv2.INTER_NEAREST)>0
                camera_pose=np.linalg.inv(canonical_pose(record['cameraFromObject']));camera_pose[:3,3]*=1000
                camera=structs.PinholePlaneCameraModel(width=512,height=640,f=(k[0,0],k[1,1]),c=(k[0,2],k[1,2]),T_world_from_eye=camera_pose)
                rendered=renderer.render_object_model(manifest['object_id'],camera)
                visible=mask&(rendered[renderer_base.RenderType.DEPTH]>0)
                appearance=appearance_metrics(rgb,rendered[renderer_base.RenderType.COLOR],visible)
                appearance['visible_overlap']=float(visible.sum()/max(1,mask.sum()))
                valid,reason=appearance_decision(appearance,settings)
                entry.update(appearance=appearance,vetoed=not valid,reason=reason)
            rows.append(entry)
    finally:cap.release();renderer.close()
    save_result(CACHE/'diagnostics/bottle-stable-identity.json',dict(scope=__doc__,settings=asdict(settings),rows=rows,
        prediction_sha256=digest(prediction_path),asset_sha256=digest(bundle/'object.glb'),
        reference_or_annotations_loaded=False,new_tracking_predictions=False))
    print('Actual poses inspected',len(rows),'vetoes',sum(r['vetoed'] for r in rows))


if __name__=='__main__':main()
