"""Separate evaluator for exploratory axial proposals; estimated references only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from .quality_assets import CACHE,ROOT,save_result
from .quality_surface_agreement import sample_surface,agreement
from .glb_model import read_glb


def main():
    out=CACHE/'diagnostics/bottle-texture-detail';probe=json.loads((out/'probe.json').read_text())
    saved=json.loads((CACHE/'results/ranch/render-stability-appearance/complete.json').read_text())
    by_id={f['frameId']:f for f in saved['frames']};ids=[r['frame_id'] for r in probe['rows'] if r.get('best_degrees') is not None]
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='ranch')
    mesh=read_glb(CACHE/'inputs/ranch/object.glb','ranch');points,normals,_,_=sample_surface(mesh);cache={}
    control=dict(frames=[by_id[fid] for fid in ids]);candidate=dict(frames=[])
    for row in probe['rows']:
        if row['frame_id'] not in ids:continue
        proposal=next(c for c in row['candidates'] if c['degrees']==row['best_degrees'])
        candidate['frames'].append(dict(frameId=row['frame_id'],pose_state='tracking',cameraFromObject=proposal['camera_from_object']))
    # Temporary 'tracking' enables the numeric evaluator; these remain raw
    # unvalidated proposals and are never published as tracking records.
    before=agreement(control,original,mesh,points,normals,cache);after=agreement(candidate,original,mesh,points,normals,cache)
    report=dict(scope=__doc__,selection=probe['selection'],requested=len(probe['rows']),proposals=len(ids),
        control_secondary=before,proposal_secondary=after,new_accepted_tracking_poses=False,independent_accuracy_verified=False)
    save_result(out/'evaluation.json',report)
    fig,axes=plt.subplots(4,4,figsize=(10,9),layout='constrained')
    for i,fid in enumerate([10,209,225,230]):
        with np.load(out/f'{fid}.npz',allow_pickle=False) as arrays:
            mask=arrays['mask'];yy,xx=np.nonzero(mask);pad=12
            xmin=max(0,xx.min()-pad);xmax=min(mask.shape[1],xx.max()+pad)
            ymin=max(0,yy.min()-pad);ymax=min(mask.shape[0],yy.max()+pad)
            overlay=arrays['rgb'].copy();overlay[mask]=np.rint(.55*overlay[mask]+.45*np.array([40,230,160])).astype(np.uint8)
            row=next(r for r in probe['rows'] if r['frame_id']==fid)
            for j,(image,title) in enumerate([(arrays['rgb'],'Actual image'),(overlay,'Automatic mask'),
                (arrays['seed_render'],'Saved accepted pose render'),(arrays['proposal_render'],f'Unvalidated yaw {row["best_degrees"]}°')]):
                axes[i,j].imshow(image[ymin:ymax,xmin:xmax]);axes[i,j].axis('off')
                axes[i,j].set_title((f'Frame {fid}\n' if j==0 else '')+title,fontsize=9)
    fig.suptitle('Exploratory texture-detail hypotheses — model renders are not camera depth',fontsize=12)
    fig.savefig(out/'comparison.jpg',dpi=120,pil_kwargs={'quality':85});plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('control_secondary','proposal_secondary')},indent=2))
    print('Secondary seed / unvalidated proposal median, P95:',before['median_720'],before['p95_720'],after['median_720'],after['p95_720'])


if __name__=='__main__':main()
