"""Supplemental area-sampled estimated-pose diagnostic; never an independent accuracy gate."""
import json
import numpy as np
from .quality_assets import CACHE, ROOT, save_result
from .quality_contract import project, checked_pose
from .glb_model import read_glb
from .hd_experiment import unoccluded
from .storage import write_artifact


def sample_surface(mesh, count=128, seed=0):
    if count < 1: raise ValueError('Positive surface sample count required')
    vertices=mesh.positions[mesh.triangles]
    cross=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
    areas=np.linalg.norm(cross,axis=1)/2
    if not np.isfinite(areas).all() or areas.sum()<=0: raise ValueError('Nondegenerate surface required')
    # Stratified area quantiles avoid privileging densely tessellated faces.
    faces=np.searchsorted(np.cumsum(areas), (np.arange(count)+.5)/count*areas.sum(),side='right')
    rng=np.random.default_rng(seed); u,v=rng.random((2,count)); root=np.sqrt(u)
    barycentric=np.column_stack([1-root,root*(1-v),root*v])
    points=np.einsum('ni,nij->nj',barycentric,vertices[faces])
    normals=np.einsum('ni,nij->nj',barycentric,mesh.normals[mesh.triangles[faces]])
    lengths=np.linalg.norm(normals,axis=1); bad=lengths<1e-12
    normals[bad]=cross[faces[bad]]; lengths=np.linalg.norm(normals,axis=1)
    normals/=lengths[:,None]
    return points,normals,faces,barycentric


def visibility(mesh,points,normals,pose,k,resolution):
    expected,z=project(points,pose,k); width,height=resolution
    camera_points=points@pose[:3,:3].T+pose[:3,3]
    visible=(z>.03)&(expected[:,0]>=0)&(expected[:,0]<width)&(expected[:,1]>=0)&(expected[:,1]<height)
    visible &= np.sum((normals@pose[:3,:3].T)*camera_points,axis=1)<0
    candidates=np.flatnonzero(visible)
    if len(candidates):
        visible[candidates] &= unoccluded(points[candidates],mesh.positions,mesh.triangles,-pose[:3,:3].T@pose[:3,3])
    return visible,expected


def agreement(results,original,mesh,points,normals,visibility_cache):
    refs={f['frameId']:f for f in original['referenceFrames']}
    k=np.asarray(original['cameraCalibration']).reshape(3,3)
    errors=[]; details=[]; visible_samples=0; reference_frames=0
    for f in results['frames']:
        reference=refs[f['frameId']]['cameraFromObject']
        if reference is None:continue
        reference_frames+=1
        key=f['frameId']
        if key not in visibility_cache:
            visibility_cache[key]=visibility(mesh,points,normals,checked_pose(reference),k,original['nativeResolution'])
        visible,expected=visibility_cache[key]; visible_samples+=int(visible.sum())
        accepted=f['pose_state']=='tracking' and f.get('cameraFromObject') is not None
        values=[]
        if accepted:
            predicted,z=project(points,checked_pose(f['cameraFromObject']),k)
            # A behind-camera prediction remains an error, not a dropped sample.
            values=(np.linalg.norm(predicted[visible]-expected[visible],axis=1)*720/original['nativeResolution'][1]).tolist()
            errors.extend(values)
        details.append(dict(frame_id=f['frameId'],accepted=accepted,reference_visible_samples=int(visible.sum()),
            accepted_samples=len(values),median_720=float(np.median(values)) if values else None,
            p95_720=float(np.percentile(values,95)) if values else None))
    return dict(processed=len(results['frames']),accepted=sum(f['pose_state']=='tracking' for f in results['frames']),
        reference_frames=reference_frames,reference_visible_samples=visible_samples,accepted_point_samples=len(errors),
        sample_availability=len(errors)/visible_samples if visible_samples else None,
        median_720=float(np.median(errors)) if errors else None,p95_720=float(np.percentile(errors,95)) if errors else None,
        details=details)


def main():
    mesh=read_glb(CACHE/'inputs/keyboard/object.glb','keyboard')
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='keyboard')
    points,normals,faces,barycentric=sample_surface(mesh)
    visibility_cache={}; runs={}
    for name,path in [('previous','complete.json'),('single_sample','render-stability/complete.json'),
                      ('interruption_association','recovery-association-1239/complete.json')]:
        result=json.loads((CACHE/'results/keyboard'/path).read_text())
        if len(result['frames'])!=240: raise ValueError('Preserve full original window')
        runs[name]=agreement(result,original,mesh,points,normals,visibility_cache)
    report=dict(scope=__doc__,sample_count=128,seed=0,sampling='stratified triangle area and uniform barycentric samples',
        visibility='Model self-visibility only; excludes neither physical hands nor model reconstruction errors',
        independent_accuracy_verified=False,legacy_vertex_metrics_unchanged=True,runs=runs)
    save_result(CACHE/'diagnostics/keyboard-area-agreement.json',report)
    # Keep per-frame diagnostics in the local cache; publish only the small summary.
    summary={**report,'runs':{name:{k:v for k,v in run.items() if k!='details'} for name,run in runs.items()}}
    write_artifact(ROOT/'artifacts/model-quality/surface-agreement-results.json',json.dumps(summary,indent=2))
    print(json.dumps({name:{k:v for k,v in r.items() if k!='details'} for name,r in runs.items()},indent=2),flush=True)


if __name__=='__main__':main()
