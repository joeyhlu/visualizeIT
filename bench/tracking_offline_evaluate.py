"""Read-only, post-seal evaluator for native recorded SAM -> metric pose runs.

Never runs SAM, a pose network, a renderer, or admission. The caller must supply
the independently communicated prediction-seal digest. All sealed run files and
native RGB/mask identities are checked BEFORE opening the separate evaluator.
Results are dataset-reference proxies, not independent human attachment labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from . import workbench_contract as wc
from .workbench_geometry import array_sha256, mask_sha256


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def load(path):
    return wc.loads_strict(Path(path).read_text())


def contained(root, relative):
    root=Path(root).resolve(); rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Resource escapes its root')
    target=root/rel
    for node in [target,*list(target.parents)[:len(rel.parts)-1]]:
        if node.is_symlink():raise ValueError('Symlink resource is not allowed')
    if not target.resolve().is_relative_to(root):raise ValueError('Resource escapes its root')
    return target


def checked(path, expected, size=None):
    path=Path(path)
    if not path.is_file() or digest(path)!=expected or (size is not None and path.stat().st_size!=size):
        raise ValueError('Resource identity differs: '+str(path))
    return path


def reference_protocol(seal,allow_reused_reference_development=False):
    """Distinguish old run-local isolation from deliberately reused development GT."""
    schema=seal.get('schema')
    if seal.get('evaluated') is not False:raise ValueError('Expected an unevaluated recorded prediction seal')
    if schema=='offline_recorded_prediction_seal_v1':
        if seal.get('scoredGroundTruthOpenedBeforeSeal') is not False:raise ValueError('No pre-seal ground-truth isolation attestation')
        if any(key in seal for key in ('prior_reference_access','inference_process_scored_GT_access','evaluationProtocol')):
            raise ValueError('Mixed historical v1 and expanded-development attestations are ambiguous')
        return dict(mode='historical_v1_run_local_isolation',
            note='Historical v1 field is treated as a prediction-production/run-local attestation, NOT proof of historically unopened dataset references. HB references were prepared/accessed earlier and reused in geometric benchmarks.')
    if schema!='offline_recorded_prediction_seal_v2':raise ValueError('Unknown recorded prediction seal schema')
    if not allow_reused_reference_development:raise ValueError('Expanded development/reused-reference evaluation requires explicit opt-in')
    required={'evaluationProtocol':'expanded_development_reused_reference','prior_reference_access':True,
        'inference_process_scored_GT_access':False,'scored_GT_feedback_to_inference':False,'weights_math_thresholds_pinned':True,
        'comparative_scoring_requires_all_role_seals':True}
    if 'scoredGroundTruthOpenedBeforeSeal' in seal:raise ValueError('Expanded development must not claim globally unopened scored GT')
    for key,expected in required.items():
        if (seal.get(key) is not expected if isinstance(expected,bool) else seal.get(key)!=expected):
            raise ValueError('Contradictory/missing expanded-development attestation: '+key)
    if not isinstance(seal.get('expectedComparativeRoles'),list) or set(seal['expectedComparativeRoles'])!={'primary_neural','frozen_baseline'} or len(seal['expectedComparativeRoles'])!=2:
        raise ValueError('Expanded comparison must declare primary and frozen baseline roles')
    return dict(mode='expanded_development_reused_reference',prior_reference_access=True,inference_process_scored_GT_access=False,
        scored_GT_feedback_to_inference=False,weights_math_thresholds_pinned=True,
        note='References were already accessed by earlier dataset/geometric/pilot work. This is expanded development on reused references, not an independent held-out test. Comparative scoring is allowed only after all declared role prediction seals are verified.')


def verify_seal(run, seal_path, seal_sha256,allow_reused_reference_development=False):
    """No evaluator access: reject incomplete, modified, or unanchored runs."""
    seal=load(checked(seal_path,seal_sha256))
    reference_protocol(seal,allow_reused_reference_development)
    if not {'results.jsonl','run.json','report.json'}<=set(seal['files']):raise ValueError('Incomplete sealed run')
    for name, spec in seal['files'].items():checked(contained(run,name),spec['sha256'],spec['bytes'])
    # Preserve native comparator authority: frozen_baseline cannot replace the
    # primary model. A single combined native run/seal may cover BOTH roles.
    native=wc.RunSpec.from_dict(load(Path(run)/'run.json')['run']).to_dict()
    if native['run_id']!=seal['run_id'] or native['scheduled_frame_ids']!=seal['scoredFrameIDs'] or native['roles']!=seal['roles']:
        raise ValueError('Seal/native schedule or roles differ')
    if not seal['roles'] or not seal['scoredFrameIDs'] or len(set(seal['roles']))!=len(seal['roles']) or len(set(seal['scoredFrameIDs']))!=len(seal['scoredFrameIDs']):
        raise ValueError('Duplicate sealed role/frame')
    rows=[wc.FrameResult.from_dict(wc.loads_strict(line)).to_dict() for line in (Path(run)/'results.jsonl').read_text().splitlines() if line]
    if len(rows)!=seal['rows']:raise ValueError('Sealed result row count differs')
    seen=set()
    for row in rows:
        key=row['backend_role'],row['frame_id']
        if key in seen or row['backend_role'] not in seal['roles'] or row['frame_id'] not in seal['scoredFrameIDs']:
            raise ValueError('Duplicate/unexpected prediction row')
        if row['run_id']!=seal['run_id'] or row['clip_id']!=native['clip_id']:raise ValueError('Prediction run/clip differs')
        seen.add(key)
    return seal,native,rows


def verify_comparison_seals(primary, companions):
    """All cohort bindings must be sealed before any reused-reference scoring."""
    all_seals=[primary,*companions]
    if primary['schema']=='offline_recorded_prediction_seal_v1':
        if companions:raise ValueError('Historical pilot v1 must not be mixed with expanded comparison seals')
        return
    roles=[]
    for seal in all_seals:
        if seal.get('schema')!='offline_recorded_prediction_seal_v2' or seal.get('evaluationProtocol')!=primary['evaluationProtocol']:
            raise ValueError('Mixed comparison seal protocols')
        if seal['scoredFrameIDs']!=primary['scoredFrameIDs'] or seal['nativeMaskManifestSHA256']!=primary['nativeMaskManifestSHA256'] or seal['expectedComparativeRoles']!=primary['expectedComparativeRoles']:
            raise ValueError('Comparison source schedule, observed masks, or declared cohort differs')
        roles.extend(seal['roles'])
    if len(roles)!=len(set(roles)) or set(roles)!=set(primary['expectedComparativeRoles']):
        raise ValueError('Comparative scoring requires BOTH unique declared role prediction seals')


def rotation_residuals(pose):
    r=np.asarray(pose,dtype=np.float64)[:3,:3]
    return dict(max_abs_RtR_minus_I=float(np.max(np.abs(r.T@r-np.eye(3)))),abs_detR_minus_1=float(abs(np.linalg.det(r)-1)))


def validate_pose(pose, rotation_tolerance=1e-6):
    p=np.asarray(pose,dtype=np.float64)
    if p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1],rtol=0,atol=1e-8):
        raise ValueError('Pose is not finite homogeneous SE3')
    residual=rotation_residuals(p)
    if residual['max_abs_RtR_minus_I']>rotation_tolerance or residual['abs_detR_minus_1']>rotation_tolerance:
        raise ValueError('Pose rotation is not proper orthonormal')
    return p


def model_vertices(path, count=512):
    """Read embedded identity-node source-axis GLB POSITION only; no rendering."""
    if not 1<=count<=512:raise ValueError('Geometry sample must be between1 and512')
    raw=Path(path).read_bytes()
    if len(raw)<20 or struct.unpack_from('<4sII',raw)!=(b'glTF',2,len(raw)):raise ValueError('Invalid GLB')
    offset=12; document=binary=None
    while offset<len(raw):
        size,kind=struct.unpack_from('<I4s',raw,offset);offset+=8;chunk=raw[offset:offset+size];offset+=size
        if len(chunk)!=size:raise ValueError('Truncated GLB')
        if kind==b'JSON':document=json.loads(chunk)
        elif kind==b'BIN\0':binary=chunk
    if document is None or binary is None:raise ValueError('Embedded GLB required')
    if any('uri' in buf for buf in document.get('buffers',[])):raise ValueError('External GLB buffers are not supported')
    if len(document.get('nodes',[]))!=1 or document['nodes'][0]!={'mesh':0}:raise ValueError('Identity node required')
    if len(document['meshes'])!=1 or len(document['meshes'][0]['primitives'])!=1:raise ValueError('Single source mesh required')
    accessor=document['accessors'][document['meshes'][0]['primitives'][0]['attributes']['POSITION']]
    if 'sparse' in accessor:raise ValueError('Sparse GLB positions are not supported')
    view=document['bufferViews'][accessor['bufferView']]
    if accessor['componentType']!=5126 or accessor['type']!='VEC3' or view.get('byteStride',12)!=12 or view.get('buffer',0)!=0:
        raise ValueError('Unsupported source vertex encoding')
    n=accessor['count'];start=view.get('byteOffset',0)+accessor.get('byteOffset',0)
    if not 1<=n<=1_000_000 or start<0 or start+12*n>len(binary) or accessor.get('byteOffset',0)+12*n>view['byteLength']:raise ValueError('Invalid source vertex bounds')
    vertices=np.frombuffer(binary,dtype='<f4',count=n*3,offset=start).reshape(n,3)
    if not np.isfinite(vertices).all():raise ValueError('Non-finite model vertices')
    indices=np.linspace(0,n-1,min(n,count),dtype=np.int64)
    return vertices[indices].astype(np.float64),dict(policy='evenly spaced original GLB vertex indices, including endpoints; not surface-uniform',
        fullVertexCount=n,sampleCount=len(indices),indices=indices.tolist(),indicesSHA256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())


def pose_errors(predicted, reference, vertices, K):
    a=validate_pose(predicted)
    # BOP reference matrices are serialized approximate rotations. Retain the
    # exact reference numbers, accepted by the EXISTING native contract (1e-4),
    # rather than misclassifying reference rounding as an invalid prediction.
    # Predicted outputs still face the stricter independent 1e-6 audit above.
    wc._pose(np.asarray(reference).tolist(),'dataset reference pose')
    b=validate_pose(reference,rotation_tolerance=1e-4)
    pa=vertices@a[:3,:3].T+a[:3,3];pb=vertices@b[:3,:3].T+b[:3,3]
    if np.any(pa[:,2]<=0) or np.any(pb[:,2]<=0):raise ValueError('Model-point projection behind camera')
    qa=pa@K.T;qb=pb@K.T
    return dict(mean_model_reprojection_px=float(np.linalg.norm(qa[:,:2]/qa[:,2:]-qb[:,:2]/qb[:,2:],axis=1).mean()),
        ADD_m=float(np.linalg.norm(pa-pb,axis=1).mean()),
        rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(a[:3,:3]@b[:3,:3].T)-1)/2,-1,1)))),
        translation_m=float(np.linalg.norm(a[:3,3]-b[:3,3])))


def mask_metrics(predicted, reference):
    if predicted is None:
        # A missing observed prediction is a scored failure, not an invented
        # model-produced empty mask (nor a perfect absent-object prediction).
        reference=np.asarray(reference,dtype=bool)
        result=mask_metrics(np.zeros_like(reference),reference)
        result.update(prediction_available=False,iou=0.,dice=0.,predicted_pixels=None,
            predicted_pixel_precision=None,empty_prediction=False,pixel_false_positive=None,
            unavailable_prediction_policy='IoU/Dice zero in all-frame aggregate; no fabricated observed mask')
        return result
    a,b=np.asarray(predicted,dtype=bool),np.asarray(reference,dtype=bool)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Native mask shapes differ')
    intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b));pa=int(a.sum());gt=int(b.sum())
    return dict(prediction_available=True,iou=intersection/union if union else 1.,dice=2*intersection/(pa+gt) if pa+gt else 1.,
        reference_pixel_recall=intersection/gt if gt else None,predicted_pixel_precision=intersection/pa if pa else None,
        predicted_pixels=pa,reference_pixels=gt,intersection_pixels=intersection,union_pixels=union,
        empty_prediction=not bool(pa),empty_reference=not bool(gt),empty_reference_false_foreground=bool(pa and not gt),
        pixel_false_positive=pa-intersection,pixel_false_negative=gt-intersection,
        iou_at_least_0_5=bool(gt and intersection/union>=.5),unavailable_prediction_policy=None)


def statistics(values):
    a=np.asarray([v for v in values if v is not None],dtype=np.float64)
    if not len(a):return dict(count=0,mean=None,p50=None,p95=None,min=None,max=None,total=None)
    if not np.isfinite(a).all():raise ValueError('Non-finite aggregate')
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()),total=float(a.sum()))


def verify_forward_counts(rows, roles, support):
    """Cross-check observational hooks, without importing any model runtime."""
    counts=support.get('pose',{}).get('observed-full-forward-counts.json',{}).get('record')
    if counts is None:return dict(status='unavailable',total_full_network_forwards=None)
    out={};total=0
    for index,role in enumerate(roles):
        expected=sum(iteration_count(r) for r in rows if r['backend_role']==role)
        actual=counts.get('network_'+str(index),{})
        if set(actual)!={'backbone','decoder','pose_head'} or any(value!=expected for value in actual.values()):
            raise ValueError('Observed full-forward counts differ from native refinement iterations')
        out[role]=dict(full_network_forwards=expected,scored_native_rows=sum(r['backend_role']==role for r in rows),components=actual)
        total+=expected
    if len(counts)!=len(roles):raise ValueError('Unexpected extra network in observed-forward counts')
    return dict(status='crosschecked_native_iterations',mapping='one network loaded per role in native run.roles order; observational hooks only',roles=out,total_full_network_forwards=total)


def iteration_count(row):
    value=row.get('metrics',{}).get('iterations')
    if value is None:return 0
    if isinstance(value,bool) or not np.isfinite(value) or value<0 or int(value)!=value:raise ValueError('Native refinement iteration count is invalid')
    return int(value)


def verify_seed_usage(rows,roles,schedule,masks,seed_target,seed_label):
    """Allow no seed only when the actual first-target no-init path proves it."""
    out={}
    for role in roles:
        rr=[r for r in rows if r['backend_role']==role]
        if any(r.get('evidence',{}).get('initialization') not in (None,'prior_validated_pose',seed_label) for r in rr):
            raise ValueError('Unknown/GT/different initialization source is forbidden')
        seed_rows=[r for r in rr if (r.get('evidence',{}).get('initialization') or '').startswith(('onboarding:','controlled_init:'))]
        if seed_rows:
            if len(seed_rows)!=1 or seed_rows[0]['frame_id']!=seed_target or seed_rows[0].get('evidence',{}).get('initialization')!=seed_label:
                raise ValueError('Late/multiple/different seed use is forbidden')
            if iteration_count(seed_rows[0])<=0:raise ValueError('Seed-use receipt has no actual refinement attempt')
            observed=masks.get(seed_target);seed_row=seed_rows[0]
            if observed is None or not observed.any():raise ValueError('Seed/network attempt contradicts empty/unavailable initial observed mask')
            if seed_row.get('mask_state')!='observed' or seed_row.get('mask_frame_id')!=seed_target or seed_row.get('mask_sha256')!=mask_sha256(observed):
                raise ValueError('Seed-use receipt lacks fresh same-frame nonempty observed mask identity')
            out[role]=dict(seed_uses=1,seed_target_frame_id=seed_target,status='single_first_target_seed_use')
            continue
        first=next((r for r in rr if r['frame_id']==seed_target),None)
        if first is None or seed_target!=schedule[0]:raise ValueError('Zero seed use lacks first-target native no-init evidence')
        reason=first.get('failure_reason');mask=masks.get(seed_target)
        empty=(reason=='observed_mask_empty' and first.get('mask_state')=='observed' and mask is not None and not mask.any())
        unavailable=(reason=='observed_mask_unavailable' and first.get('mask_state')=='unavailable'
            and bool(first.get('evidence',{}).get('mask_unavailable_reason')) and first.get('mask_frame_id') is None and first.get('mask_sha256') is None)
        if not (empty or unavailable):raise ValueError('Zero seed use lacks explicit initial empty/unavailable-mask no-init evidence')
        if any(r.get('pose') is not None or iteration_count(r)>0 or r.get('evidence',{}).get('initialization') for r in rr):
            raise ValueError('Zero-seed no-init run contains a public pose, network attempt, or hidden/late initialization')
        out[role]=dict(seed_uses=0,seed_target_frame_id=seed_target,status='verified_initial_mask_no_init',
            first_target_failure_reason=reason,note='No seed use is not successful initialization; every missing/lost/unavailable row remains a coverage failure')
    return out


def protocol_metadata(clip,bundle,indexed,native,manifest):
    by={f['frame_id']:f for f in clip['frames']};schedule=native['scheduled_frame_ids'];roles=native['roles']
    indices=[by[fid]['source_frame_index'] for fid in schedule]
    overlap='documented_training_object_overlap' if bundle['obj_id']==20 else 'not_audited; do not infer unseen-object generalization'
    return dict(clip_id=clip['clip_id'],dataset_clip_id=indexed['clip_id'],scene_id=indexed['scene_id'],object_id=bundle['obj_id'],
        profile=clip.get('provenance',{}).get('profile','unspecified'),scored_frame_count=len(schedule),scored_frame_ids=schedule,
        scored_source_indices=indices,scored_source_frame_bounds=[min(indices),max(indices)],
        onboarding_frame_ids=clip['onboarding']['keyframe_ids'],seed_frame_id=bundle['seed_frame_id'],
        seed_source_frame=by[bundle['seed_frame_id']]['source_frame_index'],native_roles=roles,native_role_count=len(roles),
        SAM_input_mask_count=len(manifest['masks']),training_object_overlap_status=overlap,
        scope=f"HB scene{indexed['scene_id']} object{bundle['obj_id']}, {len(schedule)} scored frames, {len(roles)} native role(s), profile {clip.get('provenance',{}).get('profile','unspecified')}; functional/development evaluation, not unseen-object generalization, phone/live/frontend or artwork quality")


def role_processing_latency(rows,sam_times):
    attempted=[r for r in rows if iteration_count(r)>0];state_only=[r for r in rows if iteration_count(r)==0]
    def times(rr,key):return statistics([r.get('metrics',{}).get(key) for r in rr])
    return dict(observed_rows=len(rows),true_network_attempt_rows=len(attempted),state_only_no_network_rows=len(state_only),
        total_native_refinement_forwards=sum(iteration_count(r) for r in rows),
        network_attempt_cached_mask_frame_ms=times(attempted,'end_to_end_latency_ms'),network_attempt_tracking_ms=times(attempted,'tracking_ms'),
        state_only_frame_ms=times(state_only,'end_to_end_latency_ms'),state_only_tracking_ms=times(state_only,'tracking_ms'),
        state_only_failure_reasons=dict(Counter(r.get('failure_reason','unspecified') for r in state_only)),
        estimated_serial_SAM_plus_native_attempt_frame_ms=statistics([sam_times[r['frame_id']]+r['metrics']['end_to_end_latency_ms'] for r in attempted if r['metrics'].get('end_to_end_latency_ms') is not None and sam_times.get(r['frame_id']) is not None]),
        note='All-frame processing may become cheap after irreversible loss. Only iterations>0 rows attempt the network; state-only lost rows are NOT faster model inference. Attempt timings include all refinement iterations, not per-forward timings.')


def verify_row_input_identity(row,frame,image_sha,observed_mask):
    """A no-input failure is evidence, not a silently accepted stale input."""
    fid=row['frame_id']
    if row['source_frame_index']!=frame['source_frame_index'] or row['timestamp_s'] is not None or row['media_time_s'] is not None:
        raise ValueError('Native prediction frame/clock differs')
    no_inference=(row.get('pose') is None and row.get('evidence',{}).get('boundary')=='no inference for this frame'
        and row.get('failure_reason','').startswith(('rgb_unavailable:','tracking_invalidated:','rgb_dimension_mismatch:','image_identity_mismatch:','decoder_media_time_mismatch:')))
    if no_inference:
        if row['image_sha256'] is not None or row['mask_frame_id'] is not None or row['mask_sha256'] is not None or row['mask_state'] not in ('blocked','rejected'):
            raise ValueError('No-inference row falsely claims consumed RGB/mask identities')
        return
    if row['image_sha256']!=image_sha:raise ValueError('Stale/different prediction RGB')
    if row['mask_state']=='observed':
        if observed_mask is None or row['mask_frame_id']!=fid or row['mask_sha256']!=mask_sha256(observed_mask):
            raise ValueError('Stale/different prediction observed mask')
    elif not (row['mask_state']=='unavailable' and row.get('pose') is None and row.get('failure_reason')=='observed_mask_unavailable'
              and row.get('evidence',{}).get('mask_unavailable_reason') and row['mask_frame_id'] is None and row['mask_sha256'] is None):
        raise ValueError('Unavailable/rejected mask lacks explicit native no-input receipt')


def summarize_role(rows, segmentation, schedule, visible_threshold=.5, pixel_threshold=5.):
    by={r['frame_id']:r for r in rows};all_rows=[]
    for fid in schedule:
        row=by.get(fid,dict(frame_id=fid,source_frame_index=segmentation[fid].get('source_frame_index'),pose_state='missing_result',render_state='unavailable',pose=None,
            valid_public_pose=False,pose_error=None,reference_visibility=segmentation[fid]['reference_visibility'],metrics={},failure_reason='missing native result'))
        error=row.get('pose_error');qualified=bool(row.get('valid_public_pose') and error is not None and error['mean_model_reprojection_px']<pixel_threshold)
        row=dict(row,qualified_reprojection=qualified,joint_mask_and_pose=bool(qualified and segmentation[fid]['iou_at_least_0_5']))
        all_rows.append(row)
    visible=[r for r in all_rows if r['reference_visibility']>=visible_threshold]
    errors=[r['pose_error'] for r in all_rows if r.get('valid_public_pose') and r.get('pose_error') is not None]
    summary=dict(scheduled_frames=len(schedule),result_rows=len(rows),missing_rows=len(schedule)-len(rows),
        pose_state_counts=dict(Counter(r['pose_state'] for r in all_rows)),render_state_counts=dict(Counter(r['render_state'] for r in all_rows)),
        valid_public_poses=sum(bool(r.get('valid_public_pose')) for r in all_rows),
        invalid_public_poses=sum(r.get('pose') is not None and not r.get('valid_public_pose') for r in all_rows),
        reprojection_qualified_frames=sum(r['qualified_reprojection'] for r in all_rows),
        all_frame_reprojection_coverage=sum(r['qualified_reprojection'] for r in all_rows)/len(schedule),
        visible_frames=len(visible),visible_reprojection_qualified_frames=sum(r['qualified_reprojection'] for r in visible),
        visible_reprojection_coverage=sum(r['qualified_reprojection'] for r in visible)/len(visible) if visible else None,
        joint_mask_and_pose_frames=sum(r['joint_mask_and_pose'] for r in all_rows),
        all_frame_joint_coverage=sum(r['joint_mask_and_pose'] for r in all_rows)/len(schedule),
        conditional_public_pose_errors={key:statistics([e[key] for e in errors]) for key in ('mean_model_reprojection_px','ADD_m','rotation_deg','translation_m')},
        prior_source_counts=dict(Counter(r.get('evidence',{}).get('initialization','unavailable') for r in all_rows)))
    return summary,all_rows


def verified_inputs(workspace,run,native,seal,rows):
    """Verify all prediction inputs, without touching any evaluator labels."""
    workspace=Path(workspace);clip_path=workspace/'clips'/native['clip_id']/'clip.json';clip=load(clip_path)
    wc.ClipSpec.from_dict(clip)
    if clip['clock_kind']!='frame_index' or clip['intrinsics']['status']!='dataset_reference':raise ValueError('Expected dataset-reference frame-index clip')
    if clip['coordinates']!=dict(camera_axes='opencv_x_right_y_down_z_forward',pixel_origin='integer_pixel_centers',pose='camera_from_object',vectors='column'):
        raise ValueError('Unexpected camera/object axes')
    frames={f['frame_id']:f for f in clip['frames']}
    if not set(seal['scoredFrameIDs'])<=set(frames) or set(clip['onboarding']['keyframe_ids'])&set(seal['scoredFrameIDs']):raise ValueError('Scored/onboarding frame leak')
    config=load(workspace/'config.json');root=Path(config['roots'][clip['media']['root_id']]);rgb_root=contained(root,clip['media']['path'])
    mask_path=workspace/'masks'/native['clip_id']/native['config']['mask_set']/'manifest.json';manifest=load(mask_path)
    if wc.canonical_sha256(manifest)!=seal['nativeMaskManifestSHA256']:raise ValueError('Observed-mask manifest differs from seal')
    if manifest.get('cancelled') or manifest.get('evaluator_inputs_read') is not False or manifest['clip_id']!=native['clip_id'] or manifest['producer']['kind']!='sam2.1_video_prompted':
        raise ValueError('Expected actual observed prompted SAM masks')
    K=np.asarray(clip['intrinsics']['K']).reshape(3,3)
    bundle_path=workspace/'metric-bundles'/(native['clip_id']+'.json');bundle=load(bundle_path)
    if bundle['principal_point_origin']!='integer_centers' or bundle['seed_frame_id'] not in clip['onboarding']['keyframe_ids']:raise ValueError('Metric seed is not authorized onboarding')
    model=checked(bundle['glb_path'],bundle['glb_sha256'])
    # Pure metadata parsing only: no model loading, inference, or rendering.
    # Recompute the production boundary's exact complete seed/bundle binding
    # instead of trusting the mutable local metric-bundle path by itself.
    from . import workbench_frame_boundary as fb
    from .workbench_neural import MetricBundle
    binding=fb.neural_binding(wc.ClipSpec.from_dict(clip),wc.RunSpec.from_dict(native),family=bundle['family'],bundle=MetricBundle.from_file(bundle_path))
    if binding['warmup_frame_ids']:raise ValueError('Unscored native warmup requires separate captured seed/forward receipts; unsupported by this evaluator')
    binding_sha=wc.canonical_sha256(binding);report=load(Path(run)/'report.json')
    for role in seal['roles']:
        if report['stages'][role]['execution']['binding_sha256']!=binding_sha:raise ValueError('Sealed neural seed/bundle binding differs')
    images={};masks={};input_hashes=[]
    seed_source_index=frames[bundle['seed_frame_id']]['source_frame_index']
    for fid in seal['scoredFrameIDs']:
        frame=frames[fid]
        if fid!=f"f{frame['source_frame_index']:06d}" or frame['source_frame_index']<=seed_source_index or frame.get('timestamp_s') is not None or frame.get('media_time_s') is not None:
            raise ValueError('Scored source frame identity/clock differs')
        path=checked(contained(rgb_root,frame['image_path']),frame['image_sha256']);rgb=np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape!=(clip['height'],clip['width'],3):raise ValueError('Native RGB dimensions differ')
        spec=manifest['masks'].get(fid);mask=None
        if spec is not None:
            mp=checked(contained(mask_path.parent,spec['path']),spec['sha256']);raw=np.asarray(Image.open(mp))
            if raw.shape!=(clip['height'],clip['width']) or not set(np.unique(raw))<= {0,1,255}:raise ValueError('Native binary mask dimensions/values differ')
            mask=raw>0
            if mask_sha256(mask)!=spec['content_sha256'] or bool(not mask.any())!=spec['empty']:raise ValueError('Observed mask decoded identity differs')
        images[fid]=array_sha256(rgb);masks[fid]=mask
        input_hashes.append(dict(frame_id=fid,source_frame_index=frame['source_frame_index'],RGBEncodedSHA256=frame['image_sha256'],RGBArraySHA256=images[fid],
            maskAvailable=spec is not None,maskEncodedSHA256=spec['sha256'] if spec else None,maskArraySHA256=spec['content_sha256'] if spec else None))
    for row in rows:
        fid=row['frame_id'];identity=row['backend_identity']
        verify_row_input_identity(row,frames[fid],images[fid],masks[fid])
        if identity.get('metric_units')!='metres' or identity.get('principal_point_origin')!='integer_centers' or identity.get('effective_K')!=clip['intrinsics']['K'] or identity.get('glb_sha256')!=bundle['glb_sha256']:
            raise ValueError('Prediction metric geometry/K differs')
        # Native contract calls these units "metric"; the backend identity
        # explicitly binds that metric convention to metres above.
        if row['pose'] is not None and row['pose_units']!='metric':raise ValueError('Public pose is not native metric units')
        if identity.get('seed_frame_id')!=bundle['seed_frame_id'] or identity.get('seed_target_frame_id')!=binding['seed_target_frame_id']:raise ValueError('Seed target/identity differs')
        if identity.get('execution_scope')!='finite_recorded_only' or identity.get('live_eligible') is not False:
            raise ValueError('Expected recorded-only execution identity')
    seed_usage=verify_seed_usage(rows,seal['roles'],seal['scoredFrameIDs'],masks,binding['seed_target_frame_id'],bundle['seed_label'])
    return clip,bundle,model,K,masks,manifest,input_hashes,dict(seedUsageByRole=seed_usage,clipSHA256=digest(clip_path),metricBundleSHA256=digest(bundle_path),neuralBindingSHA256=binding_sha,observedMaskFileSHA256=digest(mask_path),observedMaskCanonicalSHA256=wc.canonical_sha256(manifest))


def evaluate(workspace,run,seal_path,seal_sha256,pack,reference_root,stage_record=None,sam_record=None,
             allow_reused_reference_development=False,companion_seals=None):
    run=Path(run);pack=Path(pack)
    seal,native,rows=verify_seal(run,seal_path,seal_sha256,allow_reused_reference_development)
    reference_attestation=reference_protocol(seal,allow_reused_reference_development)
    companions=[]
    for other_run,other_path,other_sha in companion_seals or []:
        other_seal,other_native,other_rows=verify_seal(other_run,other_path,other_sha,allow_reused_reference_development)
        companions.append((other_run,other_sha,other_seal,other_native,other_rows))
    verify_comparison_seals(seal,[c[2] for c in companions])
    clip,bundle,model,K,masks,manifest,inputs,bindings=verified_inputs(workspace,run,native,seal,rows)
    companion_bindings=[];primary_sources=load(run/'report.json')['dependencies']['sources']
    for other_run,other_sha,other_seal,other_native,other_rows in companions:
        other=verified_inputs(workspace,other_run,other_native,other_seal,other_rows)
        if other[0]['clip_id']!=clip['clip_id'] or other[6]!=inputs or other[7]['neuralBindingSHA256']!=bindings['neuralBindingSHA256']:
            raise ValueError('Comparison roles use different native RGB/mask/seed/metric inputs')
        if load(Path(other_run)/'report.json')['dependencies']['sources']!=primary_sources:
            raise ValueError('Comparison native source/math hashes differ')
        companion_bindings.append(dict(predictionSealSHA256=other_sha,runID=other_seal['run_id'],roles=other_seal['roles'],
            files=other_seal['files'],inputBindings=other[7]))
    vertices,sampling=model_vertices(model)
    # IMPORTANT: this invocation reads no evaluator/scored GT file before this
    # boundary. V2 also requires the externally anchored companion role seals.
    # This does NOT erase previously accessed development-reference history.
    lock=load(pack/'LOCK.json');evaluator_path=checked(pack/'evaluator/evaluator.json',lock['evaluator_sha256']);evaluator=load(evaluator_path)
    index_path=pack/'inference/index.json'
    if digest(index_path)!=evaluator['inference_index_sha256'] or digest(index_path)!=lock['inference_index_sha256']:raise ValueError('Evaluator/input index differs')
    index=load(index_path)
    candidates=[obj for obj in evaluator['objects'] if clip['clip_id'].startswith(obj['clip_id']+'-') or clip['clip_id']==obj['clip_id']]
    if len(candidates)!=1:raise ValueError('Ambiguous evaluator clip identity')
    indexed=[obj for obj in index['objects'] if obj['clip_id']==candidates[0]['clip_id']]
    if len(indexed)!=1 or indexed[0]['model_sha256']!=bundle['glb_sha256'] or indexed[0]['object_id']!=bundle['obj_id']:
        raise ValueError('Evaluator/model object binding differs')
    metadata=protocol_metadata(clip,bundle,indexed[0],native,manifest)
    gt={f['frame_id']:f for f in candidates[0]['frames']};segmentation={};gt_hashes=[]
    reference_rotation_audit={};prediction_rotation_audit=[]
    for fid in seal['scoredFrameIDs']:
        g=gt[fid]
        wc._pose(g['reference_camera_from_object_m'],'dataset reference pose')
        validate_pose(g['reference_camera_from_object_m'],rotation_tolerance=1e-4)
        reference_rotation_audit[fid]=rotation_residuals(g['reference_camera_from_object_m'])
        if g['source_frame_index']!=next(f['source_frame_index'] for f in clip['frames'] if f['frame_id']==fid):raise ValueError('Evaluator frame identity differs')
        spec=g['reference_visible_mask'];path=checked(contained(reference_root,spec['source_member']),spec['sha256']);reference=np.asarray(Image.open(path))>0
        segmentation[fid]=dict(frame_id=fid,source_frame_index=g['source_frame_index'],reference_visibility=g['reference_visibility'],**mask_metrics(masks[fid],reference))
        gt_hashes.append(dict(frame_id=fid,visibleMaskSHA256=spec['sha256']))
    role_summaries={};scored=[]
    for role in seal['roles']:
        rr=[]
        for source in rows:
            if source['backend_role']!=role:continue
            row=dict(source,reference_visibility=gt[source['frame_id']]['reference_visibility'],valid_public_pose=False,pose_error=None,pose_validation_error=None)
            if source['pose'] is not None:
                row['prediction_rotation_residuals']=rotation_residuals(source['pose']);prediction_rotation_audit.append(row['prediction_rotation_residuals'])
                try:
                    row['pose_error']=pose_errors(source['pose'],gt[source['frame_id']]['reference_camera_from_object_m'],vertices,K);row['valid_public_pose']=True
                except ValueError as error:row['pose_validation_error']=str(error)
            rr.append(row)
        summary,scored_rows=summarize_role(rr,segmentation,seal['scoredFrameIDs']);role_summaries[role]=summary
        scored.extend(dict(row,backend_role=role) for row in scored_rows)
    segmentation_summary=dict(scored_frames=len(segmentation),**{key:statistics([s[key] for s in segmentation.values()]) for key in ('iou','dice','reference_pixel_recall','predicted_pixel_precision')},
        iou_at_least_0_5_frames=sum(s['iou_at_least_0_5'] for s in segmentation.values()),
        all_frame_IoU_at_least_0_5_coverage=sum(s['iou_at_least_0_5'] for s in segmentation.values())/len(segmentation),
        unavailable_predictions=sum(not s['prediction_available'] for s in segmentation.values()),
        empty_predictions=sum(s['empty_prediction'] for s in segmentation.values()),empty_references=sum(s['empty_reference'] for s in segmentation.values()),
        empty_reference_false_foreground_frames=sum(s['empty_reference_false_foreground'] for s in segmentation.values()),
        note='IoU>=0.5 is an explicit segmentation proxy; missing observed predictions get IoU/Dice zero and remain in denominator, never invented mask bytes. Empty-reference false-foreground rate is undefined if no empty references')
    report=load(run/'report.json');telemetry=report['telemetry'];sam_times=manifest['telemetry']['segmentation_ms_by_frame_id'];latencies={}
    onboarding_times=[sam_times.get(fid) for fid in clip['onboarding']['keyframe_ids']]
    onboarding_total=sum(onboarding_times) if all(value is not None for value in onboarding_times) else None
    sam_initialization=manifest['telemetry'].get('initialization_ms')
    for role in seal['roles']:
        rr=[r for r in rows if r['backend_role']==role]
        native_times=[r['metrics'].get('end_to_end_latency_ms') for r in rr]
        combined=[sam_times[r['frame_id']]+r['metrics']['end_to_end_latency_ms'] for r in rr if r['metrics'].get('end_to_end_latency_ms') is not None and sam_times.get(r['frame_id']) is not None]
        init=telemetry['stages'].get(role+'_initialization',{}).get('total_ms')
        latencies[role]=dict(native_cached_mask_frame_ms=statistics(native_times),tracking_ms=statistics([r['metrics'].get('tracking_ms') for r in rr]),
            model_initialization_ms=init,estimated_serial_SAM_plus_native_frame_ms=statistics(combined),
            estimated_serial_profile_total_including_initialization_ms=(sum(combined)+init+sam_initialization+onboarding_total) if len(combined)==len(seal['scoredFrameIDs']) and init is not None and sam_initialization is not None and onboarding_total is not None else None,
            processing_attempt_split=role_processing_latency(rr,sam_times),
            note='Sum of separately measured matching-frame stages, not measured integrated/live-stream latency. SAM cache reused; model initializations separate. Includes recovering/lost/state-only rows. Do not interpret all-frame average as network inference speed after loss.')
    support={}
    for name,folder in (('pose',stage_record),('SAM',sam_record)):
        if folder:
            folder=Path(folder);support[name]={}
            for filename in ('runtime.json','supervision.json','input-freeze.json','observed-full-forward-counts.json'):
                path=folder/filename
                if path.exists():support[name][filename]=dict(sha256=digest(path),record=load(path))
    if stage_record and 'supervision.json' in support['pose'] and support['pose']['supervision.json']['sha256']!=seal['supervisionSHA256']:
        raise ValueError('Pose supervision differs from prediction seal')
    forwards=verify_forward_counts(rows,seal['roles'],support)
    walls={name:records.get('supervision.json',{}).get('record',{}).get('elapsedSeconds') for name,records in support.items()}
    wall_sum=sum(walls.values()) if len(walls)==2 and all(value is not None for value in walls.values()) else None
    cpu_name=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None) if Path('/proc/cpuinfo').exists() else None
    summary=dict(schema='offline_sam_pose_evaluation_v2',privacy='PRIVATE: new result publication not authorized',
        scope=metadata['scope'],dataset_run=metadata,
        reference_access_protocol=reference_attestation,
        admission='fresh local pinned recorded-artifact verification; not stock historical-receipt admission; finite recorded execution only',
        thresholds=dict(mean_model_reprojection_px_strictly_less_than=5,reference_visibility_at_least=.5,segmentation_IoU_at_least=.5),
        coordinate_system=dict(pose='camera_from_object',camera_axes='OpenCV x-right/y-down/z-forward',units='metres',K=K.reshape(-1).tolist(),pixel_origin='integer_centers',GLBSourceAxes='unchanged BOP source-object axes'),
        protocol=dict(scored_source_frames=metadata['scored_source_frame_bounds'],scored_frame_ids=metadata['scored_frame_ids'],onboarding_source_frame=metadata['seed_source_frame'],one_annotation_derived_AI_reviewed_prompt=True,
            authorized_known_pose_seed_budget_per_role=1,actual_seed_usage_by_role=bindings['seedUsageByRole'],reseed_after_loss=False,scoredGTFeedback=False,warmup_frames=0,repeats=1,
            same_mask_manifest_used_by_roles_in_this_run=len(seal['roles'])>1,SAM_cache_computed_before_this_pose_run=True,
            note='This scoring invocation reads references only after complete sealed prediction/input validation (all declared comparison roles for v2). Previously accessed reference history is NOT a held-out test. Dataset-reference proxy, not independent human attachment labels. Quantiles use NumPy linear percentile, not native nearest-rank timing summaries.'),
        bindings=dict(predictionSealSHA256=seal_sha256,runID=seal['run_id'],sealedRunFiles=seal['files'],verifiedCompanionPredictions=companion_bindings,
            evaluatorSHA256=digest(evaluator_path),inferenceIndexSHA256=digest(index_path),GLBSHA256=digest(model),**bindings),
        geometry_sampling=sampling,segmentation=segmentation_summary,roles=role_summaries,
        rotation_validation=dict(prediction_tolerance=1e-6,reference_tolerance=1e-4,
            reference_policy='existing workbench_contract._pose tolerance for serialized dataset-reference matrices; no orthogonalization or axis changes',
            reference_residuals_by_frame=reference_rotation_audit,
            reference_max_abs_RtR_minus_I=max(a['max_abs_RtR_minus_I'] for a in reference_rotation_audit.values()),
            reference_max_abs_detR_minus_1=max(a['abs_detR_minus_1'] for a in reference_rotation_audit.values()),
            prediction_max_abs_RtR_minus_I=max((a['max_abs_RtR_minus_I'] for a in prediction_rotation_audit),default=None),
            prediction_max_abs_detR_minus_1=max((a['abs_detR_minus_1'] for a in prediction_rotation_audit),default=None),
            note='A reference-validation failure aborts evaluation; it is not counted as a failed prediction. Raw reference numbers used for ADD/reprojection/trace-angle, no symmetry matching.'),
        metric_caveat='ADD and reprojection use the same512 deterministic sampled original model vertices, not full-mesh official BOP scores; no symmetry matching. Conditional pose errors exclude unavailable outputs, which remain failures in coverage denominators.',
        latency=dict(SAM_initialization_ms=sam_initialization,SAM_scored_frame_ms=statistics([sam_times.get(f) for f in seal['scoredFrameIDs']]),
            SAM_onboarding_frame_ms=onboarding_total,native_pose_run_wall_ms=telemetry.get('run_wall_ms'),native_run_role_count=len(seal['roles']),roles=latencies,
            supervised_stage_wall_seconds=walls,actual_separate_jobs_wall_sum_seconds=wall_sum,
            wall_sum_scope=f"Separate SAM job ({len(manifest['masks'])} stored masks) once +this native run ({len(seal['roles'])} role(s), {len(seal['scoredFrameIDs'])} scheduled frames/role); process start/teardown included, not integrated/live-stream measurement. When comparing separate role runs, count the identical shared SAM job once, not once per evaluation file.",
            actual_forward_counts=forwards,
            caveat='Native end_to_end_latency_ms uses precomputed SAM masks: it is NOT end-to-end SAM+pose latency. No physical FPS supplied by source; no real-time claim.'),
        provenance=dict(evaluatorAdapterSHA256=digest(__file__),datasetLockSHA256=digest(pack/'LOCK.json'),cpu_model_current_host=cpu_name,nativeSources=report['dependencies']['sources'],versions=report['dependencies']['versions'],
            backendIdentities={role:next(r['backend_identity'] for r in rows if r['backend_role']==role) for role in seal['roles']},observationalSupport=support),
        score_consistency=dict(original_rows_preserved=len(rows),scored_rows_including_missing=len(scored),all_retained_input_assets_and_consumed_RGB_mask_identities_checked=True,
            unavailable_observed_prediction_frames=sum(mask is None for mask in masks.values()),all_scored_gt_masks_checked=True))
    return summary,scored,list(segmentation.values()),dict(nativeInputs=inputs,evaluatorMasks=gt_hashes)


def write_outputs(destination,summary,rows,masks,index):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();path=destination/name
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Refusing to overwrite a differing evaluation artifact')
        path.write_bytes(raw)
    write('summary.json',summary);write('scored-frames.json',rows);write('segmentation-frames.json',masks);write('input-hash-index.json',index)
    flat=[]
    for r in rows:
        flat.append(dict(backend_role=r['backend_role'],frame_id=r['frame_id'],source_frame_index=r.get('source_frame_index'),pose_state=r['pose_state'],render_state=r['render_state'],
            public_pose=r.get('valid_public_pose',False),reference_visibility=r['reference_visibility'],qualified_reprojection=r['qualified_reprojection'],joint_mask_and_pose=r['joint_mask_and_pose'],
            **(r['pose_error'] or {k:None for k in ('mean_model_reprojection_px','ADD_m','rotation_deg','translation_m')}),failure_reason=r.get('failure_reason'),pose_validation_error=r.get('pose_validation_error')))
    for name,table in (('pose-frames.csv',flat),('segmentation-frames.csv',masks)):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table);path=destination/name
        raw=stream.getvalue().encode()
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Refusing to overwrite differing CSV')
        path.write_bytes(raw)
    return {p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in destination.iterdir() if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('workspace','run','seal','seal-sha256','pack','reference-root','destination'):parser.add_argument('--'+flag,required=True)
    parser.add_argument('--stage-record');parser.add_argument('--sam-record')
    parser.add_argument('--allow-reused-reference-development',action='store_true',help='Explicitly allow expanded development on previously accessed references; never labels it held-out')
    parser.add_argument('--companion-seal',nargs=3,action='append',metavar=('RUN_DIR','SEAL_PATH','SHA256'),help='Independent anchor for another declared native comparison role; all validated before reference access')
    args=parser.parse_args()
    result=evaluate(args.workspace,args.run,args.seal,args.seal_sha256,args.pack,args.reference_root,args.stage_record,args.sam_record,
        args.allow_reused_reference_development,args.companion_seal)
    files=write_outputs(args.destination,*result);print(json.dumps(dict(destination=args.destination,segmentation=result[0]['segmentation'],roles=result[0]['roles'],files=files),indent=2))


if __name__=='__main__':main()
