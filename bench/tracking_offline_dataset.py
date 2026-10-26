"""Prepare the real HB recorded SAM -> metric pose dataset, without running models.

Only original frame 9 is authorized for the one-shot pose seed and annotation-
assisted prompt review. Original frames 10..309 are scored, with their labels
kept in a separate evaluator tree. Native clips use frame-index time, original
f000009/f000010 IDs and dataset-reference intrinsics, never invented physical FPS.

Preparation is two-phase: first create point proposals/review sheets; after
actual visual review repeat with --reviewed-by and --review-evidence-sha256.
No admission, SAM mask, model prediction or quality result is manufactured.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import workbench_contract as wc
from .workbench_cli import Workspace, FrameReader
from .workbench_neural import MetricBundle, glb_inventory, glb_mesh
from .vision import cv2

ROOT=Path(__file__).resolve().parents[1]
REVISION='985ebaef7b1bd4bffb2039f18dd81faa647fd060'
DATASET_ID='offline-sam-pose-v1'
FAMILY='offline_hb_640x480'
ROOT_ID='offline_hb_inputs'
SOURCE_RELEASE='https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1'
SOURCE_INFERENCE_SHA256='1bdc84f5865797c308c2a46dad67c90f532ecb3e963ac5713b80acd820831af4'
SOURCE_EVALUATOR_SHA256='1fc8fe971bf000c244b35d8416d473684f826fe756ae91dea2e2a41b861d9956'
DEFAULT_SOURCE=ROOT/'.cache/tracking-benchmark/real-v1'
DEFAULT_OUTPUT=ROOT/'.cache/tracking-benchmark/offline-sam-pose-v1'
DEFAULT_ARTIFACTS=ROOT/'datasets/offline-sam-pose-v1'
PROFILES={'smoke':10,'pilot':30,'long':110,'full':300}
SEED=9


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def write_new(path,raw):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Retained artifact differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw)


def write_json(path,value):write_new(path,canonical(value))


def resource(root,spec):
    root=Path(root).resolve();path=(root/spec['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink() or digest(path)!=spec['sha256']:
        raise ValueError('Source resource identity/path differs: '+str(spec.get('path')))
    return path


def link_verified(source,target,expected):
    source,target=Path(source),Path(target)
    if target.exists():
        if target.is_symlink() or digest(target)!=expected:raise ValueError('Existing linked resource differs')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    # No silent copy fallback: a cross-device consumer should point preparation
    # at the locally materialized source dataset, not duplicate a large corpus.
    os.link(source,target)
    if digest(target)!=expected:raise ValueError('Hardlinked resource differs')


def source_pose(record):
    pose=np.eye(4,dtype=np.float64)
    pose[:3,:3]=np.asarray(record['evaluationPose']['cam_R_m2c']).reshape(3,3)
    pose[:3,3]=np.asarray(record['evaluationPose']['cam_t_m2c'])*.001
    wc._pose(pose.tolist(),'authorized frame9 seed')
    return pose.tolist()


def convert_ply_to_glb(source,destination):
    """Exact topology/axes, mm -> m; preserve original normals and vertex RGBA."""
    with Path(source).open(encoding='ascii') as stream:
        if stream.readline().strip()!='ply' or stream.readline().strip()!='format ascii 1.0':raise ValueError('ASCII BOP PLY required')
        properties=[];vertex_count=face_count=0;element=None
        for line in stream:
            words=line.split()
            if words[:2]==['element','vertex']:vertex_count=int(words[2]);element='vertex'
            elif words[:2]==['element','face']:face_count=int(words[2]);element='face'
            elif words and words[0]=='property' and element=='vertex':properties.append(words[-1])
            elif words==['end_header']:break
        if not 1<=vertex_count<=1_000_000 or not 1<=face_count<=1_000_000:raise ValueError('Source geometry outside bounded corpus limits')
        vertices=np.loadtxt(stream,dtype=np.float64,max_rows=vertex_count,ndmin=2)
        polygons=np.loadtxt(stream,dtype=np.int64,max_rows=face_count,ndmin=2)
    if vertices.shape!=(vertex_count,len(properties)) or polygons.shape!=(face_count,4) or not np.all(polygons[:,0]==3):raise ValueError('Only complete triangle PLY is supported')
    original=vertices[:,[properties.index(k) for k in ('x','y','z')]]*.001
    positions=np.ascontiguousarray(original,dtype='<f4')
    normals=np.ascontiguousarray(vertices[:,[properties.index(k) for k in ('nx','ny','nz')]],dtype='<f4')
    colors=np.ascontiguousarray(vertices[:,[properties.index(k) for k in ('red','green','blue','alpha')]],dtype=np.uint8)
    faces=np.ascontiguousarray(polygons[:,1:],dtype='<u4')
    if not np.isfinite(positions).all() or not np.isfinite(normals).all() or faces.max()>=vertex_count:raise ValueError('Invalid source geometry')
    arrays=[positions,normals,colors,faces.reshape(-1)]
    views=[];binary=bytearray()
    for array in arrays:
        while len(binary)%4:binary.append(0)
        views.append(dict(buffer=0,byteOffset=len(binary),byteLength=array.nbytes))
        binary.extend(array.tobytes())
    accessors=[dict(bufferView=0,componentType=5126,count=vertex_count,type='VEC3',min=positions.min(0).tolist(),max=positions.max(0).tolist()),
        dict(bufferView=1,componentType=5126,count=vertex_count,type='VEC3'),
        dict(bufferView=2,componentType=5121,count=vertex_count,type='VEC4',normalized=True),
        dict(bufferView=3,componentType=5125,count=face_count*3,type='SCALAR')]
    document=dict(asset=dict(version='2.0',generator='VisualizeIT offline-sam-pose-v1; source axes unchanged; millimetres multiplied by0.001'),
        scene=0,scenes=[dict(nodes=[0])],nodes=[dict(mesh=0)],
        meshes=[dict(primitives=[dict(attributes={'POSITION':0,'NORMAL':1,'COLOR_0':2},indices=3,material=0,mode=4)])],
        materials=[dict(pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1],metallicFactor=0,roughnessFactor=1),doubleSided=False)],
        buffers=[dict(byteLength=len(binary))],bufferViews=views,accessors=accessors)
    body=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    body+=b' '*((-len(body))%4);binary.extend(b'\0'*((-len(binary))%4))
    raw=struct.pack('<4sII',b'glTF',2,12+8+len(body)+8+len(binary))+struct.pack('<I4s',len(body),b'JSON')+body+struct.pack('<I4s',len(binary),b'BIN\0')+binary
    write_new(destination,raw)
    decoded,triangles=glb_mesh(destination)
    if not np.array_equal(decoded,positions.astype(np.float64)) or not np.array_equal(triangles,faces):raise ValueError('Native GLB parser does not preserve the source geometry')
    return dict(sourcePLY_SHA256=digest(source),glbSHA256=digest(destination),vertices=vertex_count,triangles=face_count,
                sourceUnits='millimetres',outputUnits='metres',sourceAxes='unchanged original BOP source-object axes',
                scale=.001,nodeTransform='identity',sourceVertexColorsPreserved=True,
                maximumMetreFloat32Quantization=float(np.max(np.abs(decoded-original))))


def prompt_points(mask):
    mask=np.asarray(mask,dtype=np.uint8)
    if mask.ndim!=2 or not mask.any():raise ValueError('Authorized onboarding mask is empty')
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    y,x=np.unravel_index(np.argmax(distance),distance.shape)
    negative=next(((xx,yy) for xx,yy in ((16,16),(mask.shape[1]-17,16),(16,mask.shape[0]-17)) if not mask[yy,xx]),None)
    if negative is None:
        yy,xx=np.argwhere(mask==0)[0];negative=(int(xx),int(yy))
    return [[int(x),int(y)]],[list(negative)]


def review_sheets(output,proposals):
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    sheets=[]
    for sheet_index,start in enumerate(range(0,len(proposals),6)):
        sheet=Image.new('RGB',(1024,3*266),'#111111')
        for i,row in enumerate(proposals[start:start+6]):
            rgb=np.asarray(Image.open(output/row['rgb_path']).convert('RGB'));mask=np.asarray(Image.open(output/row['review_mask_path']))>0
            overlay=rgb.copy();overlay[mask]=np.rint(.73*rgb[mask]+.27*np.array([10,245,215])).astype(np.uint8)
            for point in row['positive']:cv2.drawMarker(overlay,tuple(point),(20,255,20),cv2.MARKER_CROSS,13,2)
            for point in row['negative']:cv2.drawMarker(overlay,tuple(point),(255,30,30),cv2.MARKER_CROSS,13,2)
            tile=Image.new('RGB',(512,266),'#111111');tile.paste(Image.fromarray(overlay).resize((320,240)),(0,0))
            ys,xs=np.where(mask);x0=max(0,int(xs.min())-12);y0=max(0,int(ys.min())-12);x1=min(640,int(xs.max())+13);y1=min(480,int(ys.max())+13)
            crop=Image.fromarray(overlay).crop((x0,y0,x1,y1));crop.thumbnail((192,230));tile.paste(crop,(320,0))
            ImageDraw.Draw(tile).text((4,243),f"HB object {row['object_id']} | source9 | cyan=annotation, green+/red-",fill='white',font=font)
            sheet.paste(tile,((i%2)*512,(i//2)*266))
        data=io.BytesIO();sheet.save(data,'PNG');name=f'prompt-review-{sheet_index+1:02d}.png';write_new(output/'onboarding-review'/name,data.getvalue())
        sheets.append(dict(path='onboarding-review/'+name,sha256=digest(output/'onboarding-review'/name)))
    evidence=dict(schema='offline_hb_prompt_review_v1',sheets=sheets,proposalsSHA256=digest(output/'onboarding-review/proposals.json'),
                  disclosure='AI review of annotation-assisted points on authorized original frame9 only; not user taps or independent human labels')
    write_json(output/'onboarding-review/review-evidence.json',evidence)
    return digest(output/'onboarding-review/review-evidence.json')


def build_clip(obj,seed,reviewed_by):
    k=seed['calibration']['cam_K'];frames=[]
    for record in [seed]+obj['frames']:
        if record['calibration']['cam_K']!=k:raise ValueError('A native static-intrinsics clip cannot silently hide varying calibration')
        number=record['frameId']
        frames.append(wc.FrameSpec(f'f{number:06d}',number,None,None,f'{number:06d}.png',record['rgb']['sha256']))
    clip_id=f"offline-hb-scene{obj['sceneId']}-object{obj['objectId']:06d}"
    return wc.ClipSpec(clip_id,f"CC0 HB scene {obj['sceneId']} object {obj['objectId']}",
        'image_sequence','frame_index',640,480,wc.MediaRef(ROOT_ID,f"rgb/scene{obj['sceneId']}",None,None),tuple(frames),
        wc.Intrinsics('dataset_reference',tuple(k),'BOP HB scene_camera.json, original integer pixel centres; not measured physical-phone calibration'),
        None,tuple(frame.frame_id for frame in frames[1:]),'real_recorded_dataset_reference',False,
        ('scored_object_pose','scored_visible_mask','scored_depth','scored_visibility'),
        'Original BOP frame index; physical FPS/timestamps are unavailable',
        dict(dataset_revision=REVISION,license='CC0-1.0',original_frame_ids_preserved=True,
             importer_identity='f%06d original complete-directory importer ordinal, then logical prefix filtered to source9 onward',
             prompt_review='annotation-assisted AI visual review of authorized frame9 only',reviewed_by=reviewed_by))


def validate_case_template(template,root):
    """Check native case fields without pretending an unobserved SAM mask exists."""
    if template.get('schema')!='offline_sam_observed_case_template_v1' or template.get('status')!='blocked_until_actual_SAM_mask':raise ValueError('Case template status/schema differs')
    if len(template.get('cases',[]))!=1:raise ValueError('Exactly one authorized onboarding case template required')
    case=wc._keys(template['cases'][0],('case_id','family','rgb_path','rgb_sha256','mask_path','mask_sha256','segmentation','K','principal_point_origin','initial_pose_m','initialization_source','glb_path','glb_sha256','obj_id','metric_provenance'),name='pending observed case')
    wc._ident(case['case_id'],'case_id');wc._pinhole_K(case['K'],'case.K');wc._pose(case['initial_pose_m'],'case.initial_pose_m')
    if case['mask_path'] is not None or case['mask_sha256'] is not None or case['segmentation']!={'producer':'PENDING actual observed SAM2 output','manifest_sha256':None}:raise ValueError('Preparation must not counterfeit an observed SAM mask/manifest')
    if not case['initialization_source'].startswith('onboarding:') or case['metric_provenance']!='dataset_metric_cad' or case['principal_point_origin']!='integer_centers':raise ValueError('Case must retain authorized dataset-reference metric initialization')
    root=Path(root)
    for key,hash_key in (('rgb_path','rgb_sha256'),('glb_path','glb_sha256')):
        spec=dict(path=case[key],sha256=case[hash_key]);resource(root,spec)
    return case


def prepare(source=DEFAULT_SOURCE,output=DEFAULT_OUTPUT,artifacts=DEFAULT_ARTIFACTS,*,reviewed_by=None,review_evidence_sha256=None):
    source,output,artifacts=Path(source).resolve(),Path(output).resolve(),Path(artifacts)
    if source==output or source in output.parents or output in source.parents:raise ValueError('Output must not overlap the original frozen corpus')
    output.mkdir(parents=True,exist_ok=True)
    if digest(source/'inference.json')!=SOURCE_INFERENCE_SHA256:raise ValueError('Source is not the exact pinned full12-object HB inference manifest')
    manifest=json.loads((source/'inference.json').read_text())
    if len(manifest['objects'])!=12:raise ValueError('Full real dataset must contain all12 objects')
    objects=manifest['objects'];proposals=[];models={};linked={}
    for obj in objects:
        if [row['frameId'] for row in obj['onboarding']]!=list(range(10)) or [row['frameId'] for row in obj['frames']]!=list(range(10,310)):raise ValueError('Frozen source/onboarding order differs')
        if any({'evaluationPose','evaluationMask','depth','evaluationInfo'} & set(row) for row in obj['frames']):raise ValueError('Scored labels found in source inference manifest')
        seed=obj['onboarding'][-1]
        for record in [seed]+obj['frames']:
            relative=f"rgb/scene{obj['sceneId']}/{record['frameId']:06d}.png"
            if relative not in linked:
                path=resource(source,record['rgb']);link_verified(path,output/'inference'/relative,record['rgb']['sha256']);linked[relative]=dict(sha256=record['rgb']['sha256'],bytes=path.stat().st_size,source_member=record['rgb']['path'])
        model_relative=f"models/obj_{obj['objectId']:06d}.glb";audit_path=output/'model-audit'/f"object-{obj['objectId']:06d}.json"
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            if audit['sourcePLY_SHA256']!=obj['model']['sha256'] or digest(output/'inference'/model_relative)!=audit['glbSHA256']:raise ValueError('Retained metric model differs')
        else:
            audit=convert_ply_to_glb(resource(source,obj['model']),output/'inference'/model_relative);write_json(audit_path,audit)
        models[obj['objectId']]=(model_relative,audit)
        mask_path=resource(source,seed['onboardingMask']);mask_relative=f"onboarding-review/object-{obj['objectId']:06d}-source9-mask.png"
        link_verified(mask_path,output/mask_relative,seed['onboardingMask']['sha256'])
        mask=np.asarray(Image.open(mask_path))>0;positive,negative=prompt_points(mask)
        proposals.append(dict(object_id=obj['objectId'],scene_id=obj['sceneId'],frame_id='f000009',source_frame_index=9,
            rgb_path=f"inference/rgb/scene{obj['sceneId']}/000009.png",rgb_sha256=seed['rgb']['sha256'],
            review_mask_path=mask_relative,authorized_mask_sha256=seed['onboardingMask']['sha256'],positive=positive,negative=negative,
            proposal_source='Authorized onboarding frame9 visible-mask interior maximum; negative background corner; must be visually reviewed',reviewed=False))
    write_json(output/'onboarding-review/proposals.json',proposals)
    evidence=review_sheets(output,proposals)
    if not reviewed_by:
        print(json.dumps(dict(status='visual_review_required',reviewEvidenceSHA256=evidence,sheets=str(output/'onboarding-review'),models=len(models),uniqueRGBFrames=len(linked))),flush=True)
        return dict(status='visual_review_required',reviewEvidenceSHA256=evidence)
    if review_evidence_sha256!=evidence:raise ValueError('Review acknowledgement does not bind the actual proposed sheets/points')
    entries=[]
    for obj,proposal in zip(objects,proposals):
        seed=obj['onboarding'][-1];clip=build_clip(obj,seed,reviewed_by)
        selection=wc.Selection('f000009',tuple(map(tuple,proposal['positive'])),tuple(map(tuple,proposal['negative'])),True,reviewed_by)
        clip=wc.ClipSpec.from_dict({**clip.to_dict(),'onboarding':wc.Onboarding(('f000009',),selection).to_dict()})
        relative,audit=models[obj['objectId']]
        bundle=dict(schema='workbench_metric_bundle_v1',family=FAMILY,glb_path=relative,glb_sha256=audit['glbSHA256'],obj_id=obj['objectId'],
            principal_point_origin='integer_centers',seed_pose_m=source_pose(seed),seed_label='onboarding:original HB frame9 authorized one-shot known pose; no reuse after loss',
            seed_frame_id='f000009',metric_provenance='dataset_metric_cad')
        body=dict(case_id=clip.clip_id+'-f000009',family=FAMILY,rgb_path=f"rgb/scene{obj['sceneId']}/000009.png",rgb_sha256=seed['rgb']['sha256'],
            mask_path=None,mask_sha256=None,segmentation=dict(producer='PENDING actual observed SAM2 output',manifest_sha256=None),
            K=seed['calibration']['cam_K'],principal_point_origin='integer_centers',initial_pose_m=source_pose(seed),
            initialization_source='onboarding:original frame9 single known pose',glb_path=relative,glb_sha256=audit['glbSHA256'],obj_id=obj['objectId'],metric_provenance='dataset_metric_cad')
        template=dict(schema='offline_sam_observed_case_template_v1',status='blocked_until_actual_SAM_mask',cases=[body],
            instruction='Do not use annotation masks here. After actual native SAM segmentation, run native gate case on f000009 to bind its observed mask/producer manifest.')
        validate_case_template(template,output/'inference')
        write_json(output/'inference/clips'/f'{clip.clip_id}.json',clip.to_dict())
        write_json(output/'inference/metric-bundles'/f'{clip.clip_id}.json',bundle)
        write_json(output/'inference/case-templates'/f'{clip.clip_id}.json',template)
        MetricBundle.from_file(output/'inference/metric-bundles'/f'{clip.clip_id}.json',root=output/'inference')
        wc.check_onboarding(clip,clip.onboarding,clip.scoring_frame_ids)
        entries.append(dict(object_id=obj['objectId'],scene_id=obj['sceneId'],clip_id=clip.clip_id,clip_path=f'clips/{clip.clip_id}.json',
            bundle_path=f'metric-bundles/{clip.clip_id}.json',case_template_path=f'case-templates/{clip.clip_id}.json',
            model=relative,model_sha256=audit['glbSHA256'],input_frames=301,scored_frames=300,seed_source_frame=9,scored_source_frames=[10,309]))
    index=dict(schema='offline_sam_pose_dataset_v1',dataset_id=DATASET_ID,dataset_revision=REVISION,license='CC0-1.0',source_release=SOURCE_RELEASE,
        evidence_role='real recorded dataset-reference evaluation; annotation-assisted reviewed onboarding, not independent human labels',
        source_inference_sha256=digest(source/'inference.json'),objects=entries,profiles=PROFILES,
        source_clock='frame_index',physical_fps=None,intrinsics_status='dataset_reference',native_frame_size=[640,480],
        runtime_mask_source='actual observed SAM2 video masks, not evaluator masks',one_shot_seed='authorized original frame9 known pose',
        review=dict(reviewed_by=reviewed_by,review_evidence_sha256=evidence,independent_human_annotations=False),
        resources=linked,private_weights_included=False,model_execution_performed=False)
    write_json(output/'inference/index.json',index)
    write_json(output/'inference/profiles.json',dict(schema='offline_sam_pose_profiles_v1',profiles=PROFILES,
        rule='Ordered source10 onward, source9 is the only onboarding frame; subset choice never uses scored labels',full_dataset_objects=12,full_scored_object_frames=3600))
    # Evaluation is a distinct phase, after inference-only clip/bundle/index files
    # are frozen. No field from this manifest selected a prompt or input frame.
    if digest(source/'evaluator.json')!=SOURCE_EVALUATOR_SHA256:raise ValueError('Source evaluator differs from the pinned full12-object HB annotation manifest')
    evaluator=json.loads((source/'evaluator.json').read_text());by_id={row['objectId']:row for row in evaluator['objects']}
    evaluation=dict(schema='offline_sam_pose_evaluator_v1',inference_index_sha256=digest(output/'inference/index.json'),
        label_source='Original BOP HB dataset references, not human-created new labels',dataset_revision=REVISION,objects=[])
    for obj,entry in zip(objects,entries):
        rows=[]
        for record in by_id[obj['objectId']]['frames']:
            rows.append(dict(frame_id=f"f{record['frameId']:06d}",source_frame_index=record['frameId'],
                reference_camera_from_object_m=source_pose(record),reference_visibility=record['evaluationInfo']['visib_fract'],
                reference_depth=dict(source_member=record['depth']['path'],sha256=record['depth']['sha256']),
                reference_visible_mask=dict(source_member=record['evaluationMask']['path'],sha256=record['evaluationMask']['sha256'])))
        evaluation['objects'].append(dict(object_id=obj['objectId'],clip_id=entry['clip_id'],frames=rows))
    write_json(output/'evaluator/evaluator.json',evaluation)
    lock=dict(schema='offline_sam_pose_lock_v1',inference_index_sha256=digest(output/'inference/index.json'),evaluator_sha256=digest(output/'evaluator/evaluator.json'),
        review_evidence_sha256=evidence,source_release=SOURCE_RELEASE,source_inference_sha256=digest(source/'inference.json'),source_evaluator_sha256=digest(source/'evaluator.json'),
        inference_files=[dict(path=path.relative_to(output/'inference').as_posix(),bytes=path.stat().st_size,sha256=digest(path)) for path in sorted((output/'inference').rglob('*')) if path.is_file()],
        model_audits=[dict(object_id=entry['object_id'],**models[entry['object_id']][1]) for entry in entries])
    write_json(output/'LOCK.json',lock)
    export_metadata(output,artifacts)
    print(json.dumps(dict(status='complete_no_models_executed',output=str(output),objects=12,scored_object_frames=3600,unique_rgb_frames=len(linked),inference_index_sha256=lock['inference_index_sha256'])),flush=True)
    return index


def export_metadata(output,artifacts):
    output,artifacts=Path(output),Path(artifacts)
    for path in sorted((output/'inference').rglob('*.json')):
        write_new(artifacts/'inference'/path.relative_to(output/'inference'),path.read_bytes())
    for path in sorted((output/'onboarding-review').glob('*.json')):
        write_new(artifacts/'onboarding-review'/path.name,path.read_bytes())
    for path in sorted((output/'onboarding-review').glob('prompt-review-*.png')):
        write_new(artifacts/'onboarding-review'/path.name,path.read_bytes())
    raw=(output/'evaluator/evaluator.json').read_bytes();compressed=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=compressed,mtime=0,compresslevel=9) as target:target.write(raw)
    write_new(artifacts/'evaluator/evaluator.json.gz',compressed.getvalue())
    write_new(artifacts/'LOCK.json',(output/'LOCK.json').read_bytes())


def lossless_frame(source,destination):
    from .workbench_geometry import array_sha256
    source,destination=Path(source),Path(destination)
    raw=source.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:rgb=np.asarray(image.convert('RGB')).copy()
    if destination.exists():encoded=destination.read_bytes()
    else:
        buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,'PNG',compress_level=6,optimize=False);encoded=buffer.getvalue();write_new(destination,encoded)
    with Image.open(io.BytesIO(encoded)) as image:decoded=np.asarray(image.convert('RGB'))
    if not np.array_equal(rgb,decoded):raise ValueError('Lossless runtime PNG changed decoded uint8 RGB')
    return dict(original_encoded_sha256=hashlib.sha256(raw).hexdigest(),runtime_encoded_sha256=hashlib.sha256(encoded).hexdigest(),
        original_encoded_bytes=len(raw),runtime_encoded_bytes=len(encoded),decoded_rgb_array_sha256=array_sha256(rgb),
        exact_uint8_rgb_equal=True,width=int(rgb.shape[1]),height=int(rgb.shape[0]),
        transformation='PNG lossless re-encoding only; no crop, resize, colour transform, geometric transform or frame renumbering')


def register(output,workspace,*,profile='smoke',object_id=20,lossless_rgb=False):
    output=Path(output).resolve();index=json.loads((output/'inference/index.json').read_text());ws=Workspace(workspace)
    item=next(row for row in index['objects'] if row['object_id']==object_id)
    base=wc.ClipSpec.from_dict(json.loads((output/'inference'/item['clip_path']).read_text()));count=PROFILES[profile]
    frames=base.frames[:count+1];media_root=output/'inference';root_id=ROOT_ID;proof=None
    if lossless_rgb:
        media_root=output/'runtime-lossless';root_id='offline_hb_lossless';runtime_frames=[];mapping=[]
        for frame in frames:
            relative=base.media.path+'/'+frame.image_path;source=output/'inference'/relative
            if digest(source)!=frame.image_sha256:raise ValueError('Original source RGB changed before lossless re-encoding')
            audit=lossless_frame(source,media_root/relative)
            runtime_frames.append(wc.FrameSpec(frame.frame_id,frame.source_frame_index,None,None,frame.image_path,audit['runtime_encoded_sha256']))
            mapping.append(dict(frame_id=frame.frame_id,source_frame_index=frame.source_frame_index,path=relative,**audit))
        frames=tuple(runtime_frames);size=sum(row['runtime_encoded_bytes'] for row in mapping)
        proof_path=media_root/'proofs'/f'{base.clip_id}-{profile}.json';write_json(proof_path,dict(schema='offline_hb_lossless_rgb_v1',frames=mapping,physical_fps=None,source_ids_preserved=True,calibration_unchanged=True))
        proof=dict(path=str(proof_path),sha256=digest(proof_path),frames=len(mapping),exact_uint8_rgb_equal=True)
    else:size=sum(index['resources'][base.media.path+'/'+frame.image_path]['bytes'] for frame in frames)
    body=base.to_dict();body.update(clip_id=base.clip_id+'-'+profile,frames=[frame.to_dict() for frame in frames],
        scoring_frame_ids=[frame.frame_id for frame in frames[1:]],media=wc.MediaRef(root_id,base.media.path,None,None).to_dict(),
        provenance={**base.provenance,'profile':profile,'counted_input_bytes':size,'copied':False,'lossless_runtime_rgb':proof})
    clip=wc.ClipSpec.from_dict(body)
    if not ws.clip_path(clip.clip_id).exists():ws.require_budget(extra_inputs=size)
    ws.init({root_id:str(media_root)});ws.save_clip(clip)
    bundle=json.loads((output/'inference'/item['bundle_path']).read_text());bundle['glb_path']=str(output/'inference'/bundle['glb_path'])
    bundle_path=ws.path/'metric-bundles'/f'{clip.clip_id}.json';write_json(bundle_path,bundle);MetricBundle.from_file(bundle_path)
    reader=FrameReader(ws,clip)
    try:
        reader.verify_frames([frame.frame_id for frame in frames]);reader.read(frames[0]);reader.read(frames[-1])
    finally:reader.close()
    result=dict(status='registered_no_models_executed',workspace=str(ws.path),clip_id=clip.clip_id,bundle_path=str(bundle_path),profile=profile,
        native_counted_input_bytes=size,frames=len(frames),scored_frames=count,first_native_frame='f000009',first_scored_frame='f000010')
    if proof:result['lossless_rgb_proof']=proof
    print(json.dumps(result),flush=True);return result


def verify(output):
    output=Path(output);lock=json.loads((output/'LOCK.json').read_text())
    for row in lock['inference_files']:
        path=output/'inference'/row['path']
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Inference resource changed: '+row['path'])
    if digest(output/'evaluator/evaluator.json')!=lock['evaluator_sha256']:raise ValueError('Evaluator manifest changed')
    index=json.loads((output/'inference/index.json').read_text())
    for entry in index['objects']:
        clip=wc.ClipSpec.from_dict(json.loads((output/'inference'/entry['clip_path']).read_text()));assertlen=len(clip.frames)
        if assertlen!=301 or len(clip.scoring_frame_ids)!=300 or clip.frames[0].source_frame_index!=9:raise ValueError('Clip protocol differs')
        MetricBundle.from_file(output/'inference'/entry['bundle_path'],root=output/'inference')
        validate_case_template(json.loads((output/'inference'/entry['case_template_path']).read_text()),output/'inference')
    print('Verified complete12-object offline SAM/pose input pack, native clip/bundle schemas and separate evaluator hashes',flush=True)


def package(output,artifacts,archive_path):
    """Publish-sized GLB/metadata adapter; reuse the existing full real-media release."""
    output,artifacts,archive_path=Path(output),Path(artifacts),Path(archive_path)
    if archive_path.exists():raise ValueError('Retain existing release archive or choose a fresh name')
    verify(output)
    files={}
    for path in sorted((output/'inference/models').glob('*.glb')):files['inference/models/'+path.name]=path
    for path in sorted((output/'model-audit').glob('*.json')):files['model-audit/'+path.name]=path
    for path in sorted(artifacts.rglob('*')):
        if path.is_file() and path.name!='release-asset.json':files['metadata/'+path.relative_to(artifacts).as_posix()]=path
    files['code/tracking_offline_dataset.py']=Path(__file__)
    if len(list((output/'inference/models').glob('*.glb')))!=12:raise ValueError('Release asset must include all12 metric models')
    inventory=[dict(path=name,bytes=path.stat().st_size,sha256=digest(path)) for name,path in sorted(files.items())]
    manifest=canonical(dict(schema='offline_sam_pose_asset_v1',dataset_id=DATASET_ID,license='CC0-1.0',
        reference_media_release=SOURCE_RELEASE,raw_RGB_included=False,scored_mask_or_depth_pixels_included=False,
        private_weights_included=False,models=12,scored_object_frames=3600,files=inventory,
        materialization='Acquire/reuse the existing full HB release, then run tracking_offline_dataset prepare. This asset adds metric GLBs and full native clip/bundle/annotation metadata without duplicating RGB/depth/mask media.'))
    archive_path.parent.mkdir(parents=True,exist_ok=True);temporary=archive_path.with_name(archive_path.name+'.part')
    if temporary.exists():raise ValueError('Unfinished archive exists; inspect before retry')
    with temporary.open('wb') as target,gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0,compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
            for name,path in sorted(files.items()):
                info=tarfile.TarInfo(DATASET_ID+'/'+name);info.size=path.stat().st_size;info.mode=0o644;info.mtime=info.uid=info.gid=0;info.uname=info.gname=''
                with path.open('rb') as source:archive.addfile(info,source)
            info=tarfile.TarInfo(DATASET_ID+'/ASSET-MANIFEST.json');info.size=len(manifest);info.mode=0o644;info.mtime=info.uid=info.gid=0;archive.addfile(info,io.BytesIO(manifest))
    temporary.replace(archive_path)
    receipt=dict(schema='offline_sam_pose_release_asset_v1',filename=archive_path.name,bytes=archive_path.stat().st_size,sha256=digest(archive_path),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),models=12,scored_object_frames=3600,existing_full_real_media_release=SOURCE_RELEASE,
        raw_RGB_duplicated=False,private_weights_included=False,publication_status='prepared_locally_not_uploaded')
    write_json(archive_path.with_suffix(archive_path.suffix+'.json'),receipt);write_new(archive_path.with_suffix(archive_path.suffix+'.manifest.json'),manifest)
    write_json(artifacts/'release-asset.json',receipt)
    print(json.dumps(receipt),flush=True);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--artifacts',type=Path,default=DEFAULT_ARTIFACTS);p.add_argument('--reviewed-by');p.add_argument('--review-evidence-sha256')
    p=sub.add_parser('register');p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--profile',choices=PROFILES,default='smoke');p.add_argument('--object-id',type=int,default=20);p.add_argument('--lossless-rgb',action='store_true')
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p=sub.add_parser('package');p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--artifacts',type=Path,default=DEFAULT_ARTIFACTS);p.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source,args.output,args.artifacts,reviewed_by=args.reviewed_by,review_evidence_sha256=args.review_evidence_sha256)
    elif args.action=='register':register(args.output,args.workspace,profile=args.profile,object_id=args.object_id,lossless_rgb=args.lossless_rgb)
    elif args.action=='package':package(args.output,args.artifacts,args.archive)
    else:verify(args.output)


if __name__=='__main__':main()
