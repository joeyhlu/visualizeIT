"""Deterministic 3-D rigid-object stress corpus; every image is SYNTHETIC.

No downloaded images, scans, pretrained models or human labels are used. The
full media is regenerated into a local cache. A compact, complete pose/visibility
annotation pack and representative contact sheet can be committed separately.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import io
from pathlib import Path
import shutil
import time
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from .model import Mesh
from .prototype_benchmark import cv_render_camera, digest, write_json
from .renderer import render
from .vision import cv2

SCHEMA = 'visualizeit-synthetic-tracking/1'
SCENARIOS = ('smooth', 'fast_motion', 'motion_blur', 'illumination',
             'partial_occlusion', 'full_occlusion', 'exit_reentry', 'low_texture')
SPLITS = ('train', 'dev', 'test')
SPLIT_SEEDS = (27181, 31415, 92653)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1]/'.cache/tracking-benchmark/synthetic-v1'
WIDTH, HEIGHT = 320, 240
K = np.array([[430., 0., 159.5], [0., 430., 119.5], [0., 0., 1.]])


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def config(frame_count=200, sequence_limit=24):
    if not isinstance(frame_count, int) or frame_count < 20 or frame_count > 1000:
        raise ValueError('frame_count must be an integer between 20 and 1000')
    if not isinstance(sequence_limit,int) or not 1 <= sequence_limit <= 24:
        raise ValueError('sequence_limit must be an integer between 1 and 24')
    return dict(schema=SCHEMA, imageSize=[WIDTH, HEIGHT], framesPerSequence=frame_count,
                onboardingFrames=10, sequenceLimit=sequence_limit, scenarios=list(SCENARIOS), splits=list(SPLITS),
                splitSeeds=list(SPLIT_SEEDS), generator='bench/tracking_synthetic_data.py',
                license='CC0-1.0', evidenceKind='synthetic_stress')


def mesh_for(seed):
    """Nonplanar convex cuboid/tapered/sheared geometries; source coordinates m."""
    rng = np.random.default_rng(seed)
    dims = rng.uniform([.065, .045, .045], [.095, .075, .075])
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)*dims
    kind = seed % 3
    if kind == 1:  # Taper top in x and shear into an asymmetric prism.
        top = corners[:, 1] < 0
        corners[top, 0] = .72*corners[top, 0]+.014
    elif kind == 2:
        corners[:, 0] += corners[:, 1]*.30
        corners[:, 2] += corners[:, 1]*.16
    faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
    points, normals, uv, triangles = [], [], [], []
    for face_id, ids in enumerate(faces):
        quad = corners[ids]
        normal = np.cross(quad[1]-quad[0], quad[2]-quad[0]); normal /= np.linalg.norm(normal)
        offset = len(points)
        for iy in range(3):
            v = iy/2
            for ix in range(3):
                u = ix/2
                points.append((1-v)*((1-u)*quad[0]+u*quad[1])+v*((1-u)*quad[3]+u*quad[2]))
                normals.append(normal)
                uv.append([(face_id%3+.03+.94*u)/3, (face_id//3+.03+.94*v)/2])
        for iy in range(2):
            for ix in range(2):
                a = offset+iy*3+ix; b=a+1; d=a+3; c=d+1
                triangles.extend([[a,b,c],[a,c,d]])
    return Mesh(f'Synthetic geometry {kind}', np.asarray(points), np.asarray(normals),
                np.asarray(uv), np.asarray(triangles)), dict(kind=('cuboid','tapered_prism','sheared_prism')[kind],
                                                          halfExtentsMetres=dims.tolist())


def write_ply(path, mesh):
    rows = ['ply','format ascii 1.0',f'element vertex {len(mesh.positions)}',
            'property float x','property float y','property float z',
            'property float texture_u','property float texture_v',
            f'element face {len(mesh.triangles)}','property list uchar int vertex_indices','end_header']
    rows += [' '.join(format(float(v), '.12g') for v in [*(p*1000), *q])
             for p,q in zip(mesh.positions,mesh.uv)]
    rows += ['3 '+' '.join(map(str,face)) for face in mesh.triangles]
    path.write_text('\n'.join(rows)+'\n', encoding='ascii')


def texture_for(seed, low=False):
    rng = np.random.default_rng(seed)
    if low:
        texture = np.full((256,384,3), 155, dtype=np.uint8)
        # Intentional repeated, weak texture, not different texture at test time.
        texture[::32, :, :] = 165; texture[:, ::32, :] = 145
    else:
        cells = rng.integers(15, 241, (64,96,3), dtype=np.uint8)
        texture = cv2.resize(cells, (576,384), interpolation=cv2.INTER_NEAREST)
        for _ in range(120):
            x,y = map(int, rng.integers([0,0],[576,384]))
            cv2.circle(texture,(x,y),int(rng.integers(1,6)),tuple(map(int,rng.integers(0,256,3))),-1)
    return texture


def background_for(seed):
    rng = np.random.default_rng(seed)
    background = np.full((HEIGHT,WIDTH,3), [52,63,71], dtype=np.uint8)
    for _ in range(32):
        x,y = map(int,rng.integers([0,0],[WIDTH,HEIGHT]))
        color=tuple(map(int,rng.integers(20,120,3)))
        cv2.rectangle(background,(x,y),(min(WIDTH-1,x+int(rng.integers(5,45))),
                                      min(HEIGHT-1,y+int(rng.integers(5,35)))),color,-1)
    return background


def pose_at(index, count, scenario, seed):
    """Independent deterministic CV X-right/Y-down/Z-forward pose trajectory."""
    u = max(0., (index-10)/max(1,count-11))
    phase = (seed%17)*.1
    yaw = np.deg2rad(29+14*np.sin(2*np.pi*u+phase))
    pitch = np.deg2rad(-19+6*np.sin(3*np.pi*u))
    roll = np.deg2rad(3*np.sin(2*np.pi*u))
    tx,ty,z = .024*np.sin(2*np.pi*u), .014*np.sin(4*np.pi*u), .58+.05*np.sin(2*np.pi*u)
    if scenario == 'fast_motion' and index >= 10:
        tx += .065*np.sin(24*np.pi*u)
        yaw += np.deg2rad(14*np.sin(20*np.pi*u))
    if scenario == 'exit_reentry' and index >= 10:
        if .32 <= u < .62:
            # A complete disappearance and return without a pose reset.
            tx += .53*np.sin(np.pi*(u-.32)/.30)
        z += .06*np.sin(4*np.pi*u)
    rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
    return rz@ry@rx, np.array([tx,ty,z])


def occluder(index,count,scenario):
    if index < 10:
        return None
    u=(index-10)/max(1,count-11)
    if scenario == 'partial_occlusion' and .27 <= u <= .74:
        x=int(115+48*np.sin(7*u))
        return (x,50,min(WIDTH,x+65),210)
    if scenario == 'full_occlusion' and (.35 <= u <= .46 or .68 <= u <= .74):
        return (0,0,WIDTH,HEIGHT)
    return None


def generate(output=DEFAULT_OUTPUT, *, frame_count=200, artifact_dir=None, sequence_limit=24):
    output=Path(output)
    configuration=config(frame_count, sequence_limit)
    selection_digest=hashlib.sha256(canonical(configuration)).hexdigest()
    if output.exists() and (output/'COMPLETE.json').exists():
        previous=json.loads((output/'COMPLETE.json').read_text())
        if previous.get('selectionDigest') != selection_digest:
            raise ValueError('Existing complete dataset has a different configuration')
        verify_dataset(output)
        if artifact_dir:
            export_cached_artifacts(output,artifact_dir)
        print('Using existing complete synthetic corpus',output,flush=True)
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh output directory; incomplete output is never silently reused')
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free < 2*1024**3:
        raise ValueError('Need at least 2 GiB free for corpus and reserve')
    common=dict(schemaVersion=1,benchmarkId='visualizeit-synthetic-tracking-v1',
                selectionDigest=selection_digest,dataset='VisualizeIT synthetic stress v1',
                evidenceKind='synthetic_stress')
    inference=dict(**common,objects=[]); evaluator=dict(**common,objects=[])
    sequences=[]; started=time.perf_counter()
    for split_index,(split,split_seed) in enumerate(zip(SPLITS,SPLIT_SEEDS)):
        for scenario_index,scenario in enumerate(SCENARIOS):
            seq_id=split_index*len(SCENARIOS)+scenario_index+1
            if seq_id > sequence_limit:
                continue
            seed=split_seed+scenario_index*101
            sequence=f'{split}-{scenario}-{seed}'
            folder=output/sequence; folder.mkdir()
            mesh,geometry=mesh_for(seed)
            write_ply(folder/'object.ply',mesh)
            model=dict(path=f'{sequence}/object.ply',sha256=digest(folder/'object.ply'))
            texture=texture_for(seed,scenario=='low_texture'); background=background_for(seed+2)
            identity=dict(objectId=seq_id,name=f'SYNTHETIC {sequence}',model=model,
                          referenceId=f'synthetic-v1/{sequence}',sequenceId=sequence,split=split,
                          stressTags=[scenario])
            obj=dict(**identity,onboarding=[],frames=[]); eval_obj=dict(**identity,frames=[])
            visible_counts=[]
            for index in range(frame_count):
                rotation,translation=pose_at(index,frame_count,scenario,seed)
                calibration=dict(cam_K=K.ravel().tolist(),depth_scale=1.)
                camera=cv_render_camera(rotation,translation,calibration,np.eye(4),WIDTH,HEIGHT)
                raster=render(mesh,camera,texture_image=texture,shade=False)
                full_mask=np.isfinite(raster.depth)
                rgb=background.copy(); rgb[full_mask]=raster.color[full_mask]
                visible_mask=full_mask.copy()
                rect=occluder(index,frame_count,scenario)
                if rect is not None:
                    x0,y0,x1,y1=rect
                    rgb[y0:y1,x0:x1]=[83,104,113]
                    visible_mask[y0:y1,x0:x1]=False
                if scenario=='motion_blur' and index>=10:
                    kernel=np.zeros((11,11),float); kernel[5]=1/11
                    rgb=cv2.filter2D(rgb,-1,kernel)
                if scenario=='illumination' and index>=10:
                    factor=.48+.65*(1+np.sin(index*.15))/2
                    rgb=np.clip(rgb.astype(float)*factor+12*np.sin(index*.11),0,255).astype(np.uint8)
                depth=np.where(visible_mask,raster.depth,0)
                depth_mm=np.rint(depth*1000).astype(np.uint16)
                rgb_path=folder/f'{index:04d}-rgb.png'; depth_path=folder/f'{index:04d}-depth.png'
                mask_path=folder/f'{index:04d}-mask.png'
                Image.fromarray(rgb).save(rgb_path,compress_level=1)
                Image.fromarray(depth_mm).save(depth_path,compress_level=1)
                Image.fromarray(visible_mask.astype(np.uint8)*255).save(mask_path,compress_level=1)
                ref=lambda path:dict(path=path.relative_to(output).as_posix(),sha256=digest(path))
                record=dict(frameId=index,logicalIndex=index,timestampKind='physical',
                            timestampSeconds=index/30,calibration=calibration,rgb=ref(rgb_path))
                pose=dict(cam_R_m2c=rotation.ravel().tolist(),cam_t_m2c=(translation*1000).tolist())
                # Convex silhouettes: projected hull gives a deterministic geometric
                # in-frame visibility estimate, separate from pixel-exact occlusion.
                camera_points=mesh.positions@rotation.T+translation
                projected=camera_points@K.T; projected=projected[:,:2]/projected[:,2,None]
                hull=cv2.convexHull(projected.astype(np.float32))
                full_area=max(float(cv2.contourArea(hull)),float(full_mask.sum()),1.)
                visibility=float(np.clip(visible_mask.sum()/full_area,0,1))
                visible_counts.append(visibility)
                if index<10:
                    obj['onboarding'].append(dict(**record,depth=ref(depth_path),onboardingMask=ref(mask_path),evaluationPose=pose))
                else:
                    obj['frames'].append(record)
                    eval_obj['frames'].append(dict(**record,depth=ref(depth_path),evaluationPose=pose,
                        evaluationInfo=dict(visib_fract=visibility,visibilityDefinition='visible_raster_pixels/projected_convex_hull_area',
                                            visiblePixels=int(visible_mask.sum()),inFrameUnoccludedPixels=int(full_mask.sum())),
                        evaluationMask=ref(mask_path),stress=dict(scenario=scenario,occluderXYXY=rect,
                            blurKernelPixels=11 if scenario=='motion_blur' else 0)))
            inference['objects'].append(obj); evaluator['objects'].append(eval_obj)
            sequences.append(dict(sequenceId=sequence,objectId=seq_id,split=split,seed=seed,
                scenario=scenario,geometry=geometry,modelSHA256=model['sha256'],frames=frame_count,onboardingFrames=10,
                scoredFrames=frame_count-10,framesBelow50PercentVisibility=sum(v<.5 for v in visible_counts[10:])))
            print(f'{sequence}: {frame_count} frames ({time.perf_counter()-started:.1f}s elapsed)',flush=True)
    write_json(output/'evaluator.json',evaluator)
    inference['evaluatorManifestSHA256']=digest(output/'evaluator.json')
    write_json(output/'inference.json',inference)
    write_json(output/'selection.json',dict(**configuration,selectionDigest=selection_digest,
        realUnseenObjects=0,sequences=sequences,selectionPolicy='Fixed analytic trajectories; no tracker output was used to select cases.'))
    complete=dict(schema=SCHEMA,selectionDigest=selection_digest,sequences=len(sequences),
        frames=sum(s['frames'] for s in sequences),scoredFrames=sum(s['scoredFrames'] for s in sequences),
        inferenceSHA256=digest(output/'inference.json'),evaluatorSHA256=digest(output/'evaluator.json'),
        diskBytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        generatedWallSeconds=time.perf_counter()-started)
    write_json(output/'COMPLETE.json',complete)
    if artifact_dir:
        export_cached_artifacts(output,artifact_dir)
    print(json.dumps(complete,indent=2),flush=True)
    return complete



def verify_dataset(output):
    """Verify sealed manifest identity and every referenced media/model byte."""
    output=Path(output)
    complete=json.loads((output/'COMPLETE.json').read_text())
    for key,name in (('inferenceSHA256','inference.json'),('evaluatorSHA256','evaluator.json')):
        if digest(output/name)!=complete[key]:
            raise ValueError('Sealed manifest hash mismatch: '+name)
    inference=json.loads((output/'inference.json').read_text())
    evaluator=json.loads((output/'evaluator.json').read_text())
    if not (inference['selectionDigest']==evaluator['selectionDigest']==complete['selectionDigest']):
        raise ValueError('Selection digest mismatch')
    if inference['evaluatorManifestSHA256']!=complete['evaluatorSHA256']:
        raise ValueError('Evaluator binding mismatch')
    refs={}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:
                if value['path'] in refs and refs[value['path']]!=value['sha256']:
                    raise ValueError('Conflicting media reference')
                refs[value['path']]=value['sha256']
            else:
                for item in value.values(): collect(item)
        elif isinstance(value,list):
            for item in value: collect(item)
    collect(inference); collect(evaluator)
    for name,expected in refs.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError('Resource escapes root')
        if digest(path)!=expected:
            raise ValueError('Media/model hash mismatch: '+name)
    return dict(verifiedResources=len(refs),sequences=len(inference['objects']),
                scoredFrames=sum(len(o['frames']) for o in inference['objects']))


def export_cached_artifacts(output, artifact_dir):
    """Re-export compact deliverables from a complete, verified local corpus."""
    output,artifact_dir=Path(output),Path(artifact_dir)
    artifact_dir.mkdir(parents=True,exist_ok=True)
    inference=json.loads((output/'inference.json').read_text())
    evaluator=json.loads((output/'evaluator.json').read_text())
    selection=json.loads((output/'selection.json').read_text())
    complete=json.loads((output/'COMPLETE.json').read_text())
    rows=[]; samples=[]
    eval_by_id={o['objectId']:o for o in evaluator['objects']}
    inventory={o['objectId']:o for o in selection['sequences']}
    for obj in inference['objects']:
        sequence=obj['sequenceId']; details=inventory[obj['objectId']]
        scenario=details['scenario']; frame_count=details['frames']
        current=obj['onboarding']+eval_by_id[obj['objectId']]['frames']
        sample_mid=int(10+.4*(frame_count-11)) if scenario=='full_occlusion' else frame_count//2
        for record in current:
            index=record['frameId']; pose=record['evaluationPose']
            rotation=np.asarray(pose['cam_R_m2c']).reshape(3,3)
            translation=np.asarray(pose['cam_t_m2c'])*.001
            # Match the original visibility definition, including onboarding.
            if index>=10:
                visibility=record['evaluationInfo']['visib_fract']
            else:
                mesh,_=mesh_for(details['seed'])
                with Image.open(output/record['onboardingMask']['path']) as image:
                    mask=np.asarray(image)>0
                camera_points=mesh.positions@rotation.T+translation
                projected=camera_points@K.T; projected=projected[:,:2]/projected[:,2,None]
                hull=cv2.convexHull(projected.astype(np.float32))
                visibility=float(np.clip(mask.sum()/max(float(cv2.contourArea(hull)),float(mask.sum()),1.),0,1))
            mask_ref=record['evaluationMask'] if index>=10 else record['onboardingMask']
            rows.append(dict(sequenceId=sequence,objectId=obj['objectId'],split=obj['split'],seed=details['seed'],
                frameId=index,phase='onboarding' if index<10 else 'evaluation',stress=scenario,
                cameraFromObject=np.column_stack([rotation,translation]).tolist(),K=K.tolist(),
                visibility=visibility,rgbSHA256=record['rgb']['sha256'],maskSHA256=mask_ref['sha256'],
                depthSHA256=record['depth']['sha256'],occluderXYXY=occluder(index,frame_count,scenario)))
            if index in (9,sample_mid,frame_count-2):
                with Image.open(output/record['rgb']['path']) as image:
                    samples.append((sequence,index,np.asarray(image.convert('RGB'))))
        model_output=artifact_dir/'synthetic-models'; model_output.mkdir(exist_ok=True)
        shutil.copyfile(output/obj['model']['path'],model_output/(sequence+'.ply'))
        details['modelSHA256']=obj['model']['sha256']
    (artifact_dir/'synthetic-annotations.jsonl.gz').write_bytes(gzip.compress(b''.join(canonical(row) for row in rows),compresslevel=9,mtime=0))
    configuration={key:selection[key] for key in config(complete['frames']//complete['sequences'],complete['sequences'])}
    write_json(artifact_dir/'synthetic-manifest.json',dict(**configuration,selectionDigest=complete['selectionDigest'],
        sequences=selection['sequences'],counts={k:complete[k] for k in ('sequences','frames','scoredFrames')},
        artifacts={'annotationsSHA256':digest(artifact_dir/'synthetic-annotations.jsonl.gz')},
        caveat='All media and labels are synthetic. Splits are procedural; no claim of real-world generalization.'))
    sheet=Image.new('RGB',(3*WIDTH,len(inference['objects'])*(HEIGHT+25)),(18,23,29)); draw=ImageDraw.Draw(sheet)
    for i,(sequence,index,rgb) in enumerate(samples):
        row,col=divmod(i,3); y=row*(HEIGHT+25)
        sheet.paste(Image.fromarray(rgb),(col*WIDTH,y+25))
        draw.text((col*WIDTH+5,y+6),f'{sequence} / {index}',fill='white')
    sheet.thumbnail((960,6360))
    sheet.save(artifact_dir/'synthetic-contact-sheet.jpg',quality=80,optimize=True)



def package_dataset(output, archive):
    """Package verified full RGB-D corpus with normalized tar/gzip metadata.

    Deterministic for identical sealed input bytes. The runtime measurement in
    COMPLETE.json remains factual; regenerating it can change archive identity.
    """
    output,archive=Path(output),Path(archive)
    if archive.resolve().is_relative_to(output.resolve()) or archive.is_symlink():
        raise ValueError('Release archive must be outside dataset root and not a symlink')
    verified=verify_dataset(output)
    archive.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    extras={
        'README.md': (root/'datasets/tracking-v1/synthetic-README.md').read_bytes(),
        'LICENSE.txt': (root/'datasets/tracking-v1/synthetic-LICENSE.txt').read_bytes(),
    }
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as compressed:
            with tarfile.open(mode='w',fileobj=compressed,format=tarfile.USTAR_FORMAT) as bundle:
                for path in sorted(output.rglob('*')):
                    if not path.is_file() or path.is_symlink():
                        if path.is_symlink(): raise ValueError('Symlinks are not allowed in a data release')
                        continue
                    name='synthetic-v1/'+path.relative_to(output).as_posix()
                    info=tarfile.TarInfo(name); info.size=path.stat().st_size
                    info.mtime=0; info.mode=0o644; info.uid=info.gid=0; info.uname=info.gname=''
                    with path.open('rb') as source: bundle.addfile(info,source)
                for name,payload in sorted(extras.items()):
                    info=tarfile.TarInfo('synthetic-v1/'+name); info.size=len(payload)
                    info.mtime=0; info.mode=0o644; info.uid=info.gid=0; info.uname=info.gname=''
                    bundle.addfile(info,io.BytesIO(payload))
    result=dict(archiveName=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,
                status='local_prepared_not_uploaded',**verified,
                scope='Full synthetic RGB/depth/mask media, models, manifests, README and CC0 license; not yet a GitHub release asset.')
    write_json(archive.with_suffix(archive.suffix+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--frames',type=int,default=200)
    parser.add_argument('--artifact-dir',type=Path)
    parser.add_argument('--archive',type=Path,help='Prepare full media release archive locally, never upload it')
    parser.add_argument('--sequences',type=int,default=24,choices=range(1,25))
    args=parser.parse_args(); generate(args.output,frame_count=args.frames,artifact_dir=args.artifact_dir,sequence_limit=args.sequences)
    if args.archive:
        print(json.dumps(package_dataset(args.output,args.archive),indent=2),flush=True)


if __name__=='__main__':
    main()
