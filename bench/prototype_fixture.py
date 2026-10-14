"""Generate one explicitly synthetic control for an offline runnable smoke demo.

It cannot count as a new real object, human accuracy, live inference, or phone QA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from .model import Mesh
from .prototype_benchmark import cv_render_camera, write_json, digest
from .renderer import render


def generate(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh synthetic control directory')
    if shutil.disk_usage(output.parent).free < 1024**3 + 30 * 1024**2:
        raise ValueError('Synthetic control would cross the 1 GiB reserve')
    output.mkdir()
    # Source metre geometry, outward winding, unique texture atlas cells per face.
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float) * [.085,.065,.06]
    faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
    points, normals, uv, triangles = [], [], [], []
    for face_id, indices in enumerate(faces):
        quad = corners[indices]
        normal = np.cross(quad[1]-quad[0], quad[2]-quad[0]); normal /= np.linalg.norm(normal)
        offset = len(points); points.extend(quad); normals.extend([normal]*4)
        x,y = face_id % 3, face_id // 3
        uv.extend([[(x+.02)/3,(y+.02)/2],[(x+.98)/3,(y+.02)/2],
                   [(x+.98)/3,(y+.98)/2],[(x+.02)/3,(y+.98)/2]])
        triangles.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
    points, normals, uv, triangles = map(np.asarray, (points,normals,uv,triangles))
    mesh = Mesh('Synthetic textured box', points, normals, uv, triangles)
    ply = ['ply','format ascii 1.0','element vertex 24','property float x','property float y',
           'property float z','property float texture_u','property float texture_v',
           'element face 12','property list uchar int vertex_indices','end_header']
    ply += [' '.join(map(str,[*(p*1000),*q])) for p,q in zip(points,uv)]
    ply += ['3 '+' '.join(map(str,tri)) for tri in triangles]
    model = output/'object.ply'; model.write_text('\n'.join(ply)+'\n',encoding='ascii')
    model_ref = dict(path='object.ply',sha256=digest(model))
    rng = np.random.default_rng(771)
    texture = np.asarray(Image.fromarray(rng.integers(20,236,(96,144,3),dtype=np.uint8))
                         .resize((576,384), Image.Resampling.NEAREST))
    k = np.array([[450.,0,159.5],[0,450.,119.5],[0,0,1.]])
    inference = dict(schemaVersion=1, benchmarkId='synthetic-box-control-v1',
        selectionDigest=hashlib.sha256(b'synthetic-box-control-v1').hexdigest(),
        dataset='Synthetic geometry control', evidenceKind='synthetic_control', objects=[])
    evaluator = {key:value for key,value in inference.items() if key!='objects'}
    evaluator['objects']=[]
    common = dict(objectId=1,name='Synthetic textured box',model=model_ref,
                  referenceId='synthetic-box-control')
    obj = dict(**common,onboarding=[],frames=[])
    eval_obj = dict(**common,frames=[])
    for index in range(120):
        yaw = np.deg2rad(24+24*index/119)
        pitch = np.deg2rad(-12)
        ry=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        r=ry@rx; t=np.array([0.,0.,.55])
        calibration=dict(cam_K=k.ravel().tolist(),depth_scale=1.)
        raster=render(mesh,cv_render_camera(r,t,calibration,np.eye(4),320,240),
                      texture_image=texture,shade=False)
        rgb=output/f'{index:03d}-rgb.png'; depth=output/f'{index:03d}-depth.png'; mask=output/f'{index:03d}-mask.png'
        Image.fromarray(raster.color).save(rgb)
        valid=np.isfinite(raster.depth)
        depth_mm=np.where(valid,raster.depth,0)*1000
        Image.fromarray(np.rint(depth_mm).astype(np.uint16)).save(depth)
        Image.fromarray((valid*255).astype(np.uint8)).save(mask)
        ref=lambda path:dict(path=path.name,sha256=digest(path))
        record=dict(frameId=index,logicalIndex=index,timestampKind='frame_index_no_fps',
                    timestampSeconds=None,calibration=calibration,rgb=ref(rgb))
        pose=dict(cam_R_m2c=r.ravel().tolist(),cam_t_m2c=(t*1000).tolist())
        if index<10:obj['onboarding'].append(dict(**record,depth=ref(depth),evaluationPose=pose))
        else:
            obj['frames'].append(record)
            eval_obj['frames'].append(dict(**record,depth=ref(depth),evaluationPose=pose,
                evaluationInfo=dict(visib_fract=1.),evaluationMask=ref(mask)))
    inference['objects'].append(obj); evaluator['objects'].append(eval_obj)
    write_json(output/'evaluator.json',evaluator)
    inference['evaluatorManifestSHA256']=digest(output/'evaluator.json')
    write_json(output/'inference.json',inference)
    write_json(output/'selection.json',dict(status='synthetic_control',realUnseenObjects=0,
        benchmarkId=inference['benchmarkId'],note='Synthetic smoke only, not external-data evaluation'))
    print('Generated one synthetic control; real unseen objects: 0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output)
