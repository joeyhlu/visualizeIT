"""Generate descriptor views from the scanned model's photographic texture.

Model-only references never read evaluation footage, poses or masks.
"""
from io import BytesIO
import json
import struct
from itertools import product
import numpy as np
from PIL import Image
from .vision import cv2
from .model import Camera,unit
from .renderer import render,ray_triangle
from .performance_experiment import compact_bank


def texture(path):
    raw=path.read_bytes();size=struct.unpack_from('<I',raw,12)[0]
    document=json.loads(raw[20:20+size]);binary=raw[28+size:]
    materials=document.get('materials',[])
    if len(materials)!=1 or 'baseColorTexture' not in materials[0].get('pbrMetallicRoughness',{}):return None
    index=materials[0]['pbrMetallicRoughness']['baseColorTexture']['index']
    source=document['textures'][index]['source'];view=document['bufferViews'][document['images'][source]['bufferView']]
    begin=view.get('byteOffset',0)
    image=np.array(Image.open(BytesIO(binary[begin:begin+view['byteLength']])).convert('RGB'))
    # glTF images use a top-left origin; our OBJ renderer samples bottom-left UVs.
    return image[::-1].copy()


def build(root,mesh,path):
    cache=root/f'{mesh.name}-model-references-v1.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:return data['points'].copy(),data['descriptors'].copy()
    image=texture(path)
    if image is None:return np.empty((0,3)),np.empty((0,32),np.uint8)
    centre=(mesh.positions.min(axis=0)+mesh.positions.max(axis=0))/2
    radius=np.linalg.norm(mesh.positions-centre,axis=1).max();distance=radius*300/120
    orb=cv2.ORB_create(nfeatures=600,edgeThreshold=15,fastThreshold=12)
    points=[];descriptors=[]
    directions=[np.array(v,float) for v in product((-1,0,1),repeat=3) if any(v)]
    for index,direction in enumerate(directions):
        eye=centre+unit(direction)*distance;forward=unit(centre-eye)
        up=[0,1,0] if abs(forward[1])<.9 else [0,0,1]
        right=unit(np.cross(up,forward));down=-unit(np.cross(forward,right))
        camera=Camera(320,320,300,300,160,160,eye,np.array([right,down,forward]))
        frame=render(mesh,camera,texture_image=image,shade=False)
        mask=cv2.erode((frame.triangle>=0).astype(np.uint8)*255,np.ones((3,3),np.uint8))
        features,desc=orb.detectAndCompute(cv2.cvtColor(frame.color,cv2.COLOR_RGB2GRAY),mask)
        if desc is not None:
            for feature,descriptor in zip(features,desc):
                x,y=feature.pt;face=frame.triangle[int(y),int(x)]
                if face<0:continue
                ids=mesh.triangles[face];ray=camera.rays([x,y])
                hit=ray_triangle(camera.eye,ray,mesh.positions[ids],mesh.uv[ids])
                if hit is not None:points.append(camera.eye+ray*hit[0]);descriptors.append(descriptor)
        print(f'{mesh.name}: model-only reference view {index+1}/26',flush=True)
    if not points:return np.empty((0,3)),np.empty((0,32),np.uint8)
    points,descriptors=compact_bank(np.array(points),np.array(descriptors,np.uint8),maximum=2500)
    np.savez_compressed(cache,points=points,descriptors=descriptors)
    return points,descriptors
