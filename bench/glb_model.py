"""Read bounded uncompressed glTF binary geometry, retaining mesh surface UVs."""
import json
import struct
import numpy as np
from .model import Mesh


def read_glb(path,name):
    raw=path.read_bytes()
    if len(raw)>24*1024*1024 or len(raw)<20:raise ValueError('GLB size out of bounds.')
    magic,version,length=struct.unpack_from('<III',raw)
    if magic!=0x46546c67 or version!=2 or length!=len(raw):raise ValueError('Expected complete GLB 2.0.')
    offset=12;document=None;binary=None
    while offset<len(raw):
        size,kind=struct.unpack_from('<II',raw,offset);offset+=8
        payload=raw[offset:offset+size];offset+=size
        if len(payload)!=size:raise ValueError('Truncated GLB chunk.')
        if kind==0x4e4f534a:document=json.loads(payload)
        elif kind==0x004e4942:binary=payload
    if document is None or binary is None:raise ValueError('Embedded geometry required.')
    if any('uri' in b for b in document.get('buffers',[])):raise ValueError('External buffers unsupported.')
    def array(index,index_buffer=False):
        a=document['accessors'][index]
        if index_buffer and (a.get('type')!='SCALAR' or a.get('componentType') not in (5121,5123,5125)):
            raise ValueError('Triangle indices require unsigned SCALAR accessor.')
        if 'sparse' in a or a.get('normalized'):raise ValueError('Sparse/normalized geometry unsupported.')
        view=document['bufferViews'][a['bufferView']]
        if view.get('buffer',0)!=0:raise ValueError('Only embedded buffer zero supported.')
        dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
        components={'SCALAR':1,'VEC2':2,'VEC3':3}[a['type']]
        count=a['count'];stride=view.get('byteStride',components*dtype.itemsize)
        start=view.get('byteOffset',0)+a.get('byteOffset',0)
        # A triangle has three indices; its index count is not a vertex count.
        limit=900000 if index_buffer else 300000
        if type(count) is not int or count<1 or count>limit or start+(count-1)*stride+components*dtype.itemsize>len(binary):
            raise ValueError('Geometry accessor out of bounds.')
        return np.ndarray((count,components),dtype=dtype,buffer=binary,offset=start,strides=(stride,dtype.itemsize)).copy()
    positions=[];normals=[];uvs=[];faces=[];total=0;triangle_total=0
    def visit(index,parent,ancestors):
        nonlocal total,triangle_total
        if index in ancestors:raise ValueError('Cyclic GLB node graph.')
        node=document['nodes'][index]
        local=np.eye(4)
        if 'matrix' in node:local=np.array(node['matrix'],float).reshape(4,4).T
        else:
            x,y,z,w=node.get('rotation',[0,0,0,1])
            if abs(x*x+y*y+z*z+w*w-1)>1e-5:raise ValueError('Invalid node quaternion.')
            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            local[:3,:3]=rotation@np.diag(node.get('scale',[1,1,1]))
            local[:3,3]=node.get('translation',[0,0,0])
        transform=parent@local
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:3,:3]))<1e-15:raise ValueError('Invalid node transform.')
        if 'mesh' in node:
            for primitive in document['meshes'][node['mesh']]['primitives']:
                if primitive.get('mode',4)!=4 or 'extensions' in primitive:raise ValueError('Plain triangles required.')
                attributes=primitive['attributes'];p=array(attributes['POSITION']);n=array(attributes['NORMAL']);u=array(attributes['TEXCOORD_0'])
                f=array(primitive['indices'],index_buffer=True).ravel()
                if len(f)%3:raise ValueError('Incomplete triangles.')
                f=f.reshape(-1,3).astype(np.int64)
                if f.min()<0 or f.max()>=len(p):raise ValueError('Invalid triangle vertex.')
                if total+len(p)>300000 or triangle_total+len(f)>300000:
                    raise ValueError('Combined scene geometry exceeds vertex/triangle budget.')
                p=p@transform[:3,:3].T+transform[:3,3]
                n=n@np.linalg.inv(transform[:3,:3]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-15)
                if np.linalg.det(transform[:3,:3])<0:f=f[:,[0,2,1]]
                positions.append(p);normals.append(n);uvs.append(u);faces.append(f+total);total+=len(p);triangle_total+=len(f)
        for child in node.get('children',[]):visit(child,transform,ancestors|{index})
    for index in document['scenes'][document.get('scene',0)]['nodes']:visit(index,np.eye(4),set())
    if not positions:raise ValueError('Empty GLB geometry.')
    return Mesh(name,np.concatenate(positions),np.concatenate(normals),np.concatenate(uvs),np.concatenate(faces))
