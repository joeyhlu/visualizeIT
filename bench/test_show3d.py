import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from .show3d_experiment import reference_pose,choose_window
from .glb_model import read_glb


class Show3DTests(unittest.TestCase):
    def test_rig_relative_pose_and_mm_to_m_conversion(self):
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=[1000,0,0]
        calibration=dict(DistortionModel='PinholePlane',T_WorldFromCamera_by_index={'3':dict(T_WorldFromCamera=transform.tolist(),is_pose_valid=True,timestamp=2.)})
        pose=dict(R=rotation.tolist(),t=[[2000],[0],[1000]],confidence=.8,timestamp=2.)
        r,t=reference_pose(pose,calibration,3)
        np.testing.assert_allclose(r,np.eye(3));np.testing.assert_allclose(t,[0,-1,1])
        calibration['T_WorldFromCamera_by_index']['3']['is_pose_valid']=False
        self.assertIsNone(reference_pose(pose,calibration,3))
        pose.update(confidence=0,R=[],t=[])
        self.assertIsNone(reference_pose(pose,calibration,3))

    def test_60fps_window_does_not_substitute_30fps_timestamps(self):
        poses={str(i):dict(index=i,R=np.eye(3).tolist(),t=[[0],[0],[1000]],confidence=.8,timestamp=i/60) for i in range(270)}
        cameras={str(i):dict(T_WorldFromCamera=np.eye(4).tolist(),is_pose_valid=True,timestamp=i/60) for i in range(270)}
        calibration=dict(DistortionModel='PinholePlane',T_WorldFromCamera_by_index=cameras)
        poses['0']['confidence']=0
        start,_=choose_window(poses,calibration)
        self.assertEqual(start,1)
        for i in range(270):poses[str(i)]['timestamp']=i/30;cameras[str(i)]['timestamp']=i/30
        with self.assertRaises(ValueError):choose_window(poses,calibration)

    def test_binary_interleaving_and_node_scale_preserve_metric_geometry(self):
        vertices=np.array([[0,0,0,0,0,1,0,0],[1000,0,0,0,0,1,1,0],[0,1000,0,0,0,1,0,1]],dtype='<f4')
        binary=vertices.tobytes()+np.array([0,1,2],dtype='<u2').tobytes()+b'\0\0'
        accessors=[dict(bufferView=0,byteOffset=offset,componentType=5126,count=3,type=kind) for offset,kind in ((0,'VEC3'),(12,'VEC3'),(24,'VEC2'))]
        accessors.append(dict(bufferView=1,componentType=5123,count=3,type='SCALAR'))
        doc=dict(asset=dict(version='2.0'),buffers=[dict(byteLength=len(binary))],
                 bufferViews=[dict(buffer=0,byteOffset=0,byteLength=96,byteStride=32),dict(buffer=0,byteOffset=96,byteLength=6)],
                 accessors=accessors,meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,NORMAL=1,TEXCOORD_0=2),indices=3)])],
                 nodes=[dict(mesh=0,scale=[.001]*3)],scenes=[dict(nodes=[0])],scene=0)
        text=json.dumps(doc).encode();text+=b' '*((-len(text))%4)
        chunks=struct.pack('<II',len(text),0x4e4f534a)+text+struct.pack('<II',len(binary),0x004e4942)+binary
        raw=struct.pack('<III',0x46546c67,2,12+len(chunks))+chunks
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.glb';path.write_bytes(raw);mesh=read_glb(path,'test')
        np.testing.assert_allclose(mesh.positions,[[0,0,0],[1,0,0],[0,1,0]])
        np.testing.assert_allclose(mesh.normals,[[0,0,1]]*3)
        np.testing.assert_array_equal(mesh.triangles,[[0,1,2]])


if __name__=='__main__':unittest.main()
