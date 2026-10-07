"""Bounded ASCII PLY reader for the verified official BOP YCB models."""
import numpy as np
from .model import Mesh, Camera
from .obj_model import Scan


def read_ply(path, name):
    with path.open(encoding='ascii') as source:
        if source.readline().strip() != 'ply' or source.readline().strip() != 'format ascii 1.0':
            raise ValueError('Expected the official ASCII BOP PLY variant.')
        props, count, faces_count, element = [], 0, 0, None
        for line in source:
            tokens = line.split()
            if tokens[:2] == ['element', 'vertex']: count = int(tokens[2]); element = 'vertex'
            elif tokens[:2] == ['element', 'face']: faces_count = int(tokens[2]); element = 'face'
            elif tokens and tokens[0] == 'property' and element == 'vertex': props.append(tokens[-1])
            elif tokens and tokens[0] == 'end_header': break
        if min(count, faces_count) < 1 or max(count, faces_count) > 1000000:
            raise ValueError('Invalid/oversized mesh.')
        data = np.array([[float(v) for v in source.readline().split()] for _ in range(count)])
        faces = []
        for _ in range(faces_count):
            row = [int(v) for v in source.readline().split()]
            if row[0] != 3 or len(row) != 4: raise ValueError('Only triangles are accepted.')
            faces.append(row[1:])
    positions_source = data[:, [props.index(k) for k in ('x', 'y', 'z')]]*.001
    if not np.isfinite(positions_source).all(): raise ValueError('Nonfinite model positions.')
    matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    positions = positions_source@matrix.T
    shift = -np.array([(positions[:, 0].min()+positions[:, 0].max())/2, positions[:, 1].min(),
                       (positions[:, 2].min()+positions[:, 2].max())/2])
    positions += shift
    triangles = np.array(faces, dtype=int)[:, [0, 2, 1]]
    if (triangles < 0).any() or (triangles >= count).any(): raise ValueError('Invalid model indices.')
    normals = np.zeros_like(positions)
    p = positions[triangles]
    face_normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    for i in range(3): np.add.at(normals, triangles[:, i], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    uv = data[:, [props.index('texture_u'), props.index('texture_v')]] if 'texture_u' in props else np.zeros((count, 2))
    original = Mesh(name, positions, normals, uv, triangles)
    transform = np.eye(4); transform[:3, :3] = matrix; transform[:3, 3] = shift
    return Scan(positions, normals, triangles, original), positions_source, transform


def pose_camera(rotation, translation, calibration, bench_from_source, width, height):
    rotation = np.asarray(rotation).reshape(3, 3)
    translation = np.asarray(translation).reshape(3)
    basis = bench_from_source[:3, :3]
    shift = bench_from_source[:3, 3]
    eye = basis@(-rotation.T@translation)+shift
    k = np.array(calibration['cam_K']).reshape(3, 3)
    return Camera(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2], eye, rotation@basis.T)
