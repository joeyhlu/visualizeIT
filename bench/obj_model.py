"""Read scanned OBJ geometry without loading scripts, material paths or external files."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .model import Mesh


@dataclass
class Scan:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    original: Mesh
    repairedNormalVertices: int = 0


def obj_index(value, count):
    number = int(value)
    result = number-1 if number > 0 else count+number
    if number == 0 or result < 0 or result >= count:
        raise ValueError('OBJ index outside declared attributes.')
    return result


def read_obj(path: Path, name='scan'):
    positions, texcoords, normals, corners = [], [], [], []
    with path.open(encoding='utf-8') as source:
        for line in source:
            tokens = line.partition('#')[0].split()
            if not tokens:
                continue
            kind, values = tokens[0], tokens[1:]
            if kind in ('v', 'vn'):
                if len(values) < 3:
                    raise ValueError('Incomplete OBJ vertex or normal.')
                (positions if kind == 'v' else normals).append([float(v) for v in values[:3]])
            elif kind == 'vt':
                if len(values) < 2:
                    raise ValueError('Incomplete OBJ UV.')
                texcoords.append([float(v) for v in values[:2]])
            elif kind == 'f':
                if len(values) < 3:
                    raise ValueError('OBJ face needs at least three corners.')
                face = []
                for value in values:
                    parts = value.split('/')
                    if len(parts) < 2 or not parts[1]:
                        raise ValueError('This benchmark requires explicit source texture UVs.')
                    face.append((obj_index(parts[0], len(positions)), obj_index(parts[1], len(texcoords)),
                                 obj_index(parts[2], len(normals)) if len(parts) > 2 and parts[2] else -1))
                for i in range(1, len(face)-1):
                    corners.append([face[0], face[i+1], face[i]])
    if not positions or not corners:
        raise ValueError('OBJ has no geometry.')
    p = np.asarray(positions, dtype=float)[:, [0, 2, 1]]
    if not np.isfinite(p).all():
        raise ValueError('OBJ positions must be finite.')
    # Original YCB OBJ coordinates are treated as metres and Z-up; retain metric scale.
    # Swapping Y/Z reverses handedness, so winding was reversed while reading faces.
    p -= np.array([(p[:, 0].min()+p[:, 0].max())/2, p[:, 1].min(), (p[:, 2].min()+p[:, 2].max())/2])
    triples = np.asarray(corners, dtype=int)
    faces = triples[:, :, 0]
    edges = p[faces]
    face_normals = np.cross(edges[:, 1]-edges[:, 0], edges[:, 2]-edges[:, 0])
    vertex_normals = np.zeros_like(p)
    for column in range(3):
        np.add.at(vertex_normals, faces[:, column], face_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-15)
    n = np.asarray(normals, dtype=float)[:, [0, 2, 1]] if normals else np.empty((0, 3))
    uv = np.asarray(texcoords, dtype=float)
    unique, inverse = np.unique(triples.reshape(-1, 3), axis=0, return_inverse=True)
    expanded_normals = vertex_normals[unique[:, 0]].copy()
    has_normal = unique[:, 2] >= 0
    if has_normal.any():
        expanded_normals[has_normal] = n[unique[has_normal, 2]]
    invalid_normal = ~np.isfinite(expanded_normals).all(axis=1)
    expanded_normals[invalid_normal] = vertex_normals[unique[invalid_normal, 0]]
    original = Mesh(name, p[unique[:, 0]], expanded_normals, uv[unique[:, 1]], inverse.reshape(-1, 3))
    return Scan(p, vertex_normals, faces, original, int(invalid_normal.sum()))
