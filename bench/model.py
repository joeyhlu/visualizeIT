"""App mesh data and a calibrated synthetic camera.

World/object coordinates match Unity: metres, X right, Y up, Z forward.
Camera coordinates are X right, Y down, Z forward; image origin is top-left.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


def unit(value):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Direction must be finite and nonzero.')
    return value/length


def texture_uv(metric_uv, tile_width, aspect=1, rotation_degrees=0):
    if not all(np.isfinite(x) for x in (tile_width, aspect, rotation_degrees)) or tile_width <= 0 or aspect <= 0:
        raise ValueError('Invalid texture parameters.')
    angle = np.deg2rad(rotation_degrees % 360)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (np.asarray(metric_uv) @ rotation.T)/np.array([tile_width, tile_width*aspect])


@dataclass
class Mesh:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        self.positions = np.asarray(self.positions, dtype=float)
        self.normals = np.asarray(self.normals, dtype=float)
        self.uv = np.asarray(self.uv, dtype=float)
        indices = np.asarray(self.triangles)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Triangle indices must be integers.')
        self.triangles = indices.astype(np.int64)
        count = len(self.positions)
        if self.positions.shape != (count, 3) or self.normals.shape != (count, 3) or self.uv.shape != (count, 2):
            raise ValueError('Mesh attribute shapes must agree.')
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or count == 0:
            raise ValueError('Mesh needs vertices and triangle triplets.')
        if not all(np.isfinite(a).all() for a in (self.positions, self.normals, self.uv)):
            raise ValueError('Mesh attributes must be finite.')
        if (self.triangles < 0).any() or (self.triangles >= count).any():
            raise ValueError('Triangle index outside mesh.')

    @classmethod
    def load(cls, path: Path, material='fitted'):
        data = json.loads(path.read_text())
        pattern = next(case for case in data['patternCases'] if case['name'] == material)
        mesh = cls(data['shape'].lower(), data['positions'], data['normals'], pattern['textureUV'],
                   np.array(data['triangles']).reshape(-1, 3))
        calculated = texture_uv(data['uvMetres'], pattern['tileWidth'], pattern['aspect'], pattern['rotationDegrees'])
        error = float(np.abs(calculated-mesh.uv).max())
        if error > 2e-6:
            raise ValueError('Python mapping disagrees with the C# mapping export.')
        return mesh, data, error

    def transformed(self, translation=(0, 0, 0), yaw_degrees=0, scale=1):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Scale must be positive.')
        angle = np.deg2rad(yaw_degrees)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        return Mesh(self.name, scale*self.positions@rotation.T + np.asarray(translation),
                    self.normals@rotation.T, self.uv.copy(), self.triangles.copy())


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    eye: np.ndarray
    axes: np.ndarray
    near: float = .03

    def __post_init__(self):
        self.eye = np.asarray(self.eye, dtype=float)
        self.axes = np.asarray(self.axes, dtype=float)
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) <= 0:
            raise ValueError('Image dimensions must be positive integers.')
        if not all(np.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy, self.near)) or min(self.fx, self.fy, self.near) <= 0:
            raise ValueError('Invalid calibration.')
        if self.eye.shape != (3,) or self.axes.shape != (3, 3) or not np.isfinite(self.eye).all() or not np.isfinite(self.axes).all():
            raise ValueError('Invalid camera pose.')
        if not np.allclose(self.axes@self.axes.T, np.eye(3), atol=1e-8):
            raise ValueError('Camera basis must be orthonormal.')

    @classmethod
    def look_at(cls, eye, target=(0, .1, 0), width=1280, height=720, focal=850):
        eye = np.asarray(eye, dtype=float)
        forward = unit(np.asarray(target)-eye)
        right = unit(np.cross([0, 1, 0], forward))
        down = -unit(np.cross(forward, right))
        return cls(width, height, focal, focal, width/2, height/2, eye, np.array([right, down, forward]))

    def camera_points(self, world):
        return (np.asarray(world)-self.eye)@self.axes.T

    def project_camera(self, points):
        points = np.asarray(points, dtype=float)
        pixels = points[..., :2]/points[..., 2, None]
        return pixels*np.array([self.fx, self.fy])+np.array([self.cx, self.cy])

    def project(self, world):
        points = self.camera_points(world)
        return self.project_camera(points), points[..., 2]

    def rays(self, pixels):
        pixels = np.asarray(pixels, dtype=float)
        xy = (pixels-np.array([self.cx, self.cy]))/np.array([self.fx, self.fy])
        return np.concatenate([xy, np.ones(xy.shape[:-1]+(1,))], axis=-1)@self.axes

    def manifest(self):
        transform = np.eye(4)
        transform[:3, :3] = self.axes
        transform[:3, 3] = -self.axes@self.eye
        return dict(width=self.width, height=self.height, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                    nearMetres=self.near, cameraFromUnityWorld=transform.tolist(),
                    basis='Camera X right/Y down/Z forward; Unity world X right/Y up/Z forward; metres')
