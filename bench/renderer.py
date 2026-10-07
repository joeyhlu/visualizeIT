"""CPU reference renderer: near clipping, depth buffering and perspective-correct UVs.

Basic diffuse illumination is for inspecting shape, not a Unity PBR simulation.
"""
from dataclasses import dataclass
import numpy as np
from .model import unit


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    uv: np.ndarray
    triangle: np.ndarray
    object_id: np.ndarray


def empty_frame(camera):
    color = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    color[:] = [239, 243, 241]
    return Frame(color, np.full((camera.height, camera.width), np.inf),
                 np.full((camera.height, camera.width, 2), np.nan),
                 np.full((camera.height, camera.width), -1, dtype=np.int32),
                 np.full((camera.height, camera.width), -1, dtype=np.int32))


def near_clip(vertices, near):
    # Vertex data packs camera XYZ, UV, and world normal; interpolation occurs before projection.
    polygon = list(vertices)
    output = []
    for i, current in enumerate(polygon):
        previous = polygon[i-1]
        current_inside = current[2] >= near
        previous_inside = previous[2] >= near
        if current_inside != previous_inside:
            amount = (near-previous[2])/(current[2]-previous[2])
            intersection = previous+amount*(current-previous)
            intersection[2] = near
            output.append(intersection)
        if current_inside:
            output.append(current)
    return [np.array([output[0], output[i], output[i+1]]) for i in range(1, len(output)-1)]


def checkerboard(uv, squares_per_tile=8):
    parity = (np.floor(uv[:, 0]*squares_per_tile)+np.floor(uv[:, 1]*squares_per_tile)).astype(np.int64) % 2
    return np.where(parity[:, None] == 0, [229, 237, 232], [26, 100, 92]).astype(float)


def render(mesh, camera, frame=None, perspective=True, cull=True, solid_color=None, object_id=0, texture_image=None, face_colors=None, texture_repeat=False, shade=True):
    frame = empty_frame(camera) if frame is None else frame
    positions = camera.camera_points(mesh.positions)
    light = unit([-.5, 1, -.7])
    for index, indices in enumerate(mesh.triangles):
        world = mesh.positions[indices]
        normal = np.cross(world[1]-world[0], world[2]-world[0])
        if cull and normal.dot(camera.eye-world.mean(axis=0)) <= 0:
            continue
        packed = np.column_stack([positions[indices], mesh.uv[indices], mesh.normals[indices]])
        for triangle in near_clip(packed, camera.near):
            pixels = camera.project_camera(triangle[:, :3])
            low = np.maximum(0, np.floor(pixels.min(axis=0)-.5).astype(int))
            high = np.minimum([camera.width-1, camera.height-1], np.ceil(pixels.max(axis=0)-.5).astype(int))
            if (high < low).any():
                continue
            x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
            sample_x, sample_y = x.ravel()+.5, y.ravel()+.5
            a, b, c = pixels
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-12:
                continue
            w0 = ((b[1]-c[1])*(sample_x-c[0])+(c[0]-b[0])*(sample_y-c[1]))/denominator
            w1 = ((c[1]-a[1])*(sample_x-c[0])+(a[0]-c[0])*(sample_y-c[1]))/denominator
            weights = np.column_stack([w0, w1, 1-w0-w1])
            inside = (weights >= -1e-9).all(axis=1)
            if not inside.any():
                continue
            weights = weights[inside]
            px, py = x.ravel()[inside], y.ravel()[inside]
            divided = weights/triangle[:, 2]
            reciprocal_depth = divided.sum(axis=1)
            depth = 1/reciprocal_depth
            visible = depth < frame.depth[py, px]-1e-10
            if not visible.any():
                continue
            px, py, depth = px[visible], py[visible], depth[visible]
            corrected = divided[visible]/reciprocal_depth[visible, None]
            interpolation = corrected if perspective else weights[visible]
            uv = interpolation@triangle[:, 3:5]
            normals = corrected@triangle[:, 5:8]
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            illumination = .42+.58*np.maximum(0, normals@light) if shade else np.ones(len(uv))
            if face_colors is not None:
                color = np.tile(face_colors[index], (len(uv), 1))
            elif solid_color is not None:
                color = np.tile(solid_color, (len(uv), 1))
            elif texture_image is not None:
                # OBJ photographic atlas uses bottom-left UV origin and clamped image borders.
                height, width = texture_image.shape[:2]
                sample_uv = np.mod(uv, 1) if texture_repeat else uv
                tx = np.clip(sample_uv[:, 0], 0, 1)*(width-1)
                ty = (1-np.clip(sample_uv[:, 1], 0, 1))*(height-1)
                x0, y0 = np.floor(tx).astype(int), np.floor(ty).astype(int)
                x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
                dx, dy = (tx-x0)[:, None], (ty-y0)[:, None]
                color = ((1-dx)*(1-dy)*texture_image[y0, x0]+dx*(1-dy)*texture_image[y0, x1]
                         +(1-dx)*dy*texture_image[y1, x0]+dx*dy*texture_image[y1, x1])
            else:
                color = checkerboard(uv)
            frame.color[py, px] = np.clip(color*illumination[:, None], 0, 255).astype(np.uint8)
            frame.depth[py, px] = depth
            frame.uv[py, px] = uv
            frame.triangle[py, px] = index
            frame.object_id[py, px] = object_id
    return frame


def ray_triangle(origin, direction, positions, uv):
    """Independent geometric oracle for tests (Moller-Trumbore ray intersection)."""
    a, b, c = np.asarray(positions)
    edge1, edge2 = b-a, c-a
    p = np.cross(direction, edge2)
    determinant = edge1.dot(p)
    if abs(determinant) < 1e-12:
        return None
    offset = origin-a
    u = offset.dot(p)/determinant
    q = np.cross(offset, edge1)
    v = direction.dot(q)/determinant
    t = edge2.dot(q)/determinant
    if t <= 0 or min(u, v, 1-u-v) < -1e-9:
        return None
    weights = np.array([1-u-v, u, v])
    return t, weights@np.asarray(uv)
