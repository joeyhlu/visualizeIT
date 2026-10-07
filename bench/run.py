"""Build labelled synthetic fixtures, camera orbits, and pose-error experiments."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .model import Camera, Mesh
from .renderer import render, ray_triangle
from tools.evaluate_tracking import evaluate

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('exact_pose_control', 'translation_3mm', 'translation_5mm', 'yaw_2deg',
             'scale_plus_2percent', 'focal_plus_2percent', 'surface_lift_2mm',
             'surface_lift_0_25mm', 'lost_3frames', 'stale_world_anchor')


def font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def card():
    return Mesh('synthetic occluder', [[-.02, .04, -.1], [.04, .04, -.1], [.04, .15, -.1], [-.02, .15, -.1]],
                [[0, 0, -1]]*4, [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 2, 1], [0, 3, 2]])


def visible_landmarks(mesh, camera, frame):
    # Centroids are known synthetic landmarks. No image feature detection is performed.
    faces = mesh.positions[mesh.triangles]
    points = faces.mean(axis=1)
    normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    pixels, depth = camera.project(points)
    front = np.sum(normals*(camera.eye-points), axis=1) > 0
    visible = front & (depth >= camera.near) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
    visible &= (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    indices = np.flatnonzero(visible)
    nearest = np.floor(pixels[indices]).astype(int)
    # The object itself must be visible: exclude points hidden by the known synthetic card.
    # 2 mm tolerance accommodates pixel-center depth on a sloping facet, not pose error.
    good = ((frame.object_id[nearest[:, 1], nearest[:, 0]] == 0)
            & (np.abs(frame.depth[nearest[:, 1], nearest[:, 0]]-depth[indices]) < .002))
    return indices[good], points[indices[good]], pixels[indices[good]]


def overlay_error_view(mesh, camera):
    frame = render(mesh, camera)
    indices, points, expected = visible_landmarks(mesh, camera, frame)
    estimated, _ = camera.project(points+np.array([.005, 0, 0]))
    image = Image.fromarray(frame.color)
    draw = ImageDraw.Draw(image)
    for truth, estimate in zip(expected[::8], estimated[::8]):
        draw.line([tuple(truth), tuple(estimate)], fill='#d64836', width=3)
        x, y = truth
        draw.ellipse((x-3, y-3, x+3, y+3), fill='#1c665e')
        x, y = estimate
        draw.ellipse((x-3, y-3, x+3, y+3), fill='#d64836')
    return image


def perspective_oracle(mesh, camera):
    good = render(mesh, camera)
    affine = render(mesh, camera, perspective=False)
    samples = np.argwhere(good.triangle >= 0)[::97]
    correct_errors, affine_errors = [], []
    for py, px in samples:
        triangle = mesh.triangles[good.triangle[py, px]]
        hit = ray_triangle(camera.eye, camera.rays([px+.5, py+.5]), mesh.positions[triangle], mesh.uv[triangle])
        if hit is None:
            raise RuntimeError('Rasterized sample did not intersect its source triangle.')
        correct_errors.append(float(np.linalg.norm(good.uv[py, px]-hit[1])))
        affine_errors.append(float(np.linalg.norm(affine.uv[py, px]-hit[1])))
    return dict(samples=len(samples), maximumCorrectUVError=max(correct_errors),
                medianAffineUVError=float(np.median(affine_errors)),
                maximumAffineUVError=max(affine_errors),
                units='Texture tile coordinates; independent ray-triangle intersection oracle'), good, affine


def save_poster(panels, output):
    poster = Image.new('RGB', (1536, 1080), '#f0f4f1')
    draw = ImageDraw.Draw(poster)
    draw.text((36, 22), 'VisualizeIt | surface mapping test bench', font=font(34), fill='#163e38')
    draw.text((36, 70), 'Actual app meshes • known camera poses • synthetic checkerboard and depth', font=font(20), fill='#405951')
    for i, (title, subtitle, image) in enumerate(panels):
        x, y = 24+(i % 3)*504, 118+(i//3)*440
        draw.text((x+8, y), title, font=font(23), fill='#173e38')
        draw.text((x+8, y+34), subtitle, font=font(16), fill='#486158')
        # Fixed central crop for presentation only. Measurements use full 1280x720 frames.
        crop = image.crop((280, 0, 1000, 720)).resize((380, 380), Image.Resampling.LANCZOS)
        poster.paste(crop, (x+62, y+60))
    draw.rectangle((0, 1010, 1536, 1080), fill='#e6ede8')
    draw.text((36, 1023), 'SIMULATION • no real object tracking, phone depth, or Unity PBR validation', font=font(23), fill='#92501e')
    poster.save(output/'mapping-model.png')


def experiment(mesh, output, frame_count):
    rows_by_case = {name: [] for name in SCENARIOS}
    cameras, animation = [], []
    frame_directory = output/'orbit'
    frame_directory.mkdir(exist_ok=True)
    loss_frames = list(range(frame_count//3, frame_count//3+3))
    for frame_index in range(frame_count):
        angle = 2*math.pi*frame_index/frame_count
        camera = Camera.look_at([.36*math.sin(angle), .24, -.36*math.cos(angle)])
        cameras.append(dict(frame=frame_index, timestampSeconds=frame_index*.13, **camera.manifest()))
        truth_frame = render(mesh, camera)
        image = Image.fromarray(truth_frame.color)
        image.save(frame_directory/f'{frame_index:03d}.png')
        animated = image.crop((280, 0, 1000, 720)).resize((480, 480), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(animated)
        draw.rectangle((0, 0, 480, 43), fill='#f0f4f1')
        draw.text((12, 8), f'SYNTHETIC CAMERA ORBIT | {round(math.degrees(angle))}°', font=font(18), fill='#173e38')
        draw.rectangle((0, 442, 480, 480), fill='#f0f4f1')
        draw.text((12, 450), 'Object-fixed UVs • known pose • no tracker', font=font(17), fill='#92501e')
        animation.append(animated)
        indices, points, pixels = visible_landmarks(mesh, camera, truth_frame)
        if not len(indices):
            raise RuntimeError(f'No visible reference points in frame {frame_index}.')
        for name in SCENARIOS:
            expected = pixels
            observation_points = points
            estimate_camera = camera
            current_indices = indices
            if name == 'translation_3mm':
                observation_points = points+np.array([.003, 0, 0])
            elif name == 'translation_5mm':
                observation_points = points+np.array([.005, 0, 0])
            elif name == 'yaw_2deg':
                observation_points = mesh.transformed(yaw_degrees=2).positions[mesh.triangles].mean(axis=1)[indices]
            elif name == 'scale_plus_2percent':
                observation_points = points*1.02
            elif name == 'focal_plus_2percent':
                estimate_camera = Camera(camera.width, camera.height, camera.fx*1.02, camera.fy*1.02,
                                         camera.cx, camera.cy, camera.eye, camera.axes, camera.near)
            elif name in ('surface_lift_2mm', 'surface_lift_0_25mm'):
                lift = .002 if name == 'surface_lift_2mm' else .00025
                lifted = mesh.positions+mesh.normals*lift
                observation_points = lifted[mesh.triangles].mean(axis=1)[indices]
            elif name == 'stale_world_anchor':
                # Ground truth moves; estimate stays at the original room placement.
                translation = np.array([.025*math.sin(angle+math.pi/4), 0, 0])
                moving_mesh = mesh.transformed(translation=translation)
                moving_frame = render(moving_mesh, camera)
                current_indices, moving_points, expected = visible_landmarks(moving_mesh, camera, moving_frame)
                observation_points = moving_points-translation
            observed, observed_depth = estimate_camera.project(observation_points)
            valid = name != 'lost_3frames' or frame_index not in loss_frames
            if valid and (observed_depth < camera.near).any():
                raise RuntimeError('A simulated valid pose projected behind the near plane.')
            for point, truth, estimate in zip(current_indices, expected, observed):
                rows_by_case[name].append(dict(frame=frame_index, point=int(point), valid=int(valid),
                                              expected_x=truth[0], expected_y=truth[1],
                                              observed_x=estimate[0] if valid else '', observed_y=estimate[1] if valid else ''))
    animation[0].save(output/'camera-orbit.gif', save_all=True, append_images=animation[1:],
                      duration=130, loop=0, optimize=False)
    scores = {}
    for name, rows in rows_by_case.items():
        with (output/f'{name}.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        score = evaluate(rows)
        score['simulatedThresholdsMet'] = score.pop('attachmentGatePassed')
        score['physicalValidationPassed'] = False
        score['poseSource'] = 'Known ground truth with deliberate pose/calibration/surface perturbation; no pose estimation'
        scores[name] = score
    return scores, cameras, loss_frames


def run(output, geometry, frames=24, shape='cylinder'):
    if frames < 12 or frames > 120:
        raise ValueError('Use 12–120 orbit frames.')
    output.mkdir(parents=True, exist_ok=True)
    meshes, fixtures, parity = {}, {}, {}
    for name in ('plane', 'box', 'cylinder'):
        mesh, data, error = Mesh.load(geometry/f'{name}.json')
        meshes[name], fixtures[name], parity[name] = mesh, data, error
        # Check all exported mapping cases, including rotation and image aspect ratio.
        for case in data['patternCases']:
            _, _, error = Mesh.load(geometry/f'{name}.json', case['name'])
            parity[f'{name}/{case["name"]}'] = error
    camera = Camera.look_at([.18, .23, -.3])
    panels = []
    for name in ('plane', 'box', 'cylinder'):
        result = render(meshes[name], camera)
        image = Image.fromarray(result.color)
        image.save(output/f'{name}-mapped.png')
        panels.append((name.title(), 'UVs attached to the exported app mesh', image))
    plane_camera = Camera.look_at([.1, .11, -.12], target=(0, 0, 0))
    oracle, good, affine = perspective_oracle(meshes['plane'], plane_camera)
    Image.fromarray(affine.color).save(output/'incorrect-affine.png')
    Image.fromarray(good.color).save(output/'correct-perspective.png')
    panels.extend([('Incorrect interpolation', 'Deliberate screen-space mapping defect', Image.fromarray(affine.color)),
                   ('Perspective corrected', 'Checked against independent ray intersections', Image.fromarray(good.color))])
    front_camera = Camera.look_at([.08, .23, -.32])
    occluded = render(meshes['cylinder'], front_camera)
    original_mask = occluded.object_id == 0
    render(card(), front_camera, frame=occluded, solid_color=[220, 155, 80], object_id=1)
    masked_pixels = int((original_mask & (occluded.object_id == 1)).sum())
    if masked_pixels <= 0:
        raise RuntimeError('Synthetic occluder did not cover the fixture.')
    Image.fromarray(occluded.color).save(output/'synthetic-occlusion.png')
    panels.append(('Synthetic depth occlusion', f'{masked_pixels:,} fixture pixels hidden by a card', Image.fromarray(occluded.color)))
    save_poster(panels, output)
    overlay_error_view(meshes['cylinder'], camera).save(output/'injected-error-5mm.png')
    scores, cameras, loss_frames = experiment(meshes[shape], output, frames)
    report = dict(schemaVersion=1, synthetic=True, physicalValidationPassed=False,
                  poseEstimatorImplemented=False, benchmarkShape=shape, imageSize=[1280, 720], orbitFrames=frames,
                  sequenceTiming='Synthetic presentation interval 0.13 seconds; not captured sensor timestamps',
                  lossFrames=loss_frames, mappingParityMaximumUVError=max(parity.values()),
                  mappingParityByCase=parity, fixtures={name: data['mappingQuality'] for name, data in fixtures.items()},
                  perspectiveOracle=oracle, syntheticOccludedPixels=masked_pixels, scenarios=scores,
                  assumptions=['Exact geometry, calibration and visibility for controls',
                               'Zero surface lift for controls; separate lift cases simulate the shader vertex offset',
                               'No sensor noise, scan reconstruction, rolling shutter or lens distortion',
                               'Known synthetic triangle-centroid landmarks; no feature detection or tracker',
                               'Basic diffuse lighting; not Unity shader/PBR validation'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'cameras.json').write_text(json.dumps(cameras, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), perspective=oracle, scenarios=scores), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'bench')
    parser.add_argument('--geometry', type=Path, default=ROOT/'artifacts'/'geometry')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--shape', choices=('plane', 'box', 'cylinder'), default='cylinder')
    args = parser.parse_args()
    run(args.output, args.geometry, args.frames, args.shape)


if __name__ == '__main__':
    main()
