"""Independent annotated-pose controls, RGB tracking, measured-depth compositing and scoring."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
from io import StringIO
import numpy as np
from PIL import Image
from .model import Mesh
from .mapping_candidates import correct_scale, design_uv
from .ply_model import read_ply, pose_camera
from .renderer import render, ray_triangle
from .scan_mapping import unwrap, mapping_quality
from .tracker import RigidTracker
from .vision import cv2
from .ycb_assets import ROOT
from .storage import save_image, save_arrays, write_artifact


def load_frame(root, record):
    images = []
    for key in ('rgb', 'depth'):
        path = root/record[key]['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record[key]['sha256']: raise ValueError('Capture hash mismatch.')
        images.append(np.asarray(Image.open(path).convert('RGB')) if key == 'rgb' else np.asarray(Image.open(path)).astype(float)*record['calibration']['depth_scale']*.001)
    return images


def annotated_pose(record):
    pose = record['evaluationPose']
    rotation = np.array(pose['cam_R_m2c']).reshape(3, 3)
    u, _, vt = np.linalg.svd(rotation)
    repaired = u@vt
    if np.linalg.det(repaired) < 0 or np.linalg.norm(rotation-repaired) > .001:
        raise ValueError('Annotation rotation is not a near-rigid proper rotation.')
    return repaired, np.array(pose['cam_t_m2c'])*.001


def build_references(root, records, mesh, source_transform):
    points, descriptors = [], []
    last_pixels, last_points, last_gray = None, None, None
    inverse = np.linalg.inv(source_transform)
    orb = cv2.ORB_create(nfeatures=2000)
    for record in records:
        rgb, measured_depth = load_frame(root, record)
        rotation, translation = annotated_pose(record)
        camera = pose_camera(rotation, translation, record['calibration'], source_transform, rgb.shape[1], rgb.shape[0])
        frame = render(mesh, camera)
        mask = ((frame.triangle >= 0)&(measured_depth > 0)&(np.abs(frame.depth-measured_depth) <= np.maximum(.005, measured_depth*.01))).astype(np.uint8)*255
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        features, desc = orb.detectAndCompute(gray, mask)
        current_points, current_pixels, current_descriptors = [], [], []
        if desc is not None:
            for feature, descriptor in zip(features, desc):
                x, y = feature.pt; px, py = int(x), int(y)
                index = frame.triangle[py, px]
                if index < 0: continue
                direction = camera.rays([x, y])
                indices = mesh.triangles[index]
                hit = ray_triangle(camera.eye, direction, mesh.positions[indices], mesh.uv[indices])
                if hit is None: continue
                bench_point = camera.eye+direction*hit[0]
                source_point = inverse[:3, :3]@bench_point+inverse[:3, 3]
                current_points.append(source_point); current_pixels.append([x, y]); current_descriptors.append(descriptor)
        if current_points:
            points.extend(current_points); descriptors.extend(current_descriptors)
        last_points = np.asarray(current_points).reshape(-1, 3)
        last_pixels = np.asarray(current_pixels).reshape(-1, 2)
        last_gray = gray
    return np.asarray(points).reshape(-1, 3), np.asarray(descriptors, dtype=np.uint8).reshape(-1, 32), last_gray, last_pixels, last_points


def composite(rgb, measured_depth, rendered):
    tolerance = np.maximum(.005, measured_depth*.01)
    mask = (rendered.triangle >= 0)&(measured_depth > 0)&(rendered.depth <= measured_depth+tolerance)
    # Diagnostic neutral shading; photometric realism is deliberately not claimed.
    output = rgb.copy(); output[mask] = rendered.color[mask]
    return output, mask


def projection(points, rotation, translation, k):
    camera = points@rotation.T+translation
    image = camera[:, :2]/camera[:, 2, None]
    return image@k[:2, :2].T+k[:2, 2], camera[:, 2]


def load_evaluation_mask(root, record):
    if 'evaluationMask' not in record: raise ValueError('Evaluation mask missing; complete bounded acquisition first.')
    raw = (root/record['evaluationMask']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['evaluationMask']['sha256']: raise ValueError('Evaluation mask hash changed.')
    return np.asarray(Image.open(root/record['evaluationMask']['path'])) > 0


def annotations_for(record, source_points, source_normals, depth, result, k, evaluation_mask=None):
    rotation, translation = annotated_pose(record)
    expected, z = projection(source_points, rotation, translation, k)
    facing = np.sum((source_normals@rotation.T)*(source_points@rotation.T+translation), axis=1) < 0
    h, w = depth.shape
    x, y = np.floor(expected[:, 0]).astype(int), np.floor(expected[:, 1]).astype(int)
    inside = (x >= 0)&(x < w)&(y >= 0)&(y < h)&(z > .03)&facing
    sample = depth[np.clip(y, 0, h-1), np.clip(x, 0, w-1)]
    visible = inside&(sample > 0)&(np.abs(z-sample) <= np.maximum(.005, sample*.01))
    if evaluation_mask is not None:
        if evaluation_mask.shape != depth.shape: raise ValueError('Evaluation mask/calibration resolution mismatch.')
        visible &= evaluation_mask[np.clip(y,0,h-1),np.clip(x,0,w-1)]
    actual = projection(source_points, result.rotation, result.translation, k)[0] if result.state == 'tracking' else None
    rows = []
    # Isotropic 720p-equivalent scaling; report native errors separately.
    factor = 720/h
    for point in np.flatnonzero(visible):
        rows.append(dict(frame=record['frameId'], point=int(point), valid=int(actual is not None),
                         expected_x=float(expected[point, 0]*factor), expected_y=float(expected[point, 1]*factor),
                         observed_x=float(actual[point, 0]*factor) if actual is not None else '',
                         observed_y=float(actual[point, 1]*factor) if actual is not None else ''))
    return rows


def scorer(rows):
    spec = importlib.util.spec_from_file_location('attachment_scorer', ROOT/'tools'/'evaluate_tracking.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.evaluate(rows) if rows else dict(attachmentGatePassed=False, reason='No visible evaluation landmarks')


def write_annotations(path, rows):
    stream = StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=['frame','point','valid','expected_x','expected_y','observed_x','observed_y'])
    writer.writeheader(); writer.writerows(rows)
    write_artifact(path,stream.getvalue())


def build_gallery(output,reports):
    sections = []
    def number(value): return 'unavailable' if value is None else f'{value:.2f}'
    for report in reports:
        row = report['thumbnails'][0]; root_name = Path(row['control']).parent.as_posix(); score = report['score']
        images = ''.join(f'<figure><img src="{root_name}/{mode}.gif"><figcaption>{mode}</figcaption></figure>' for mode in ('supplied-pose','estimated-pose'))
        sections.append(f'<h2>{report["name"]}</h2><p>Selection gate: {report["selectionGatePassed"]}; tracking gate: {score["provisionalTrackingGatePassed"]}. {report["selectionFailureReason"] or ""}</p><p>720p-equivalent attachment error: median <b>{number(score.get("medianErrorPixels"))} px</b>, P95 <b>{number(score.get("p95ErrorPixels"))} px</b>. Eligible-frame tracking availability: <b>{score["eligibleVisibleFrameAvailability"]*100:.1f}%</b>.</p><div>{images}</div><a href="{root_name}/report.json">Frame results</a> · <a href="{root_name}/annotations.csv">Independent evaluation CSV</a>')
    html = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real footage tracking</title><style>body{font:17px system-ui;margin:30px;background:#edf2ef;color:#183f37}div{display:flex}figure{width:50%;margin:6px}img{width:100%}p{line-height:1.5}a{color:#176c61}</style><h1>Real recorded RGB-D footage</h1><p>Left: supplied-pose rendering control. Right: our estimated poses; overlay hidden when limited/lost. First ten frames supply reference onboarding. GIFs sample evaluation frames with presentation timing, not dataset FPS. YCB-Video/BOP data; this is not phone validation or a BOP leaderboard score.</p>'+''.join(sections)+'<p>Source: <a href="https://bop.felk.cvut.cz/datasets/">BOP/YCB-Video</a> (MIT, Xiang et al.). Original YCB models: Calli et al., CC BY 4.0. <a href="report.json">Full results</a></p>'
    write_artifact(output/'index.html',html)


def evaluate_saved(output, root=None):
    reports = []
    for path in sorted(output.glob('object-*/report.json')):
        report = json.loads(path.read_text())
        if root is not None:
            capture = json.loads((root/f'object-{report["objectId"]:06d}.json').read_text())
            scan, positions, transform = read_ply(root/'models'/f'obj_{report["objectId"]:06d}.ply',report['name'])
            indices = np.linspace(0,len(positions)-1,min(512,len(positions)),dtype=int)
            points = positions[indices]; normals = scan.normals[indices]@transform[:3,:3]
            rows = []
            if len(capture['frames'])-10 != len(report['frames']): raise ValueError('Saved trace must include every held-out frame.')
            for record, result in zip(capture['frames'][10:],report['frames']):
                if record['frameId'] != result['frameId']: raise ValueError('Saved trace and capture frame IDs differ.')
                matrix = np.array(result['cvCameraFromSourceObject']) if result['cvCameraFromSourceObject'] is not None else None
                from .tracker import TrackingResult
                estimate = TrackingResult(result['frameId'],result['state'],result['failureReason'],
                    matrix[:3,:3] if matrix is not None else None,matrix[:3,3] if matrix is not None else None,{})
                _,depth = load_frame(root,record); k = np.array(record['calibration']['cam_K']).reshape(3,3)
                rows.extend(annotations_for(record,points,normals,depth,estimate,k,load_evaluation_mask(root,record)))
            write_annotations(path.parent/'annotations.csv',rows)
            report['visibilityEvaluation'] = 'Independent annotated visibility mask + source depth + facing normals; no masks fed to tracker'
        with (path.parent/'annotations.csv').open(newline='') as source:
            updated = scorer(list(csv.DictReader(source)))
        report['score'].update(updated)
        if root is not None:
            factor = 720/depth.shape[0]
            report['score']['nativeMedianErrorPixels'] = updated['medianErrorPixels']/factor if updated.get('medianErrorPixels') is not None else None
            report['score']['nativeP95ErrorPixels'] = updated['p95ErrorPixels']/factor if updated.get('p95ErrorPixels') is not None else None
        report['score']['provisionalTrackingGatePassed'] = bool(updated.get('medianErrorPixels') is not None
            and updated['medianErrorPixels'] < 5 and updated['p95ErrorPixels'] < 10
            and report['score']['eligibleVisibleFrameAvailability'] >= .9 and report['selectionGatePassed'])
        asset = np.load(path.parent/'object-asset.npz')
        if 'baselineMappingQuality' not in report: report['baselineMappingQuality'] = report['mappingQuality']
        report['mappingQuality'] = mapping_quality(asset['positions'],asset['triangles'],asset['metricUV'])[0]
        recoveries, loss_start = [], None
        for index, result in enumerate(report['frames']):
            if result['state'] != 'tracking' and loss_start is None: loss_start = index
            elif result['state'] == 'tracking' and loss_start is not None:
                recoveries.append(index-loss_start); loss_start = None
        report['reacquisition'] = dict(lossRunsRecovered=len(recoveries), delaysFrames=recoveries,
                                      unresolvedFinalLossFrames=len(report['frames'])-loss_start if loss_start is not None else 0)
        if root is not None:
            translation_jitter, rotation_jitter, previous = [], [], None
            for index,(record,result) in enumerate(zip(capture['frames'][10:],report['frames'])):
                if result['state'] != 'tracking': previous = None; continue
                matrix = np.array(result['cvCameraFromSourceObject']); reference_r,reference_t = annotated_pose(record)
                residual_t = matrix[:3,3]-reference_t; residual_r = matrix[:3,:3]@reference_r.T
                if previous is not None:
                    translation_jitter.append(float(np.linalg.norm(residual_t-previous[0])))
                    difference = residual_r@previous[1].T
                    rotation_jitter.append(float(np.degrees(np.arccos(np.clip((np.trace(difference)-1)/2,-1,1)))))
                previous = (residual_t,residual_r)
            report['rawPoseResidualJitter'] = dict(scope='Differences in pose error versus supplied annotations on consecutive valid frames; not sensor jitter',
                p95TranslationMetres=float(np.percentile(translation_jitter,95)) if translation_jitter else None,
                p95RotationDegrees=float(np.percentile(rotation_jitter,95)) if rotation_jitter else None)
        write_artifact(path,json.dumps(report, indent=2, allow_nan=False)+'\n')
        reports.append(report)
    if not reports: raise ValueError('No existing tracking results to evaluate.')
    write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1, physicalPhoneValidationPassed=False, objects=reports), indent=2)+'\n')
    build_gallery(output,reports)
    print(f'Rescored {len(reports)} saved annotation files without invoking tracker or renderer.')


def load_design(path):
    if path is None: return None
    with Image.open(path) as image:
        if image.width*image.height > 16_000_000: raise ValueError('Design exceeds 16 million pixels.')
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        return np.asarray(Image.alpha_composite(background, rgba).convert('RGB'))


def display_mesh(mesh, metric, texture, mode, tile, rotation):
    aspect = texture.shape[1]/texture.shape[0] if texture is not None else 1
    uv = design_uv(metric, mode == 'repeat', tile, aspect, rotation)
    return Mesh(mesh.name, mesh.positions, mesh.normals, uv, mesh.triangles)


def controls_only(root, output, manifests, stride, texture=None, mode='repeat', tile=.04, rotation=0):
    links = []
    for path in manifests:
        manifest = json.loads(path.read_text()); obj = manifest['objectId']; name = manifest['name']
        scan, _, transform = read_ply(root/'models'/f'obj_{obj:06d}.ply', name)
        mesh, _, _, _, details = unwrap(scan, name,details=True)
        metric=correct_scale(mesh,details['metricUV'],details['chartIds'])
        mesh=display_mesh(mesh,metric,texture,mode,tile,rotation)
        images = []; directory = output/f'object-{obj:06d}-control'; directory.mkdir(exist_ok=True)
        records = manifest['frames'][10:]
        for record in records[::stride]:
            rgb, depth = load_frame(root, record); pose_rotation, translation = annotated_pose(record)
            camera = pose_camera(pose_rotation, translation, record['calibration'], transform, rgb.shape[1], rgb.shape[0])
            images.append(Image.fromarray(composite(rgb, depth, render(mesh, camera,
                texture_image=texture, texture_repeat=mode == 'repeat'))[0]))
        save_image(images[0],directory/'supplied-pose.gif',save_all=True,append_images=images[1:],duration=350,loop=0)
        (directory/'report.json').write_text(json.dumps(dict(name=name, isTrackingResult=False,
            poseSource='Evaluation annotations', selectionGatePassed=manifest['selectionGatePassed'],
            designMode=mode, tileWidthMetres=tile, rotationDegrees=rotation, artworkSeamsValidated=False,
            phoneValidationPassed=False, cameraFrames=[r['calibration'] for r in records[::stride]]),indent=2)+'\n')
        links.append(f'<h2>{name}: supplied-pose control</h2><img width="640" src="{directory.name}/supplied-pose.gif">')
    (output/'controls.html').write_text('<!doctype html><meta charset="utf-8"><title>Supplied-pose controls</title><h1>Projection/compositing controls, not tracking results</h1>'+''.join(links))


def run(root, output, operation='all', interrupted=False, object_id=None, stride=10,
        design=None, mode='repeat', tile=.04, rotation_degrees=0):
    output.mkdir(parents=True, exist_ok=True)
    if operation == 'evaluate':
        evaluate_saved(output,root); return
    texture = load_design(design)
    if mode == 'artwork' and texture is None: raise ValueError('Artwork mode requires --design.')
    manifests = sorted(root.glob('object-*.json'))
    if object_id is not None: manifests = [p for p in manifests if json.loads(p.read_text())['objectId'] == object_id]
    if not manifests: raise ValueError('No acquired object manifests; run bounded acquisition first.')
    if operation == 'control':
        controls_only(root, output, manifests, stride, texture, mode, tile, rotation_degrees); return
    reports = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        obj = manifest['objectId']; name = manifest['name']
        directory = output/f'object-{obj:06d}'
        if interrupted: directory = output/f'object-{obj:06d}-interruptions'
        directory.mkdir(exist_ok=True)
        if 'model' not in manifest or hashlib.sha256((root/manifest['model']['path']).read_bytes()).hexdigest() != manifest['model']['sha256']:
            raise ValueError('Matching model hash missing or changed.')
        scan, source_positions, transform = read_ply(root/'models'/f'obj_{obj:06d}.ply', name)
        mesh, quality, _, _, details = unwrap(scan, name, details=True)
        metric = correct_scale(mesh, details['metricUV'], details['chartIds'])
        baseline_quality = quality
        quality, _, _ = mapping_quality(mesh.positions, mesh.triangles, metric)
        mesh = display_mesh(mesh, metric, texture, mode, tile, rotation_degrees)
        frames = manifest['frames']; onboarding = frames[:10]; evaluation = frames[10:]
        refs = build_references(root, onboarding, mesh, transform)
        if not len(refs[0]): raise ValueError(f'{name}: no valid onboarding references.')
        tracker = RigidTracker(refs[0], refs[1], source_positions, *refs[2:])
        save_arrays(directory/'object-asset.npz', schemaVersion=2, positions=mesh.positions, normals=mesh.normals,
                            triangles=mesh.triangles, metricUV=metric, sourceVertices=details['sourceVertices'],
                            sourceFaces=details['sourceFaces'], chartIds=details['chartIds'], benchFromSource=transform,
                            referencePoints=refs[0], referenceDescriptors=refs[1])
        sample_indices = np.linspace(0, len(source_positions)-1, min(512, len(source_positions)), dtype=int)
        source_points = source_positions[sample_indices]
        source_normals = scan.normals[sample_indices]@transform[:3, :3]
        rows, results, durations, control_frames, tracked_frames = [], [], [], [], []
        thumbnails = []
        for index, record in enumerate(evaluation):
            rgb, depth = load_frame(root, record)
            k = np.array(record['calibration']['cam_K']).reshape(3, 3)
            started = time.perf_counter()
            tracker_rgb = np.zeros_like(rgb) if interrupted and 30 <= index < 33 else rgb
            result = tracker.update(tracker_rgb, k, record['frameId'])
            durations.append(time.perf_counter()-started)
            results.append(result.manifest())
            rows.extend(annotations_for(record, source_points, source_normals, depth, result, k,load_evaluation_mask(root,record)))
            # Full tracking is scored on every frame; expensive CPU renders are sampled.
            if index%stride == 0 or index == len(evaluation)-1:
                rotation, translation = annotated_pose(record)
                camera = pose_camera(rotation, translation, record['calibration'], transform, rgb.shape[1], rgb.shape[0])
                control = composite(rgb, depth, render(mesh, camera, texture_image=texture, texture_repeat=mode == 'repeat'))[0]
                tracked = rgb.copy()
                if result.state == 'tracking':
                    estimated_camera = pose_camera(result.rotation, result.translation, record['calibration'], transform, rgb.shape[1], rgb.shape[0])
                    tracked = composite(rgb, depth, render(mesh, estimated_camera, texture_image=texture, texture_repeat=mode == 'repeat'))[0]
                label = f'{record["frameId"]:06d}'
                save_image(Image.fromarray(rgb),directory/f'{label}-original.jpg', quality=90)
                save_image(Image.fromarray(control),directory/f'{label}-supplied-pose.jpg', quality=90)
                save_image(Image.fromarray(tracked),directory/f'{label}-estimated-pose.jpg', quality=90)
                control_frames.append(Image.fromarray(control)); tracked_frames.append(Image.fromarray(tracked))
                thumbnails.append(dict(frameId=record['frameId'], state=result.state,
                                       original=f'{directory.name}/{label}-original.jpg', control=f'{directory.name}/{label}-supplied-pose.jpg',
                                       tracked=f'{directory.name}/{label}-estimated-pose.jpg'))
                print(f'{name}: evaluated frame {index+1}/{len(evaluation)}, {result.state}, {result.reason}', flush=True)
        for pose_mode, images in (('supplied-pose', control_frames), ('estimated-pose', tracked_frames)):
            save_image(images[0],directory/f'{pose_mode}.gif', save_all=True, append_images=images[1:], duration=350, loop=0)
        write_annotations(directory/'annotations.csv',rows)
        score = scorer(rows)
        all_availability = sum(r['state'] == 'tracking' for r in results)/len(results)
        eligible = [r for r, capture in zip(results, evaluation) if capture['evaluationInfo']['visib_fract'] >= .5]
        eligible_availability = sum(r['state'] == 'tracking' for r in eligible)/len(eligible) if eligible else 0
        factor = 720/load_frame(root, evaluation[0])[0].shape[0]
        score.update(nativeMedianErrorPixels=score.get('medianErrorPixels')/factor if score.get('medianErrorPixels') is not None else None,
                     nativeP95ErrorPixels=score.get('p95ErrorPixels')/factor if score.get('p95ErrorPixels') is not None else None,
                     allFrameAvailability=all_availability, eligibleVisibleFrameAvailability=eligible_availability,
                     eligibleFrames=len(eligible), frameResolution='Native source; scorer errors are 720p-equivalent',
                     provisionalTrackingGatePassed=bool(score.get('medianErrorPixels') is not None and score['medianErrorPixels'] < 5
                         and score['p95ErrorPixels'] < 10 and eligible_availability >= .9 and manifest['selectionGatePassed']))
        report = dict(schemaVersion=1, name=name, objectId=obj, sceneId=manifest['sceneId'],
                      selectionGatePassed=manifest['selectionGatePassed'], selectionFailureReason=manifest['selectionFailureReason'],
                      physicalPhoneValidationPassed=False, onboarding='First ten annotated-pose frames; no evaluation pose/mask fed to tracker',
                      evaluationFrames=len(results), referenceFeatures=len(refs[0]), injectedInterruptions=interrupted,
                      suppliedPoseIsTrackingResult=False, appearance='Neutral diffuse diagnostic, not PBR',
                      presentation='JPEG/GIF previews; original lossless captures retained and used for scoring',
                      annotationRotationPolicy='Project near-rigid rotations onto SO(3); reject correction above 0.001 Frobenius norm',
                      occlusion='Measured source depth; unknown depth suppressed; no GT mask used for tracking/compositing',
                      score=score, trackingTimeMedianSeconds=float(np.median(durations)), trackingTimeP95Seconds=float(np.percentile(durations, 95)),
                      throughputScope='CPU tracking only, excluding loading/rendering; not phone FPS', thumbnails=thumbnails,
                      frames=results, mappingQuality=quality)
        report['baselineMappingQuality'] = baseline_quality
        report['designSettings'] = dict(mode=mode, tileWidthMetres=tile, rotationDegrees=rotation_degrees,
            imageAspect=texture.shape[1]/texture.shape[0] if texture is not None else 1,
            imageSha256=hashlib.sha256(design.read_bytes()).hexdigest() if design is not None else None,
            artworkSeamsValidated=False)
        recoveries, loss_start = [], None
        for index, result in enumerate(results):
            if result['state'] != 'tracking' and loss_start is None: loss_start = index
            elif result['state'] == 'tracking' and loss_start is not None:
                recoveries.append(index-loss_start); loss_start = None
        report['reacquisition'] = dict(lossRunsRecovered=len(recoveries), delaysFrames=recoveries,
                                      unresolvedFinalLossFrames=len(results)-loss_start if loss_start is not None else 0)
        write_artifact(directory/'report.json',json.dumps(report, indent=2, allow_nan=False)+'\n')
        reports.append(report)
        write_artifact(output/'report.json',json.dumps(dict(schemaVersion=1, physicalPhoneValidationPassed=False, objects=reports), indent=2)+'\n')
        print(f'{name}: median {score.get("medianErrorPixels")}, p95 {score.get("p95ErrorPixels")}, availability {eligible_availability:.1%}; gate {score["provisionalTrackingGatePassed"]}', flush=True)
    if not reports: raise ValueError('No acquired object manifests; run bounded acquisition first.')
    build_gallery(output,reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT/'.cache'/'bop')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'video-tracking')
    parser.add_argument('--interruptions', action='store_true')
    parser.add_argument('--operation', choices=['all', 'control', 'track', 'evaluate'], default='all')
    parser.add_argument('--object', type=int, choices=[5,14,15])
    parser.add_argument('--render-stride', type=int, default=10, help='Score every frame; render every Nth frame.')
    parser.add_argument('--design', type=Path, help='Imported PNG/JPEG; defaults to diagnostic checker.')
    parser.add_argument('--design-mode', choices=['repeat', 'artwork'], default='repeat')
    parser.add_argument('--tile-metres', type=float, default=.04)
    parser.add_argument('--rotation-degrees', type=float, default=0)
    args = parser.parse_args()
    if args.render_stride < 1: parser.error('Render stride must be positive.')
    if not np.isfinite(args.tile_metres) or args.tile_metres <= 0 or not np.isfinite(args.rotation_degrees):
        parser.error('Tile width must be positive and rotation finite.')
    run(args.assets, args.output, args.operation, args.interruptions, args.object, args.render_stride,
        args.design, args.design_mode, args.tile_metres, args.rotation_degrees)
