"""Initialized tracking benchmark. Inference and evaluator are separate CLI stages.

Only the first ten explicitly authorized onboarding records may carry poses.
Dataset-reference scores are not independent human annotations or live FPS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .vision import cv2
from .ply_model import read_ply, pose_camera
from .renderer import render, ray_triangle
from .prototype_resource_store import ResourceStore, provider
from .prototype_output_budget import CompactOutputBudget, CompactBudgetExceeded

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def resource(root, spec, role='model'):
    if isinstance(root, ResourceStore):
        return root.open_binary(spec, role)
    path = (Path(root) / spec['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Resource escapes dataset root')
    if digest(path) != spec['sha256']:
        raise ValueError('Resource hash mismatch: ' + spec['path'])
    return path


def load_rgb(root, record):
    source = resource(root, record['rgb'], 'rgb')
    with (closing(source) if isinstance(root, ResourceStore) else nullcontext()), Image.open(source) as image:
        return np.asarray(image.convert('RGB'))


def load_depth(root, record):
    role = 'onboarding_depth' if isinstance(root, ResourceStore) and root.phase == 'inference' else 'evaluation_depth'
    source = resource(root, record['depth'], role)
    with (closing(source) if isinstance(root, ResourceStore) else nullcontext()), Image.open(source) as image:
        return np.asarray(image).astype(float) * record['calibration']['depth_scale'] * .001


def rigid_pose(record):
    pose = record['evaluationPose']
    rotation = np.asarray(pose['cam_R_m2c'], dtype=float).reshape(3, 3)
    translation = np.asarray(pose['cam_t_m2c'], dtype=float).reshape(3) * .001
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Invalid dataset pose')
    u, _, vt = np.linalg.svd(rotation)
    repaired = u @ vt
    if np.linalg.det(repaired) < 0 or np.linalg.norm(rotation - repaired) > .001:
        raise ValueError('Dataset rotation is not near proper SO(3)')
    return repaired, translation


def cv_render_camera(rotation, translation, calibration, transform, width, height):
    """Convert integer-centered CV calibration to corner-coordinate rasterizer."""
    k = np.asarray(calibration['cam_K'], dtype=float).reshape(3, 3).copy()
    k[0, 2] += .5
    k[1, 2] += .5
    return pose_camera(rotation, translation, {'cam_K': k.ravel().tolist()},
                       transform, width, height)


def build_bank(root, onboarding, mesh, transform):
    if len(onboarding) != 10:
        raise ValueError('Exactly ten authorized onboarding frames required')
    points, descriptors = [], []
    inverse = np.linalg.inv(transform)
    detector = cv2.ORB_create(nfeatures=2000)
    for record in onboarding:
        rgb, depth = load_rgb(root, record), load_depth(root, record)
        if depth.shape != rgb.shape[:2]:
            raise ValueError('RGB/depth mismatch')
        r, t = rigid_pose(record)
        camera = cv_render_camera(r, t, record['calibration'], transform,
                                  rgb.shape[1], rgb.shape[0])
        raster = render(mesh, camera)
        mask = ((raster.triangle >= 0) & (depth > 0)
                & (np.abs(raster.depth - depth) <= np.maximum(.005, depth * .01)))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        features, desc = detector.detectAndCompute(gray, mask.astype(np.uint8) * 255)
        current_points, current_pixels = [], []
        if desc is not None:
            for feature, descriptor in zip(features, desc):
                x, y = feature.pt
                px, py = int(np.floor(x + .5)), int(np.floor(y + .5))
                if not (0 <= px < camera.width and 0 <= py < camera.height):
                    continue
                face = int(raster.triangle[py, px])
                if face < 0:
                    continue
                direction = camera.rays([x + .5, y + .5])
                tri = mesh.triangles[face]
                hit = ray_triangle(camera.eye, direction, mesh.positions[tri], mesh.uv[tri])
                if hit is None:
                    continue
                point = inverse[:3, :3] @ (camera.eye + direction * hit[0]) + inverse[:3, 3]
                points.append(point)
                descriptors.append(descriptor)
                current_points.append(point)
                current_pixels.append([x, y])
        print('onboarding', record['frameId'], 'anchors', len(current_points), flush=True)
    return (np.asarray(points, dtype=float).reshape(-1, 3),
            np.asarray(descriptors, dtype=np.uint8).reshape(-1, 32), gray,
            np.asarray(current_pixels, dtype=np.float32).reshape(-1, 2),
            np.asarray(current_points, dtype=float).reshape(-1, 3))


def freeze_source():
    names = ['bench/prototype_benchmark.py', 'bench/prototype_tracking.py',
             'bench/tracker.py', 'bench/renderer.py', 'bench/ply_model.py',
             'bench/model.py', 'bench/obj_model.py', 'bench/vision.py', 'tools/prototype_bootstrap.py']
    names.append('bench/prototype_resource_store.py')
    names.append('bench/prototype_output_budget.py')
    return {name: digest(ROOT / name) for name in names}


def runtime_identity():
    import sys
    import PIL
    binaries = {Path(sys.executable), Path(np.__file__), Path(cv2.__file__), Path(Image.__file__)}
    for folder in (Path(cv2.__file__).parent, Path(np.__file__).parent / '_core',
                   Path(np.__file__).parent / 'core',
                   Path(np.__file__).parent.parent / 'numpy.libs', Path(Image.__file__).parent):
        if folder.is_dir():
            binaries.update(folder.glob('*.pyd'))
            binaries.update(folder.glob('*.dll'))
    return dict(pythonVersion=sys.version, numpyVersion=np.__version__,
                opencvVersion=cv2.__version__, pillowVersion=PIL.__version__,
                packagePaths={'numpy': np.__file__, 'opencv': cv2.__file__, 'PIL': Image.__file__},
                fileHashes={str(path.resolve()): digest(path) for path in sorted(binaries)})


def validate_test_records(records):
    if len(records) != 110:
        raise ValueError('Exactly 110 test frames required')
    allowed = {'frameId', 'logicalIndex', 'timestampSeconds', 'timestampKind', 'rgb',
               'calibration', 'width', 'height', 'sessionId'}
    for record in records:
        if set(record) - allowed:
            raise ValueError('Evaluator or unknown fields in inference record')
    ids = [row['frameId'] for row in records]
    if ids != list(range(ids[0], ids[0] + 110)):
        raise ValueError('Test frames must be consecutive and ordered')


def result_summary(frames):
    return {'scheduledFrames': len(frames),
            'trackingFrames': sum(f['state'] == 'tracking' for f in frames),
            'allFrameAvailability': sum(f['state'] == 'tracking' for f in frames) / len(frames) if frames else 0,
            'states': dict(Counter(f['state'] for f in frames)),
            'failureReasons': dict(Counter(f.get('failureReason') for f in frames if f.get('failureReason')))}


def score_trace(records, frames, errors_by_frame, corpus='YCB-Video'):
    """Pure scoring: failures stay in availability even when no landmarks are visible."""
    if [r['frameId'] for r in records] != [f['frameId'] for f in frames]:
        raise ValueError('Trace missing/reordered/extra frame')
    result = result_summary(frames)
    errors = [error for f in frames if f['state'] == 'tracking'
              for error in errors_by_frame.get(f['frameId'], [])]
    if any(not np.isfinite(e) or e < 0 for e in errors):
        raise ValueError('Invalid errors')
    eligible = [f for r, f in zip(records, frames) if r['evaluationInfo']['visib_fract'] >= .5]
    result.update(eligibleFrames=len(eligible), eligibleVisibleFrameAvailability=(
        sum(f['state'] == 'tracking' for f in eligible) / len(eligible) if eligible else 0),
        pairedPoints=len(errors), pairedFrames=sum(bool(errors_by_frame.get(f['frameId']))
                                                  for f in frames if f['state'] == 'tracking'),
        medianErrorPixels=float(np.median(errors)) if errors else None,
        p95ErrorPixels=float(np.percentile(errors, 95)) if errors else None,
        provenance=f'BOP/{corpus} dataset-reference poses, depth and visibility; no independent human annotations')
    result['attachmentGatePassed'] = bool(errors and result['medianErrorPixels'] < 5
        and result['p95ErrorPixels'] < 10 and result['allFrameAvailability'] >= .9
        and result['eligibleVisibleFrameAvailability'] >= .9)
    return result


def validate_pose_trace(frames):
    for frame in frames:
        pose = frame.get('cvCameraFromSourceObject')
        if frame['state'] != 'tracking':
            if pose is not None:
                raise ValueError('Nontracking output contains a stale pose')
            continue
        matrix = np.asarray(pose, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Tracking output is not a finite proper metre SE(3) pose')


def geometry_binding(obj):
    return dict(objectId=obj['objectId'], name=obj['name'], model=obj['model'],
        frames=[{key: r[key] for key in ('frameId', 'calibration', 'rgb')}
                for r in obj['frames']])


def validate_evaluator_binding(evaluator, seal):
    if (evaluator['benchmarkId'] != seal['benchmarkId']
            or evaluator['selectionDigest'] != seal['selectionDigest']):
        raise ValueError('Evaluator belongs to a different frozen selection')
    bindings = {obj['objectId']: obj for obj in seal['geometryBindings']}
    objects = evaluator['objects']
    if len(objects) != len(bindings) or {o['objectId'] for o in objects} != set(bindings):
        raise ValueError('Evaluator object set has duplicates, omissions or additions')
    for obj in objects:
        if geometry_binding(obj) != bindings[obj['objectId']]:
            raise ValueError('Evaluator model/RGB/frame/calibration differs from inference')


def recovery_metrics(records, frames, per_frame):
    by_id = {row['frameId']: row for row in per_frame}
    first_public = next((i for i in range(55, len(frames)) if frames[i]['state'] == 'tracking'), None)
    first_score = by_id.get(frames[first_public]['frameId']) if first_public is not None else None
    def qualified(index):
        row = by_id.get(frames[index]['frameId'])
        return bool(frames[index]['state'] == 'tracking' and row and row['pairedPoints'] > 0
            and row['medianErrorPixels'] < 5 and row['p95ErrorPixels'] < 10)
    observable_count, previous_correct, delay = 0, False, None
    for i in range(55, len(frames)):
        if records[i]['evaluationInfo']['visib_fract'] < .5:
            previous_correct = False
            continue
        observable_count += 1
        correct = qualified(i)
        if correct and previous_correct and delay is None:
            delay = observable_count
        previous_correct = correct
    interruption_observed = all(frames[i]['state'] != 'tracking'
        and frames[i].get('cvCameraFromSourceObject') is None for i in range(50, 55))
    was_tracking = frames[49]['state'] == 'tracking'
    return dict(firstPublicTrackingScoredIndex=first_public,
        firstPublicTrackingAttachment=first_score,
        interruptionObserved=interruption_observed,
        wasTrackingBeforeInterruption=was_tracking,
        wrongOrUnscorableFirstReacquisition=first_public is not None and not qualified(first_public),
        recoveryObservableFrames=delay,
        recoveryGatePassed=bool(interruption_observed and was_tracking
                               and delay is not None and delay <= 15
                               and first_public is not None and qualified(first_public)))


def evaluate_object(root, obj, frames, corpus='YCB-Video'):
    # This is intentionally invoked only after inference outputs are sealed.
    from .video_experiment import annotations_for
    from .tracker import TrackingResult
    validate_pose_trace(frames)
    try:
        scan, positions, transform = read_ply(resource(root, obj['model']), obj['name'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        score = score_trace(obj['frames'], frames, {}, corpus)
        score.update(evaluationUnavailableReason=f'{type(exc).__name__}: {exc}',
            evaluatorUnavailableFrames=len(frames), perFrameAttachment=[dict(
                frameId=f['frameId'], pairedPoints=0, medianErrorPixels=None,
                p95ErrorPixels=None, unavailableReason='evaluation_model_unavailable') for f in frames])
        return score
    indices = np.linspace(0, len(positions) - 1, min(512, len(positions)), dtype=int)
    points, normals = positions[indices], scan.normals[indices] @ transform[:3, :3]
    records = obj['frames']
    if [r['frameId'] for r in records] != [f['frameId'] for f in frames]:
        raise ValueError('Evaluation trace join mismatch')
    errors, native_errors, per_frame = {}, [], []
    for record, frame in zip(records, frames):
        matrix = frame.get('cvCameraFromSourceObject')
        matrix = np.asarray(matrix) if matrix is not None else None
        estimate = TrackingResult(frame['frameId'], frame['state'], frame.get('failureReason'),
            matrix[:3, :3] if matrix is not None else None,
            matrix[:3, 3] if matrix is not None else None, {})
        try:
            depth = load_depth(root, record)
            source = resource(root, record['evaluationMask'], 'evaluation_mask')
            with (closing(source) if isinstance(root, ResourceStore) else nullcontext()), Image.open(source) as image:
                mask = np.asarray(image) > 0
            k = np.asarray(record['calibration']['cam_K']).reshape(3, 3)
            rows = annotations_for(record, points, normals, depth, estimate, k, mask)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            per_frame.append(dict(frameId=record['frameId'], pairedPoints=0,
                medianErrorPixels=None, p95ErrorPixels=None,
                unavailableReason=f'{type(exc).__name__}: {exc}'))
            continue
        errors[record['frameId']] = [float(np.hypot(row['observed_x'] - row['expected_x'],
                                                 row['observed_y'] - row['expected_y']))
                                    for row in rows if row['valid']]
        values = errors[record['frameId']]
        native_errors.extend(e * depth.shape[0] / 720 for e in values)
        per_frame.append(dict(frameId=record['frameId'], pairedPoints=len(values),
            medianErrorPixels=float(np.median(values)) if values else None,
            p95ErrorPixels=float(np.percentile(values, 95)) if values else None,
            visibleLandmarks=len(rows), nativeHeight=depth.shape[0]))
    score = score_trace(records, frames, errors, corpus)
    score.update(nativeMedianErrorPixels=float(np.median(native_errors)) if native_errors else None,
                 nativeP95ErrorPixels=float(np.percentile(native_errors, 95)) if native_errors else None,
                 perFrameAttachment=per_frame,
                 evaluatorUnavailableFrames=sum('unavailableReason' in row for row in per_frame))
    if score['evaluatorUnavailableFrames']:
        score['attachmentGatePassed'] = False
    return score


def inference_packet(record, rgb, session, object_id, reference_id, logical_index):
    from .prototype_tracking import FrameInputV1
    return FrameInputV1(session, object_id, reference_id, record['frameId'], logical_index,
                        None, 'frame_index_no_fps', rgb,
                        np.asarray(record['calibration']['cam_K'], dtype=float).reshape(3, 3),
                        (rgb.shape[1], rgb.shape[0]))


def failed_trace(records, reason):
    return [dict(frameId=r['frameId'], state='lost', failureReason=reason,
                 cvCameraFromSourceObject=None, processingSeconds=None)
            for r in records]


def run_inference(dataset, output):
    """Reads no evaluator file; seals outputs before a separate evaluation process."""
    from .prototype_tracking import PrototypeTrackerV1
    dataset, output = Path(dataset), Path(output)
    manifest_path = dataset / 'inference.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    inputs = provider(dataset, manifest, 'inference')
    guard = CompactOutputBudget(dataset, output) if isinstance(inputs, ResourceStore) else None
    write = guard.write_json if guard else write_json
    if manifest.get('resourceStorage') and manifest.get('dataset') != 'BOP HB val_primesense':
        raise ValueError('Compact storage requires the validated HB corpus label')
    object_ids = [o['objectId'] for o in manifest['objects']]
    if not object_ids or len(set(object_ids)) != len(object_ids):
        raise ValueError('Inference object set is empty or duplicated')
    if output.exists():
        raise ValueError('Use a fresh output directory; original trials are preserved')
    guard.mkdir(output) if guard else output.mkdir(parents=True)
    frozen = dict(schemaVersion=1, inferenceManifestSHA256=digest(manifest_path),
                  dataset=manifest.get('dataset', 'BOP YCB-Video'),
                  evidenceKind=manifest.get('evidenceKind', 'dataset_reference'),
                  benchmarkMode=manifest.get('benchmarkMode', 'full'),
                  plannedObjects=manifest.get('plannedObjects', len(manifest['objects'])),
                  runtime=runtime_identity(),
                  benchmarkId=manifest['benchmarkId'], selectionDigest=manifest['selectionDigest'],
                  evaluatorManifestSHA256=manifest['evaluatorManifestSHA256'],
                  sourceHashes=freeze_source(), configuration=dict(
                      nfeatures=2000, matchRatio=.75, minInliers=12, pnpIterations=150,
                      reprojectionPixels=3, recoveryCandidates=2,
                      interruptionScoredIndices=[50, 51, 52, 53, 54], randomSeed=17),
                  sourceClock='frame_index_no_fps', liveCameraFPSMeasured=False,
                  neuralUpdatesPerSecond=0, humanAnnotationsAvailable=False)
    write(output / 'freeze.json', frozen)
    if manifest.get('resourceStorage'):
        frozen['resourceStorage'] = manifest['resourceStorage']
        frozen['datasetName'], frozen['objectIdLabel'] = 'HB', 'HB ID'
        frozen['pretrainedExposure'] = 'unknown'
        write(output / 'freeze.json', frozen)
    snapshot = output / 'source-snapshot'
    for relative, expected in frozen['sourceHashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Source changed while creating freeze')
        destination = snapshot / relative
        if guard:
            guard.copy(ROOT / relative, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        if digest(destination) != expected:
            raise ValueError('Source snapshot copy differs from frozen source')
    reports = []
    for obj in manifest['objects']:
        oid = obj['objectId']
        directory = output / f'object-{oid:06d}'
        guard.mkdir(directory) if guard else directory.mkdir()
        records = obj['frames']
        validate_test_records(records)
        onboarding = obj['onboarding']
        object_report = dict(objectId=oid, name=obj['name'], trials={},
                             initializationScope='Ten annotated onboarding frames only')
        try:
            scan, model, transform = read_ply(resource(inputs, obj['model']), obj['name'])
            started = time.perf_counter()
            references = build_bank(inputs, onboarding, scan.original, transform)
            object_report['onboardingSeconds'] = time.perf_counter() - started
            if len(references[0]) < 12:
                raise ValueError('insufficient_onboarding_references')
            asset_path = directory / 'tracking-asset.npz'
            save_asset = guard.npz if guard else np.savez_compressed
            save_asset(asset_path, referencePoints=references[0],
                referenceDescriptors=references[1], modelPositions=model,
                positions=scan.original.positions, normals=scan.original.normals,
                triangles=scan.original.triangles, uv=scan.original.uv,
                benchFromSource=transform, schemaVersion=1)
            asset_id = digest(asset_path)
            write(directory / 'asset.json', dict(objectId=oid, name=obj['name'],
                evidenceKind=frozen['evidenceKind'],
                assetPath='tracking-asset.npz', assetSHA256=asset_id, units='metres',
                sourceBasis='Original right-handed BOP source model',
                onboardingFrameIds=[r['frameId'] for r in onboarding],
                inferenceManifestSHA256=frozen['inferenceManifestSHA256']))
            seed_rgb = load_rgb(inputs, onboarding[-1])
            for interrupted in (False, True):
                session = f'benchmark-{oid}-' + ('interrupted' if interrupted else 'natural')
                seed = inference_packet(onboarding[-1], seed_rgb, session, oid, asset_id, 9)
                tracker = PrototypeTrackerV1(references[0], references[1], model,
                    session_id=session, object_id=oid, reference_id=asset_id,
                    onboarding_seed=seed, seed_pixels=references[3], seed_points=references[4])
                frames, durations = [], []
                for index, record in enumerate(records):
                    started = None
                    try:
                        rgb = load_rgb(inputs, record)
                        if interrupted and 50 <= index <= 54:
                            rgb = np.zeros_like(rgb)
                        started = time.perf_counter()
                        packet = inference_packet(record, rgb, session, oid, asset_id, index + 10)
                        result = tracker.update(packet).manifest()
                        duration = time.perf_counter() - started
                        durations.append(duration)
                        result['cvCameraFromSourceObject'] = result.pop('cameraFromObject')
                        result['processingSeconds'] = duration
                        result['forcedInterruption'] = interrupted and 50 <= index <= 54
                    except CompactBudgetExceeded:
                        raise
                    except Exception as exc:
                        tracker.reset_temporal()
                        result = failed_trace([record], f'{type(exc).__name__}: {exc}')[0]
                    frames.append(result)
                    if index % 20 == 0 or index == 109:
                        print(obj['name'], session, index + 1, result['state'], flush=True)
                trial_name = 'interrupted' if interrupted else 'natural'
                trial = dict(frames=frames, summary=result_summary(frames),
                             processingScope='Offline wrapper/preprocessing and geometric pose; excludes file decode, bank, scoring and rendering',
                             medianProcessingSeconds=float(np.median(durations)) if durations else None,
                             p95ProcessingSeconds=float(np.percentile(durations, 95)) if durations else None,
                             sourceFrameRate=None, liveInferenceFPSMeasured=False)
                write(directory / f'{trial_name}.json', trial)
                object_report['trials'][trial_name] = f'object-{oid:06d}/{trial_name}.json'
        except CompactBudgetExceeded:
            raise
        except Exception as exc:
            reason = 'onboarding_unavailable: ' + type(exc).__name__ + ': ' + str(exc)
            object_report['onboardingFailure'] = reason
            for trial_name in ('natural', 'interrupted'):
                trial_path = directory / f'{trial_name}.json'
                if not trial_path.exists():
                    trial = dict(frames=failed_trace(records, reason), liveInferenceFPSMeasured=False)
                    write(trial_path, trial)
                object_report['trials'][trial_name] = f'object-{oid:06d}/{trial_name}.json'
        reports.append(object_report)
        write(output / 'inference-progress.json', {'objects': reports})
    if (freeze_source() != frozen['sourceHashes'] or digest(manifest_path) != frozen['inferenceManifestSHA256']
            or runtime_identity() != frozen['runtime']):
        write(output / 'invalidated.json', {'reason': 'Sources or input manifest changed during inference'})
        raise ValueError('Frozen source/input changed; trial invalidated')
    if isinstance(inputs, ResourceStore):
        inputs.assert_sealed()
        write(output / 'resource-access.json', inputs.access_log)
    seal = dict(objects=reports,
        dataset=frozen['dataset'], evidenceKind=frozen['evidenceKind'],
        benchmarkMode=frozen['benchmarkMode'], plannedObjects=frozen['plannedObjects'],
        benchmarkId=manifest['benchmarkId'], selectionDigest=manifest['selectionDigest'],
        evaluatorManifestSHA256=frozen['evaluatorManifestSHA256'],
        geometryBindings=[geometry_binding(obj) for obj in manifest['objects']],
        freezeSHA256=digest(output / 'freeze.json'),
        trialHashes={path: digest(output / path) for obj in reports for path in obj['trials'].values()},
        evaluatorOpenedDuringInference=False)
    if manifest.get('resourceStorage'):
        seal.update(resourceStorage=manifest['resourceStorage'], datasetName='HB', objectIdLabel='HB ID',
                    resourceAccessSHA256=digest(output / 'resource-access.json'), pretrainedExposure='unknown')
    write(output / 'inference-seal.json', seal)


def run_evaluation(dataset, output):
    dataset, output = Path(dataset), Path(output)
    seal = json.loads((output / 'inference-seal.json').read_text(encoding='utf-8'))
    if (output / 'invalidated.json').exists() or digest(output / 'freeze.json') != seal['freezeSHA256']:
        raise ValueError('Inference freeze invalidated')
    if seal.get('resourceStorage'):
        freeze = json.loads((output / 'freeze.json').read_text(encoding='utf-8'))
        if freeze.get('resourceStorage') != seal['resourceStorage']:
            raise ValueError('Compact storage differs from sealed inference freeze')
        if digest(output / 'resource-access.json') != seal['resourceAccessSHA256']:
            raise ValueError('Inference resource access log changed after seal')
    for path, sha in seal['trialHashes'].items():
        if digest(output / path) != sha:
            raise ValueError('Inference output changed after seal')
    if digest(dataset / 'evaluator.json') != seal['evaluatorManifestSHA256']:
        raise ValueError('Evaluator annotations changed after selection freeze')
    evaluator = json.loads((dataset / 'evaluator.json').read_text(encoding='utf-8'))
    validate_evaluator_binding(evaluator, seal)
    if evaluator.get('resourceStorage') != seal.get('resourceStorage'):
        raise ValueError('Evaluator storage differs from inference seal')
    inputs = provider(dataset, evaluator, 'evaluation')
    guard = CompactOutputBudget(dataset, output) if isinstance(inputs, ResourceStore) else None
    write = guard.write_json if guard else write_json
    corpus = 'HB' if seal.get('resourceStorage') else 'YCB-Video'
    if corpus == 'HB' and (seal['dataset'] != 'BOP HB val_primesense' or evaluator.get('dataset') != seal['dataset']):
        raise ValueError('HB sealed corpus label mismatch')
    outputs = {obj['objectId']: obj for obj in seal['objects']}
    evaluator_hashes = {name: digest(ROOT / name) for name in (
        'bench/prototype_benchmark.py', 'bench/video_experiment.py', 'bench/tracker.py',
        'bench/model.py', 'bench/ply_model.py', 'bench/renderer.py', 'bench/vision.py',
        'bench/prototype_resource_store.py', 'bench/prototype_output_budget.py')}
    evaluator_manifest_sha = digest(dataset / 'evaluator.json')
    reports = []
    for obj in evaluator['objects']:
        row = dict(objectId=obj['objectId'], name=obj['name'], trials={})
        for name, path in outputs[obj['objectId']]['trials'].items():
            trial = json.loads((output / path).read_text(encoding='utf-8'))
            row['trials'][name] = evaluate_object(inputs, obj, trial['frames'], corpus)
            if seal['evidenceKind'] == 'synthetic_control':
                row['trials'][name]['provenance'] = 'Synthetic generated geometry/depth control; no real-object or human accuracy claim'
            if name == 'interrupted':
                row['trials'][name].update(recovery_metrics(obj['frames'], trial['frames'],
                    row['trials'][name]['perFrameAttachment']))
                row['trials'][name]['recoveryGatePassed'] &= row['trials'][name]['attachmentGatePassed']
        reports.append(row)
        write(output / 'evaluation-progress.json', {'objects': reports})
    report = dict(schemaVersion=1, objects=reports, attemptedObjects=len(reports),
        dataset=seal['dataset'], evidenceKind=seal['evidenceKind'],
        benchmarkMode=seal.get('benchmarkMode', 'full'), plannedObjects=seal.get('plannedObjects', len(reports)),
        realUnseenObjects=len(reports) if seal['evidenceKind'] == 'dataset_reference' else 0,
        passingObjects=sum(r['trials']['natural']['attachmentGatePassed'] for r in reports),
        tenObjectsEvaluated=len(reports) >= 10 and seal['evidenceKind'] == 'dataset_reference', independentHumanAccuracy='blocked',
        physicalPhoneValidation='blocked', liveCamera30FPS='unverified',
        inferenceSealSHA256=digest(output / 'inference-seal.json'),
        evaluatorManifestSHA256=digest(dataset / 'evaluator.json'))
    report['benchmarkGatePassed'] = bool(report['tenObjectsEvaluated']
        and report['passingObjects'] == report['attemptedObjects']
        and all(r['trials']['interrupted']['recoveryGatePassed'] for r in reports))
    if (any(digest(ROOT / name) != sha for name, sha in evaluator_hashes.items())
            or digest(dataset / 'evaluator.json') != evaluator_manifest_sha):
        raise ValueError('Evaluator source or labels changed during scoring')
    report['evaluatorSourceHashes'] = evaluator_hashes
    if isinstance(inputs, ResourceStore):
        inputs.assert_sealed()
        report.update(datasetName='HB', objectIdLabel='HB ID', pretrainedExposure='unknown',
            resourceStorage=seal['resourceStorage'], benchmarkGateEligible=True,
            provenance='HB initialized tracking transfer; dataset-reference labels; pretrained exposure unknown; no independent human truth.')
    write(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['infer', 'evaluate'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'infer':
        run_inference(args.dataset, args.output)
    else:
        run_evaluation(args.dataset, args.output)


if __name__ == '__main__':
    main()
