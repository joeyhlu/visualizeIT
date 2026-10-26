"""Portable, variable-length CPU regression benchmark for production rigid trackers.

Inference opens only inference.json; evaluation is a separate sealed-output stage.
Ten annotated RGB-D onboarding frames are explicitly permitted. No evaluation
pose, mask, depth, visibility or stress label is supplied to a tracker update.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .prototype_benchmark import digest, write_json, inference_packet, build_bank
from .prototype_tracking import PrototypeTrackerV1
from .tracker import RigidTracker
from .ply_model import read_ply
from .vision import cv2

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_COMMIT = 'df788084e1e0eb79950fe81cf87b603044c1a2d5'
HISTORICAL_SHA256 = '0d619def904edaf68ffb35e2bedee45b1dcafdb3d2053a4af59be7d40a5d89d1'
SYSTEMS = ('original_rigid', 'current_rigid', 'current_prototype')
FORBIDDEN = {'evaluationPose', 'evaluationInfo', 'evaluationMask', 'depth',
             'onboardingMask', 'pose', 'mask', 'stress', 'visibility'}


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_path(root, spec):
    root = Path(root).resolve()
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256'}:
        raise ValueError('Resource must specify exactly path and sha256')
    path = (root / spec['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or escaping resource: ' + str(spec['path']))
    if digest(path) != spec['sha256']:
        raise ValueError('Resource hash mismatch: ' + spec['path'])
    return path


def validate_manifest(manifest):
    """Reject annotation leakage, empty/duplicate streams and reordered frames."""
    if manifest.get('schemaVersion') != 1 or not manifest.get('objects'):
        raise ValueError('Expected nonempty schemaVersion 1 inference manifest')
    identities = [obj['objectId'] for obj in manifest['objects']]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate object/sequence identity')
    for obj in manifest['objects']:
        if len(obj['onboarding']) != 10 or not obj['frames']:
            raise ValueError('Exactly ten onboarding and at least one scored frame required')
        records = obj['onboarding'] + obj['frames']
        ids = [record['frameId'] for record in records]
        if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
            raise ValueError('Frame IDs must be integers')
        if any(b <= a for a, b in zip(ids, ids[1:])):
            raise ValueError('Frames must be strictly ordered and disjoint')
        for record in obj['frames']:
            if FORBIDDEN.intersection(record):
                raise ValueError('Evaluator labels in inference frame')
            if set(record) - {'frameId', 'logicalIndex', 'timestampKind', 'timestampSeconds',
                              'rgb', 'calibration', 'width', 'height', 'sessionId'}:
                raise ValueError('Unknown inference frame fields')
        for record in records:
            k = np.asarray(record['calibration']['cam_K'], dtype=float).reshape(3, 3)
            if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
                raise ValueError('Invalid camera intrinsics')


def load_rgb(root, record):
    with Image.open(checked_path(root, record['rgb'])) as image:
        return np.asarray(image.convert('RGB'))


def build_depth_bank(root, obj):
    """Shared fixed bank from ONLY the ten authorized masked RGB-D poses.

    Backprojection avoids rendering full scanned meshes during CPU setup. This is
    an explicitly different initialization adapter, not a production tracker edit.
    Every tracker receives byte-identical anchors/descriptors/last-frame seeds.
    """
    detector = cv2.ORB_create(nfeatures=2000)
    points, descriptors = [], []
    last_pixels, last_points = [], []
    for record in obj['onboarding']:
        rgb = load_rgb(root, record)
        with Image.open(checked_path(root, record['depth'])) as image:
            depth = np.asarray(image).astype(float) * float(record['calibration']['depth_scale']) * .001
        with Image.open(checked_path(root, record['onboardingMask'])) as image:
            mask = np.asarray(image) > 0
        if depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
            raise ValueError('Onboarding RGB/depth/mask dimensions differ')
        # Exclude mixed foreground/background depth at segmentation boundaries.
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        mask &= np.isfinite(depth) & (depth > .03)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, desc = detector.detectAndCompute(gray, mask.astype(np.uint8) * 255)
        k = np.asarray(record['calibration']['cam_K'], dtype=float).reshape(3, 3)
        pose = record['evaluationPose']
        rotation = np.asarray(pose['cam_R_m2c'], dtype=float).reshape(3, 3)
        translation = np.asarray(pose['cam_t_m2c'], dtype=float) * .001
        last_pixels, last_points = [], []
        if desc is None:
            continue
        # Fixed predeclared CPU initialization budget, not selected using scores.
        # Keep the detector's first 300 features per onboarding view (response
        # ordering is deterministic in the pinned OpenCV); at most 3,000 anchors.
        for feature, descriptor in list(zip(keypoints, desc))[:300]:
            x, y = feature.pt
            px, py = int(np.floor(x + .5)), int(np.floor(y + .5))
            if not (0 <= py < depth.shape[0] and 0 <= px < depth.shape[1]) or not mask[py, px]:
                continue
            camera = np.linalg.solve(k, [x, y, 1.]) * depth[py, px]
            point = (camera - translation) @ rotation
            points.append(point)
            descriptors.append(descriptor)
            last_pixels.append([x, y])
            last_points.append(point)
    if len(points) < 12 or len(last_points) < 12:
        raise ValueError('insufficient_onboarding_references')
    return (np.asarray(points, dtype=np.float64), np.asarray(descriptors, dtype=np.uint8),
            gray, np.asarray(last_pixels, dtype=np.float64), np.asarray(last_points, dtype=np.float64))


def historical_tracker():
    """Load the exact repository implementation, not a rewritten approximation."""
    result = subprocess.run(['git', 'show', f'{HISTORICAL_COMMIT}:bench/tracker.py'],
                            cwd=ROOT, check=True, capture_output=True)
    if hashlib.sha256(result.stdout).hexdigest() != HISTORICAL_SHA256:
        raise ValueError('Historical tracker source hash differs')
    directory = ROOT / '.cache' / 'tracking-benchmark' / 'historical'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{HISTORICAL_COMMIT}.py'
    path.write_bytes(result.stdout)
    name = 'bench._tracking_benchmark_original'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.RigidTracker


def make_tracker(system, bank, model, obj, seed_rgb, session):
    points, descriptors, gray, pixels, seed_points = bank
    if system == 'current_prototype':
        packet = inference_packet(obj['onboarding'][-1], seed_rgb, session, obj['objectId'], session,
                                  obj['onboarding'][-1].get('logicalIndex', 9))
        return PrototypeTrackerV1(points, descriptors, model, session_id=session,
            object_id=obj['objectId'], reference_id=session, onboarding_seed=packet,
            seed_pixels=pixels, seed_points=seed_points)
    cls = historical_tracker() if system == 'original_rigid' else RigidTracker
    return cls(points.copy(), descriptors.copy(), model.copy(), gray.copy(), pixels.copy(), seed_points.copy())


def update_tracker(system, tracker, record, rgb, obj, session, index):
    if system == 'current_prototype':
        packet = inference_packet(record, rgb, session, obj['objectId'], session,
                                  record.get('logicalIndex', index + 10))
        row = tracker.update(packet).manifest()
        row['cvCameraFromSourceObject'] = row.pop('cameraFromObject')
        return row
    k = np.asarray(record['calibration']['cam_K'], dtype=float).reshape(3, 3)
    return tracker.update(rgb, k, record['frameId']).manifest()


def source_hashes():
    return {name: digest(ROOT / name) for name in (
        'bench/tracking_benchmark.py', 'bench/tracker.py', 'bench/prototype_tracking.py',
        'bench/vision.py', 'bench/prototype_benchmark.py', 'bench/ply_model.py',
        'bench/renderer.py', 'bench/model.py', 'bench/obj_model.py',
        'bench/tracking_production_banks.py')}


def hardware():
    import PIL
    model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    memory = next((line.split(':', 1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemTotal:')), None) if Path('/proc/meminfo').exists() else None
    return dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__,
        platform=platform.platform(), cpuModel=model, logicalCPUs=os.cpu_count(), memoryTotal=memory,
        opencvThreads=cv2.getNumThreads(), gpuUsed=False,
        note='Shared CPU VM; elapsed time is environment-specific, not live-phone FPS')


def run_inference(dataset, output, systems=SYSTEMS, repeats=3, warmup=20, limit=None,
                  initialization='depth_budget', max_scored_frames=None, bank_cache=None):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Choose a fresh output directory; previous trials are immutable')
    if (repeats < 1 or warmup < 0 or not systems or set(systems) - set(SYSTEMS)
            or len(systems) != len(set(systems)) or (limit is not None and limit < 1)
            or initialization not in ('depth_budget', 'production_mesh')
            or (max_scored_frames is not None and max_scored_frames < 1)
            or (bank_cache is not None and initialization != 'production_mesh')):
        raise ValueError('Invalid benchmark configuration')
    manifest_path = dataset / 'inference.json'
    manifest = json_read(manifest_path)
    validate_manifest(manifest)
    objects = manifest['objects'][:limit] if limit else manifest['objects']
    if max_scored_frames is not None:
        objects = [dict(obj, frames=obj['frames'][:max_scored_frames]) for obj in objects]
    cv2.setNumThreads(1)
    cv2.setRNGSeed(17)
    output.mkdir(parents=True)
    frozen = dict(schemaVersion=1, benchmarkId=manifest['benchmarkId'],
        dataset=manifest['dataset'], evidenceKind=manifest.get('evidenceKind'),
        inferenceSHA256=digest(manifest_path), evaluatorSHA256=manifest['evaluatorManifestSHA256'],
        sourceHashes=source_hashes(), historicalCommit=HISTORICAL_COMMIT,
        historicalTrackerSHA256=HISTORICAL_SHA256, hardware=hardware(),
        configuration=dict(systems=list(systems), repeats=repeats, warmupFrames=warmup,
            initializationMode=initialization,
            initialization=('ten annotated RGB-D/pose frames; exact production prototype_benchmark.build_bank mesh-ray intersections; ORB nfeatures=2000 per view, no extra cap'
                if initialization == 'production_mesh' else
                'SIMPLIFIED initialization: ten annotated masked RGB-D frames; ORB depth-backprojected immutable bank; first 300 detector features per view, <=3000 anchors'),
            runtime='serial systems and sequences; OpenCV one thread; no detector/neural fallback',
            thresholds=dict(accurateReprojectionPixels=5., visibleFraction=.5,
                            addFractionOfSampledBoundingBoxDiagonal=.1), limitSequences=limit,
            maxScoredFrames=max_scored_frames, bankCacheEnabled=bank_cache is not None,
            bankCachePolicy='Hash-bound exact production builder outputs; preparation does not overlap measured tracker timing; original build seconds recorded separately' if bank_cache else None),
        evaluatorOpenedDuringInference=False)
    write_json(output / 'freeze.json', frozen)
    results = []
    for obj in objects:
        sequence = dict(objectId=obj['objectId'], name=obj['name'], trials=[])
        sequence_start = time.perf_counter()
        try:
            start = time.perf_counter()
            scan, model, transform = read_ply(checked_path(dataset, obj['model']), obj['name'])
            if initialization == 'production_mesh':
                if bank_cache is not None:
                    from .tracking_production_banks import load_bank
                    entry = load_bank(dataset, bank_cache, obj['objectId'])
                    if entry.metadata['helperSourceSHA256'] != digest(ROOT / 'bench/tracking_production_banks.py'):
                        raise ValueError('Cached production-bank helper source differs')
                    sequence['bankCacheMetadataSHA256'] = entry.metadata['metadataSHA256']
                    sequence['bankCacheStatus'] = entry.metadata['status']
                    sequence['bankSourceBuildSeconds'] = entry.metadata['sourceBuildSeconds']
                    if entry.arrays is None:
                        raise ValueError('cached_exact_production_builder_failure: ' + entry.metadata['failure'])
                    bank = entry.arrays
                else:
                    bank = build_bank(dataset, obj['onboarding'], scan.original, transform)
                if len(bank[0]) < 12:
                    raise ValueError('insufficient_onboarding_references')
            else:
                bank = build_depth_bank(dataset, obj)
            sequence['initializationSeconds'] = time.perf_counter() - start
            sequence['anchors'] = len(bank[0])
            sequence['seedViewAnchors'] = len(bank[3])
            sequence['bankSHA256'] = hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in bank)).hexdigest()
            bank_relative = f'object-{obj["objectId"]:06d}/reference-bank.npz'
            bank_path = output / bank_relative
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(bank_path, referencePoints=bank[0], referenceDescriptors=bank[1],
                                previousGray=bank[2], seedPixels=bank[3], seedPoints=bank[4])
            sequence['bankArtifact'] = dict(path=bank_relative, sha256=digest(bank_path))
            seed_rgb = load_rgb(dataset, obj['onboarding'][-1])
        except Exception as exc:
            sequence['initializationFailure'] = f'{type(exc).__name__}: {exc}'
            bank = None
        for system in systems:
            for repeat in range(repeats):
                session = f'{obj["objectId"]}-{system}-{repeat}'
                rows, durations, decode_durations = [], [], []
                setup_failure = sequence.get('initializationFailure')
                if bank is not None:
                    # A separate disposable stream warms the same update path; none
                    # of this stream's state/poses enter the measured trial.
                    try:
                        warm = make_tracker(system, bank, model, obj, seed_rgb, session + '-warmup')
                        for index, record in enumerate(obj['frames'][:warmup]):
                            update_tracker(system, warm, record, load_rgb(dataset, record), obj, session + '-warmup', index)
                        tracker = make_tracker(system, bank, model, obj, seed_rgb, session)
                    except Exception as exc:
                        setup_failure = f'trial_setup_unavailable:{type(exc).__name__}:{exc}'
                start = time.perf_counter()
                for index, record in enumerate(obj['frames']):
                    attempted = False
                    if setup_failure is not None:
                        row = dict(frameId=record['frameId'], state='lost',
                            failureReason=setup_failure, cvCameraFromSourceObject=None)
                        decode, elapsed = 0., 0.
                    else:
                        begin_decode = time.perf_counter()
                        try:
                            rgb = load_rgb(dataset, record)
                            decode = time.perf_counter() - begin_decode
                            begin = time.perf_counter()
                            attempted = True
                            row = update_tracker(system, tracker, record, rgb, obj, session, index)
                        except Exception as exc:
                            # Preserve the failure and denominator; never drop a hard frame.
                            row = dict(frameId=record['frameId'], state='lost',
                                failureReason=f'exception:{type(exc).__name__}:{exc}', cvCameraFromSourceObject=None)
                            if hasattr(tracker, 'reset_temporal'):
                                tracker.reset_temporal()
                            else:
                                tracker.previous_gray = tracker.previous_pixels = tracker.previous_points = None
                                tracker.lost = True
                        if attempted:
                            elapsed = time.perf_counter() - begin
                        else:
                            elapsed, decode = 0., time.perf_counter() - begin_decode
                    row['processingSeconds'], row['decodeAndHashSeconds'], row['updateAttempted'] = elapsed, decode, attempted
                    rows.append(row)
                    if attempted:
                        durations.append(elapsed)
                    decode_durations.append(decode)
                relative = f'object-{obj["objectId"]:06d}/{system}-r{repeat}.json'
                trial = dict(system=system, repeat=repeat, frames=rows,
                    timedUpdateFrames=len(durations), setupFailure=setup_failure,
                    updateMedianSeconds=float(np.median(durations)) if durations else None,
                    updateP95Seconds=float(np.percentile(durations, 95)) if durations else None,
                    updateTotalSeconds=sum(durations), decodeAndHashTotalSeconds=sum(decode_durations),
                    replayWallSeconds=time.perf_counter() - start,
                    timingScope='CPU RGB preprocessing, correspondence search/flow, 6D pose and stream wrapper; not detector-only; replay also includes decode/hash; excludes onboarding, warmup, evaluation and rendering')
                write_json(output / relative, trial)
                sequence['trials'].append(relative)
                print(json.dumps(dict(sequence=obj['name'], system=system, repeat=repeat,
                    frames=len(rows), tracking=sum(r['state'] == 'tracking' for r in rows),
                    medianUpdateMs=1000 * trial['updateMedianSeconds'] if durations else None)), flush=True)
        sequence['totalSeconds'] = time.perf_counter() - sequence_start
        results.append(sequence)
        write_json(output / 'progress.json', dict(sequences=results))
    if source_hashes() != frozen['sourceHashes'] or digest(manifest_path) != frozen['inferenceSHA256']:
        raise ValueError('Source/input changed during inference; output not sealed')
    seal = dict(**frozen, sequences=results, freezeSHA256=digest(output / 'freeze.json'),
        trialHashes={path: digest(output / path) for obj in results for path in obj['trials']},
        geometryBindings=[dict(objectId=o['objectId'], model=o['model'],
            frames=[dict(frameId=r['frameId'], rgb=r['rgb'], calibration=r['calibration']) for r in o['frames']])
            for o in objects])
    write_json(output / 'inference-seal.json', seal)


def project(points, rotation, translation, k):
    camera = points @ rotation.T + translation
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .03):
        raise ValueError('Pose projects model behind near plane')
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def score_rows(records, rows, points):
    if [r['frameId'] for r in records] != [r['frameId'] for r in rows]:
        raise ValueError('Missing, extra or reordered prediction frame')
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    errors, additions, rotations, translations, details = [], [], [], [], []
    eligible = tracked = eligible_tracked = accurate = false_public = 0
    longest, current = 0, 0
    for record, row in zip(records, rows):
        visible = float(record['evaluationInfo']['visib_fract']) >= .5
        hidden = float(record['evaluationInfo']['visib_fract']) == 0
        eligible += visible
        public = row['state'] == 'tracking'
        tracked += public; eligible_tracked += public and visible; false_public += public and hidden
        detail = dict(frameId=row['frameId'], visible=visible, public=public,
                      stress=record.get('stress'), accurate=False)
        matrix = row.get('cvCameraFromSourceObject')
        if not public and matrix is not None:
            raise ValueError('Nontracking state exposes stale pose')
        if public:
            matrix = np.asarray(matrix, dtype=float)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
                raise ValueError('Public output is not a proper metre SE(3) pose')
            pose = record['evaluationPose']
            r = np.asarray(pose['cam_R_m2c'], dtype=float).reshape(3, 3)
            t = np.asarray(pose['cam_t_m2c'], dtype=float) * .001
            k = np.asarray(record['calibration']['cam_K'], dtype=float).reshape(3, 3)
            gt = points @ r.T + t
            estimated = points @ matrix[:3, :3].T + matrix[:3, 3]
            add = float(np.mean(np.linalg.norm(estimated - gt, axis=1)))
            error = float(np.mean(np.linalg.norm(project(points, matrix[:3, :3], matrix[:3, 3], k)
                                                  - project(points, r, t, k), axis=1)))
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3] @ r.T) - 1) / 2, -1, 1))))
            translation = float(np.linalg.norm(matrix[:3, 3] - t))
            good = bool(visible and error < 5.)
            accurate += good
            detail.update(reprojectionPixels=error, addMetres=add, rotationDegrees=rotation,
                          translationMetres=translation, accurate=good, addPass=add < .1 * diameter)
            errors.append(error); additions.append(add); rotations.append(rotation); translations.append(translation)
        current = current + 1 if detail['accurate'] else 0
        longest = max(longest, current)
        details.append(detail)
    # Recovery follows a label-defined hidden interval, not an inserted evaluator
    # perturbation. It requires two consecutive visible, accurate public outputs.
    recovery = []
    for i in range(1, len(records)):
        if records[i-1]['evaluationInfo']['visib_fract'] == 0 and records[i]['evaluationInfo']['visib_fract'] > 0:
            observable = 0; previous = False; delay = None
            for j, detail in enumerate(details[i:], start=i):
                if records[j]['evaluationInfo']['visib_fract'] == 0:
                    break
                if not detail['visible']:
                    previous = False
                    continue
                observable += 1
                if previous and detail['accurate']:
                    delay = observable
                    break
                previous = detail['accurate']
            recovery.append(dict(returnFrameId=records[i]['frameId'], observableFramesToTwoAccurate=delay))
    summary = dict(frames=len(rows), publicTrackingFrames=tracked, visibleFrames=eligible,
        visibleTrackingFrames=eligible_tracked, accurateVisibleFrames=accurate,
        trackingAvailability=tracked / len(rows),
        visibleTrackingAvailability=eligible_tracked / eligible if eligible else None,
        accurateVisibleAvailability=accurate / eligible if eligible else None,
        hiddenPublicPoseFrames=false_public, longestAccurateVisibleRun=longest,
        failures=dict(Counter(r.get('failureReason', r.get('reason')) for r in rows if r['state'] != 'tracking')),
        recovery=recovery, modelBoundingBoxDiagonalMetres=diameter,
        boundingBoxScope='deterministic sampled vertices, not true/full model diameter or official BOP ADD threshold',
        metricScope='Dataset-reference full-model corresponding-point pose errors; no symmetry matching, no independently annotated physical attachment or live-camera claim')
    for name, values in [('reprojectionPixels', errors), ('addMetres', additions),
                         ('rotationDegrees', rotations), ('translationMetres', translations)]:
        summary[name + 'Median'] = float(np.median(values)) if values else None
        summary[name + 'P95'] = float(np.percentile(values, 95)) if values else None
    return summary, details


def run_evaluation(dataset, output):
    dataset, output = Path(dataset), Path(output)
    seal = json_read(output / 'inference-seal.json')
    if source_hashes() != seal['sourceHashes']:
        raise ValueError('Benchmark source differs from frozen inference source')
    if digest(output / 'freeze.json') != seal['freezeSHA256']:
        raise ValueError('Freeze tampered')
    for path, expected in seal['trialHashes'].items():
        if digest(output / path) != expected:
            raise ValueError('Prediction tampered after inference')
    if digest(dataset / 'evaluator.json') != seal['evaluatorSHA256']:
        raise ValueError('Evaluator differs from frozen inference binding')
    evaluator = json_read(dataset / 'evaluator.json')
    if evaluator['benchmarkId'] != seal['benchmarkId']:
        raise ValueError('Evaluator benchmark identity differs')
    by_id = {o['objectId']: o for o in evaluator['objects']}
    if len(by_id) != len(evaluator['objects']):
        raise ValueError('Duplicate evaluator sequence identity')
    reports = []
    for binding in seal['geometryBindings']:
        obj = by_id[binding['objectId']]
        if seal['configuration'].get('maxScoredFrames') is not None:
            obj = dict(obj, frames=obj['frames'][:seal['configuration']['maxScoredFrames']])
        joined = dict(objectId=obj['objectId'], model=obj['model'],
            frames=[dict(frameId=r['frameId'], rgb=r['rgb'], calibration=r['calibration']) for r in obj['frames']])
        if joined != binding:
            raise ValueError('Evaluator geometry/RGB/calibration binding differs')
        _, model, _ = read_ply(checked_path(dataset, obj['model']), obj['name'])
        # Deterministic geometry-only sample, same for every system and repeat.
        points = model[np.linspace(0, len(model) - 1, min(512, len(model)), dtype=int)]
        source = next(o for o in seal['sequences'] if o['objectId'] == obj['objectId'])
        for path in source['trials']:
            trial = json_read(output / path)
            summary, details = score_rows(obj['frames'], trial['frames'], points)
            report = dict(objectId=obj['objectId'], name=obj['name'], system=trial['system'],
                          repeat=trial['repeat'], split=obj.get('split'), tags=obj.get('tags'), **summary,
                          updateMedianSeconds=trial['updateMedianSeconds'], updateP95Seconds=trial['updateP95Seconds'],
                          timedUpdateFrames=trial['timedUpdateFrames'], setupFailure=trial['setupFailure'],
                          updateTotalSeconds=trial['updateTotalSeconds'], replayWallSeconds=trial['replayWallSeconds'])
            reports.append(report)
            write_json(output / path.replace('.json', '-scored.json'), dict(summary=report, frames=details))
    aggregate = {}
    for system in seal['configuration']['systems']:
        trials = [r for r in reports if r['system'] == system]
        total = sum(r['frames'] for r in trials)
        visible = sum(r['visibleFrames'] for r in trials)
        timed = sum(r['timedUpdateFrames'] for r in trials)
        median_updates = [1000 * r['updateMedianSeconds'] for r in trials if r['updateMedianSeconds'] is not None]
        aggregate[system] = dict(sequenceTrials=len(trials), framesWithRepeats=total,
            uniqueFrames=total // seal['configuration']['repeats'],
            trackingAvailability=sum(r['publicTrackingFrames'] for r in trials) / total,
            visibleTrackingAvailability=sum(r['visibleTrackingFrames'] for r in trials) / visible if visible else None,
            accurateVisibleAvailability=sum(r['accurateVisibleFrames'] for r in trials) / visible if visible else None,
            hiddenPublicPoseFrames=sum(r['hiddenPublicPoseFrames'] for r in trials),
            timedUpdateFrames=timed, skippedUpdateFrames=total-timed,
            updateFramesPerSecond=timed / sum(r['updateTotalSeconds'] for r in trials) if sum(r['updateTotalSeconds'] for r in trials) else None,
            replayFramesPerSecond=timed / sum(r['replayWallSeconds'] for r in trials if r['timedUpdateFrames']) if timed else None,
            perSequenceMedianUpdateMs=float(np.median(median_updates)) if median_updates else None,
            initializationFailureSequences=sum('initializationFailure' in r for r in seal['sequences']),
            recoveryEvents=[e for r in trials for e in r['recovery']])
    result = dict(schemaVersion=1, benchmarkId=seal['benchmarkId'], dataset=seal['dataset'],
        evidenceKind=seal['evidenceKind'], inferenceSealSHA256=digest(output / 'inference-seal.json'),
        aggregate=aggregate, sequenceTrials=reports,
        caveats=['Same masked RGB-D/pose initialization for all systems; not automatic object discovery',
                 'No independent human accuracy labels; reference poses may contain error',
                 'Corresponding-point errors do not quotient object symmetries',
                 'CPU prerecorded replay, not live camera or neural GoTrack performance'])
    write_json(output / 'results.json', result)
    print(json.dumps(aggregate, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('infer', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True, type=Path)
        p.add_argument('--output', required=True, type=Path)
        if name == 'infer':
            p.add_argument('--systems', nargs='+', choices=SYSTEMS, default=list(SYSTEMS))
            p.add_argument('--repeats', type=int, default=3)
            p.add_argument('--warmup', type=int, default=20)
            p.add_argument('--limit-sequences', type=int)
            p.add_argument('--initialization', choices=('depth_budget', 'production_mesh'),
                           default='depth_budget')
            p.add_argument('--max-scored-frames', type=int)
            p.add_argument('--bank-cache', type=Path,
                           help='Verified exact production banks, prepared before timing; never used for simplified mode')
    args = parser.parse_args()
    if args.stage == 'infer':
        run_inference(args.dataset, args.output, args.systems, args.repeats, args.warmup,
                      args.limit_sequences, args.initialization, args.max_scored_frames, args.bank_cache)
    else:
        run_evaluation(args.dataset, args.output)


if __name__ == '__main__':
    main()
