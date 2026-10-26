#!/usr/bin/env python3
"""Independently audit existing HB tracking inputs; never run a tracker/model.

Reads cached public HB data, an existing depth-bank service asset, and sealed
benchmark results. Uses scalar BOP projection math and sparse ASCII PLY samples,
not the benchmark's projection/mesh helpers or a full renderer. No downloads,
training, model inference, production edits, or dataset modifications occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FRAME_IDS = (0, 9, 10, 159, 309)
EXPECTED_SCENES = ('000001', '000002', '000003', '000004')
SCORED_KEYS = {'calibration', 'frameId', 'logicalIndex', 'rgb', 'timestampKind', 'timestampSeconds'}
BOP_FORMAT = 'https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md'
LIMITATIONS = [
    'Source/geometry/byte audit, not independent human pose ground truth or exhaustive physical sensor calibration.',
    'Twenty frame samples from one object per four scenes; not exhaustive per-object geometry validation.',
    'Projected convex hull inclusion is containment, not an occlusion-aware silhouette IoU.',
    'Front-facing sparse model depth check has no z-buffer and includes self-occluded/concave surface samples.',
    'Pixel/object roundtrips check implementation consistency; alone they do not establish correct calibration.',
    'Six hundred actual saved depth-bank anchors checked for HB object20 only; no detector/tracker rerun.',
    'Custom first-ten masked RGB-D/known-pose initialization differs from the original rendered-model bank and is not the full discovery/segmentation/neural pipeline.',
    'Observed failures do not causally separate initialization adapter, texture, depth noise, correspondence logic, or tracker weaknesses.',
    'No GoTrack quality, live-camera FPS, phone/device readiness, or official BOP challenge score is measured.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def display_path(path):
    """Do not serialize machine-specific absolute paths into public evidence."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def checked_resource(dataset, spec, acquisition):
    require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'Malformed resource reference')
    path = (dataset / spec['path']).resolve()
    require(path.is_relative_to(dataset) and path.is_file(), 'Resource escapes/misses dataset root')
    require(sha(path) == spec['sha256'] == acquisition[spec['path']], 'Resource differs from acquisition lock')
    return path


def sample_ply(path, maximum=4096):
    """Independent bounded ASCII vertex sampler; triangle payload is not parsed."""
    with path.open(encoding='ascii') as stream:
        require(stream.readline().strip() == 'ply', 'Expected PLY file')
        require(stream.readline().strip() == 'format ascii 1.0', 'Expected ASCII PLY')
        names, count, current = [], 0, None
        for line in stream:
            tokens = line.split()
            if tokens[:2] == ['element', 'vertex']:
                count, current = int(tokens[2]), 'vertex'
            elif tokens and tokens[0] == 'element':
                current = tokens[1]
            elif tokens and tokens[0] == 'property' and current == 'vertex':
                names.append(tokens[-1])
            elif tokens and tokens[0] == 'end_header':
                break
        require(4 <= count <= 1000000, 'Unexpected PLY vertex count')
        keep = set(np.linspace(0, count - 1, min(maximum, count), dtype=int).tolist())
        columns = [names.index(key) for key in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
        values, indices = [], []
        for index in range(count):
            line = stream.readline()
            require(bool(line), 'Truncated PLY vertex payload')
            if index in keep:
                row = line.split()
                values.append([float(row[column]) for column in columns])
                indices.append(index)
    values = np.asarray(values)
    require(np.isfinite(values).all(), 'Nonfinite PLY vertex sample')
    return values[:, :3], values[:, 3:6], np.asarray(indices), count


def image_array(path, *, rgb=False):
    with Image.open(path) as image:
        require(image.size == (640, 480), 'HB sample image was resized or has unexpected dimensions')
        return np.asarray(image.convert('RGB') if rgb else image).copy()


def audit_sample(dataset, obj, record, info, instance, points, normals, vertices, acquisition):
    frame_id = record['frameId']
    onboarding = frame_id < 10
    mask_key = 'onboardingMask' if onboarding else 'evaluationMask'
    rgb = image_array(checked_resource(dataset, record['rgb'], acquisition), rgb=True)
    depth = image_array(checked_resource(dataset, record['depth'], acquisition))
    mask = image_array(checked_resource(dataset, record[mask_key], acquisition)) > 0
    k = np.asarray(record['calibration']['cam_K']).reshape(3, 3)
    rotation = np.asarray(record['evaluationPose']['cam_R_m2c']).reshape(3, 3)
    translation = np.asarray(record['evaluationPose']['cam_t_m2c'])
    scale = record['calibration']['depth_scale']
    camera = points @ rotation.T + translation  # All BOP quantities here are millimetres.
    require(np.all(camera[:, 2] > 30), 'Sampled GT model crosses the near plane')
    pixels = np.column_stack([k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2],
                              k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2]])
    metres = (points * .001) @ rotation.T + translation * .001
    homogeneous = metres @ k.T
    metre_pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
    hull = cv2.convexHull(pixels.astype(np.float32))
    hull_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull_mask, np.rint(hull.reshape(-1, 2)).astype(np.int32), 1)
    yy, xx = np.where(mask)
    require(len(xx) > 0, 'Chosen audit sample has no visible object pixels')
    integer_pixels = np.floor(pixels + .5).astype(int)
    frontal = np.einsum('ij,ij->i', normals @ rotation.T, camera) < 0
    valid = ((integer_pixels[:, 0] >= 0) & (integer_pixels[:, 0] < 640)
             & (integer_pixels[:, 1] >= 0) & (integer_pixels[:, 1] < 480) & frontal)
    projected, z = integer_pixels[valid], camera[valid, 2]
    chosen = mask[projected[:, 1], projected[:, 0]] & (depth[projected[:, 1], projected[:, 0]] > 0)
    residual = np.abs(depth[projected[chosen, 1], projected[chosen, 0]] * scale - z[chosen])
    require(len(residual) > 0, 'No valid masked frontal model/depth pairs')
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    y, x = np.where(interior & (depth > 30))
    require(len(x) > 0, 'No valid interior depth sample')
    keep = np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int)
    x, y = x[keep], y[keep]
    measured_z = depth[y, x] * scale
    measured_camera = np.column_stack([(x - k[0, 2]) * measured_z / k[0, 0],
                                      (y - k[1, 2]) * measured_z / k[1, 1], measured_z])
    object_points = (measured_camera - translation) @ rotation
    forward = object_points @ rotation.T + translation
    roundtrip = np.column_stack([k[0, 0] * forward[:, 0] / forward[:, 2] + k[0, 2],
                                  k[1, 1] * forward[:, 1] / forward[:, 2] + k[1, 2]])
    nearest = np.sqrt(((object_points[:, None, :] - points[None, :, :]) ** 2).sum(2)).min(1)
    bbox = [float(pixels[:, 0].min()), float(pixels[:, 1].min()),
            float(pixels[:, 0].max()), float(pixels[:, 1].max())]
    reference_info = info[str(frame_id)][instance]
    box = reference_info['bbox_obj']
    reference_bbox = [box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1]
    row = dict(sceneId=obj['sceneId'], objectId=obj['objectId'], frameId=frame_id,
        phase='onboarding' if onboarding else 'evaluation', rgbSHA256=record['rgb']['sha256'],
        modelVertices=vertices, sampledVertices=len(points),
        rawDepthMedianMM=float(np.median(depth[mask & (depth > 0)])), sourcePoseZMM=float(translation[2]),
        modelProjectedBBox=bbox, sourceInfoBBox=reference_bbox,
        visibleMaskBBox=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
        projectedVsInfoBBoxMaxPixels=float(np.max(np.abs(np.asarray(bbox) - reference_bbox))),
        visibleMaskInsideProjectedConvexHull=float(np.mean(hull_mask[yy, xx] > 0)),
        frontFacingMaskedModelPoints=len(residual), frontFacingDepthAbsMedianMM=float(np.median(residual)),
        frontFacingDepthWithin10MM=float(np.mean(residual < 10)),
        independentPixelObjectCameraRoundtripMaxPixels=float(np.max(np.linalg.norm(roundtrip - np.column_stack([x, y]), axis=1))),
        independentMMvsMProjectionMaxPixels=float(np.max(np.abs(metre_pixels - pixels))),
        manualInteriorDepthPoints=len(x), interiorDepthToSampledModelNearestMedianMM=float(np.median(nearest)),
        visibility=float(reference_info['visib_fract']))
    overlay = None
    if frame_id == 10:
        overlay = rgb.copy()
        overlay[mask] = (overlay[mask] * .55 + np.array([0, 255, 80]) * .45).astype(np.uint8)
        cv2.polylines(overlay, [np.rint(hull).astype(np.int32)], True, (255, 40, 0), 1)
        for u, v in integer_pixels[frontal][::4]:
            if 0 <= u < 640 and 0 <= v < 480:
                cv2.circle(overlay, (int(u), int(v)), 1, (255, 0, 255), -1)
    return row, rgb, overlay


def audit_saved_bank(dataset, asset, inference, raw_models):
    metadata = read_json(asset.parent / 'asset.json')
    require(metadata['objectId'] == 20 and metadata['assetSHA256'] == sha(asset), 'Saved asset identity/hash mismatch')
    require(metadata['inferenceManifestSHA256'] == sha(dataset / 'inference.json'), 'Saved bank uses another input corpus')
    require(metadata['initializationAdapter'] == 'shared_masked_depth_bank_first_10_max_300_ORB_anchors_per_view',
            'Audit requires the declared shared depth-bank adapter, not a different initialization')
    require(metadata['onboardingFrameIds'] == list(range(10)), 'Unexpected saved-bank onboarding scope')
    require(asset.stat().st_size <= 256 * 1024**2, 'NPZ asset exceeds bounded read size')
    with zipfile.ZipFile(asset) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 512 * 1024**2, 'NPZ expansion budget exceeded')
    with np.load(asset, allow_pickle=False) as archive:
        model, anchors = archive['modelPositions'], archive['referencePoints']
    require(anchors.shape == (3000, 3) and np.isfinite(anchors).all(), 'Expected ten capped 300-anchor views')
    points, indices = raw_models[20]
    model_error = float(np.max(np.abs(model[indices] - points * .001)))
    obj = next(obj for obj in inference['objects'] if obj['objectId'] == 20)
    checks = []
    for view, source in ((0, anchors[:300]), (9, anchors[-300:])):
        record = obj['onboarding'][view]
        rotation = np.asarray(record['evaluationPose']['cam_R_m2c']).reshape(3, 3)
        translation = np.asarray(record['evaluationPose']['cam_t_m2c']) * .001
        k = np.asarray(record['calibration']['cam_K']).reshape(3, 3)
        camera = source @ rotation.T + translation
        uv = camera @ k.T
        pixels = np.floor(uv[:, :2] / uv[:, 2, None] + .5).astype(int)
        depth = image_array(dataset / record['depth']['path']) * .001
        mask = image_array(dataset / record['onboardingMask']['path']) > 0
        require(np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < 640) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 480)),
                'Saved anchor projects outside the authorized frame')
        delta = np.abs(camera[:, 2] - depth[pixels[:, 1], pixels[:, 0]]) * 1000
        checks.append(dict(objectId=20, onboardingFrameId=view, anchors=len(source),
            projectedAnchorsInsideCorrectVisibleMask=float(np.mean(mask[pixels[:, 1], pixels[:, 0]])),
            depthRoundtripAbsMedianMM=float(np.median(delta)), depthRoundtripAbsMaxMM=float(np.max(delta))))
    return model_error, checks


def audit(dataset, results_path, asset, output):
    started = time.perf_counter()
    dataset, results_path, asset, output = map(lambda p: Path(p).resolve(), (dataset, results_path, asset, output))
    require(not output.is_relative_to(dataset), 'Audit output must not modify the source dataset')
    require(not output.exists() or not any(output.iterdir()), 'Choose a fresh audit output directory')
    cv2.setNumThreads(1)
    script_hash = sha(Path(__file__))
    inference, evaluator, selection, lock = [read_json(dataset / name) for name in
                                           ('inference.json', 'evaluator.json', 'selection.json', 'sources.lock.json')]
    require(sha(dataset / 'inference.json') == lock['inferenceSHA256'], 'Inference acquisition binding mismatch')
    require(sha(dataset / 'evaluator.json') == lock['evaluatorSHA256'] == inference['evaluatorManifestSHA256'],
            'Evaluator acquisition binding mismatch')
    acquisition = {row['member']: row['sha256'] for row in lock['resources']}
    by_id = {obj['objectId']: obj for obj in evaluator['objects']}
    require(len(by_id) == len(evaluator['objects']) == len(inference['objects']) == 12, 'Expected twelve HB object tracks')
    choices = {(obj['sceneId'], obj['objectId']): obj for obj in selection['selectedObjects']}
    scenes = sorted({obj['sceneId'] for obj in inference['objects']})
    require(tuple(scenes) == EXPECTED_SCENES, 'Audit requires all four frozen HB scenes')
    first_by_scene = {scene: next(obj for obj in inference['objects'] if obj['sceneId'] == scene) for scene in scenes}
    manual, contacts, raw_models = [], [], {}
    source_joins = label_free = visible = metadata_hashes = resource_hashes = model_hashes = 0
    shapes, intrinsics, scales = set(), set(), set()
    for obj in inference['objects']:
        scene, object_id = obj['sceneId'], obj['objectId']
        evaluation, choice = by_id[object_id], choices[(scene, object_id)]
        metadata_root = dataset / 'metadata/val_primesense' / scene
        for name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'):
            require(sha(metadata_root / name) == selection['metadataSHA256'][f'val_primesense/{scene}/{name}'],
                    'Original metadata acquisition hash mismatch')
        if obj is first_by_scene[scene]:
            metadata_hashes += 3
        gt, info, camera = [read_json(metadata_root / name) for name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')]
        require([row['frameId'] for row in obj['onboarding']] == list(range(10)), 'Unexpected onboarding frame scope')
        require([row['frameId'] for row in obj['frames']] == list(range(10, 310)), 'Unexpected scored frame scope')
        for index, row in enumerate(obj['onboarding'] + evaluation['frames']):
            frame_id, instance = row['frameId'], choice['instanceIds'][index]
            require(frame_id == choice['frameIds'][index], 'Source frame join mismatch')
            require(row['calibration'] == camera[str(frame_id)], 'Source camera join mismatch')
            require(row['evaluationPose'] == gt[str(frame_id)][instance], 'Source object-pose join mismatch')
            require(row['evaluationPose']['obj_id'] == object_id, 'Wrong source object instance')
            if index >= 10:
                require(row['evaluationInfo'] == info[str(frame_id)][instance], 'Source visibility join mismatch')
            source_joins += 1
        for row, evaluation_row in zip(obj['frames'], evaluation['frames']):
            require(set(row) == SCORED_KEYS, 'Scored inference record contains labels/unsupported fields')
            require(all(row[key] == evaluation_row[key] for key in ('rgb', 'calibration', 'frameId')), 'Inference/evaluator join mismatch')
            label_free += 1
            visible += evaluation_row['evaluationInfo']['visib_fract'] >= .5
        if obj is not first_by_scene[scene]:
            continue
        model_path = checked_resource(dataset, obj['model'], acquisition)
        points, normals, indices, count = sample_ply(model_path)
        model_hashes += 1
        raw_models[object_id] = (points, indices)
        records = {row['frameId']: row for row in obj['onboarding'] + evaluation['frames']}
        for frame_id in SAMPLE_FRAME_IDS:
            record = records[frame_id]
            row, rgb, overlay = audit_sample(dataset, obj, record, info,
                choice['instanceIds'][frame_id], points, normals, count, acquisition)
            manual.append(row)
            resource_hashes += 3
            shapes.add((rgb.shape[1], rgb.shape[0]))
            intrinsics.add(tuple(record['calibration']['cam_K']))
            scales.add(record['calibration']['depth_scale'])
            if overlay is not None:
                contacts.append((scene, object_id, rgb, overlay))
    model_error, bank = audit_saved_bank(dataset, asset, inference, raw_models)
    results = read_json(results_path)
    require(results['benchmarkId'] == inference['benchmarkId'], 'Results refer to another benchmark')
    current = [row for row in results['sequenceTrials'] if row['system'] == 'current_prototype']
    require(len(current) == 36, 'Expected twelve current-prototype tracks and three repeats')
    failures = Counter()
    for trial in current:
        failures.update(trial['failures'])
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (1280, len(contacts) * 505), (18, 23, 29))
    draw = ImageDraw.Draw(sheet)
    for index, (scene, object_id, rgb, overlay) in enumerate(contacts):
        y = index * 505
        sheet.paste(Image.fromarray(rgb), (0, y + 25))
        sheet.paste(Image.fromarray(overlay), (640, y + 25))
        draw.text((10, y + 5), f'REAL HB scene {scene}, object {object_id}, frame 10: original RGB', fill='white')
        draw.text((650, y + 5), 'Green=visible mask; orange=projected convex hull; magenta=model samples', fill='white')
    sheet.save(output / 'contact-sheet.jpg', quality=88)
    summary = dict(schema='visualizeit-hb-protocol-audit/1', scriptSHA256=script_hash,
        protocol='Independent read-only input/unit/projection audit; no tracker/neural inference or full mesh rendering',
        sourceFormat=BOP_FORMAT, datasetRevision=inference['datasetRevision'],
        scope=dict(scenes=scenes, geometryObjects=[first_by_scene[s]['objectId'] for s in scenes],
                   sampleFrameIds=list(SAMPLE_FRAME_IDS), imageSamples=len(manual), objectTracks=12,
                   uniqueSourceRGBFrames=selection['uniqueRGBFrames'], uniqueScoredObjectFrames=3600),
        inputs={name: dict(path=display_path(path), sha256=sha(path)) for name, path in {
            'inference': dataset / 'inference.json', 'evaluator': dataset / 'evaluator.json',
            'selection': dataset / 'selection.json', 'acquisitionLock': dataset / 'sources.lock.json',
            'benchmarkResults': results_path, 'savedDepthBankAsset': asset}.items()},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__, opencv=cv2.__version__),
        sourceJoinsVerified=source_joins, labelFreeScoredRecords=label_free,
        eligibleVisibleUniqueObjectFrames=int(visible), testedImageSizes=[list(s) for s in sorted(shapes)],
        testedIntrinsics=[list(k) for k in sorted(intrinsics)], testedDepthScales=sorted(scales),
        originalMetadataHashesVerified=metadata_hashes, sampledModelHashesVerified=model_hashes,
        sampleRGBDepthMaskResourceHashesVerifiedAgainstAcquisitionLock=resource_hashes,
        actualSavedModelMetresVsIndependentPLYMaxErrorMetres=model_error, actualBankChecks=bank,
        samples=manual, currentPrototypeObservedFailureReasonsWithThreeRepeats=dict(failures),
        initializationFailureObjects=[dict(objectId=row['objectId'], reason=row['setupFailure'])
            for row in current if row['repeat'] == 0 and row['setupFailure']],
        metricInterpretation='Custom visible-frame full-model corresponding-point mean reprojection<5px coverage; not official BOP score; no symmetry quotient; depth-bank-initialized ORB/PnP, not GoTrack.',
        conclusion='No gross units/axes/calibration/model-frame mismatch found in sampled scope. Missing tracking outputs dominate low coverage; this is not causal proof of a particular failure mechanism.',
        limitations=LIMITATIONS, contactSheetSHA256=sha(output / 'contact-sheet.jpg'),
        auditWallSeconds=time.perf_counter() - started)
    require(sha(Path(__file__)) == script_hash, 'Audit script changed during execution')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / '.cache/tracking-benchmark/real-v1')
    parser.add_argument('--results', type=Path, default=ROOT / 'artifacts/tracking-benchmark/real-full-v1/results.json')
    parser.add_argument('--asset', type=Path, default=ROOT / 'artifacts/tracking-benchmark/service-probe-v1-assets/dataset-02/object-000020/tracking-asset.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tracking-benchmark/hb-protocol-audit-rerun')
    args = parser.parse_args(argv)
    result = audit(args.dataset, args.results, args.asset, args.output)
    print(json.dumps({key: result[key] for key in ('scriptSHA256', 'sourceJoinsVerified', 'labelFreeScoredRecords', 'conclusion')}, indent=2))


if __name__ == '__main__':
    main()
