"""Read-only full-window report for the paired mug PnP initialization ablation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import struct

import numpy as np

from .quality_assets import CACHE, ROOT, digest, inference_provenance
from .quality_contract import checked_pose, project
from .quality_failure_review import rotation_error
from .quality_surface_agreement import sample_surface, visibility
from .quality_trace_compare import compare as compare_traces
from .quality_evaluate import prefix_equal
from .glb_model import read_glb
from .show3d_experiment import evaluation_truth


MUG_FRAME_IDS = tuple(range(827, 1067))
PREFIX_LENGTHS = (30, 120, 240)
BRANCHES = ('control', 'candidate')
STAGE_IDS = tuple(f'{branch}-{length}' for branch in BRANCHES for length in PREFIX_LENGTHS)
SOURCE_MODULES = (
    'quality_assets', 'quality_contract', 'quality_runner', 'quality_gotrack', 'quality_sam2',
    'quality_foundpose', 'quality_cnos', 'quality_detection_association', 'quality_neighbors',
    'quality_trace', 'quality_memory', 'quality_appearance', 'quality_render_stability', 'vision',
    'quality_evaluate', 'quality_surface_agreement', 'quality_failure_review', 'quality_trace_compare',
    'quality_annotations', 'glb_model', 'renderer', 'model', 'show3d_experiment', 'hd_experiment',
)
EXTERNAL_SOURCE_DIRS = (CACHE / 'sources/sam2', CACHE / 'sources/gotrack')
EXPECTED_SETTINGS = {
    'model_memory': False,
    'unlit_templates': True,
    'appearance_check': False,
    'disable_multisampling': True,
}


def _relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _png_dimensions(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Cached native mask is not a PNG: {path}')
    return struct.unpack('>II', header[16:24])


def _safe_mask_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Segmentation mask path is required')
    parsed = PurePosixPath(relative.replace('\\', '/'))
    if parsed.is_absolute() or '..' in parsed.parts:
        raise ValueError(f'Unsafe segmentation mask path: {relative!r}')
    base = Path(root).resolve()
    target = (base / Path(*parsed.parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f'Segmentation mask escapes cache root: {relative!r}') from exc
    return target


def _files_have_same_bytes(left, right):
    with Path(left).open('rb') as left_stream, Path(right).open('rb') as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def write_source_snapshot(source_files, snapshot_tree, source_root=ROOT):
    """Copy source bytes to short content-addressed paths and retain their origins."""
    source_root = Path(source_root).resolve()
    snapshot_tree = Path(snapshot_tree).resolve()
    snapshot_tree.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for relative in sorted(source_files):
        expected = source_files[relative]
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or '..' in parsed.parts or not parsed.parts:
            raise ValueError(f'Unsafe source snapshot path: {relative!r}')
        source = (source_root / Path(*parsed.parts)).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f'Source snapshot path escapes the source root: {relative!r}') from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        if not isinstance(expected, str) or len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError(f'Invalid frozen source hash for {relative!r}')
        if digest(source) != expected:
            raise ValueError(f'Source changed before snapshot copy: {relative}')

        target = snapshot_tree / 'files' / expected
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or digest(target) != expected or not _files_have_same_bytes(source, target):
                raise ValueError(f'Content-addressed source snapshot collision for {relative!r}')
        else:
            with source.open('rb') as input_stream, target.open('xb') as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            if digest(target) != expected or not _files_have_same_bytes(source, target):
                raise ValueError(f'Source snapshot copy failed hash/content verification for {relative!r}')
        artifacts[relative] = {
            'artifact_path': f'source-snapshot/files/{expected}',
            'sha256': expected,
        }
    return artifacts


def prepare_experiment_snapshot(bundle, masks_root, experiment_root, smoke_path=None):
    """Freeze inference inputs and source hashes into a new experiment directory."""
    bundle = Path(bundle).resolve()
    masks_root = Path(masks_root).resolve()
    experiment_root = Path(experiment_root).resolve()
    if not experiment_root.is_dir():
        raise ValueError('Harness must create the new experiment directory first')
    if (experiment_root / 'experiment.json').exists() or (experiment_root / 'source-snapshot').exists():
        raise FileExistsError('Experiment snapshot already exists; preserve it and choose a new root')

    manifest = json.loads((bundle / 'input.json').read_text(encoding='utf-8'))
    if manifest.get('object') != 'mug' or manifest.get('units') != 'metres':
        raise ValueError('The full-window harness requires the metric mug input bundle')
    ids = manifest.get('frame_ids')
    if ids != list(MUG_FRAME_IDS):
        raise ValueError('Mug manifest must contain exactly source frames 827–1066 in order')
    setup_id = manifest.get('setup_frame_id')
    if type(setup_id) is not int:
        raise ValueError('Mug setup frame ID must be an integer')
    resolution = manifest.get('native_resolution')
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int for v in resolution):
        raise ValueError('Native mug resolution is required')

    input_file_hashes = {}
    for filename in ('input.json', manifest.get('video'), manifest.get('asset'), 'setup.png'):
        if not isinstance(filename, str):
            raise ValueError('Mug video and model asset names are required')
        path = (bundle / filename).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as exc:
            raise ValueError(f'Input bundle path escapes its root: {filename!r}') from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        input_file_hashes[_relative(path)] = digest(path)
    for filename, expected in manifest.get('source_hashes', {}).items():
        path = (bundle / filename).resolve()
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Input asset does not match the frozen manifest hash: {filename}')

    segmentation_path = masks_root / 'results.json'
    segmentation = json.loads(segmentation_path.read_text(encoding='utf-8'))
    if ('automatic' in segmentation and segmentation.get('automatic') is not True) or \
       ('diagnostic_control' in segmentation and segmentation.get('diagnostic_control') is not False):
        raise ValueError('Corrected or non-automatic segmentation is forbidden for this experiment')
    if segmentation.get('stress_test') is not None:
        raise ValueError('Stress-test segmentation cannot be used for the mug window')
    model_name = segmentation.get('model', '')
    if 'sam 2.1 hiera base plus' not in model_name.casefold():
        raise ValueError('The cached automatic SAM 2.1 mug segmentation is required')
    mask_records = segmentation.get('frames')
    expected_mask_ids = [setup_id, *MUG_FRAME_IDS]
    if not isinstance(mask_records, list) or any(not isinstance(row, dict) for row in mask_records):
        raise ValueError('Cached segmentation frame records must be objects')
    if [row.get('frameId') for row in mask_records] != expected_mask_ids:
        raise ValueError('Cached segmentation must contain the setup mask followed by all 240 scored masks')
    if any(row.get('diagnostic_corrected_mask') for row in mask_records):
        raise ValueError('Corrected diagnostic masks are forbidden')
    mask_hashes = {}
    mask_file_hashes = {}
    for row in mask_records:
        frame_id = row['frameId']
        path = _safe_mask_path(masks_root, row.get('path'))
        if not path.is_file():
            raise FileNotFoundError(path)
        if _png_dimensions(path) != (resolution[0], resolution[1]):
            raise ValueError(f'Cached mask {frame_id} is not at native resolution')
        sha = digest(path)
        mask_hashes[str(frame_id)] = sha
        mask_file_hashes[_relative(path)] = sha

    inference = inference_provenance(bundle)
    source_files = {}
    for name in SOURCE_MODULES:
        path = ROOT / 'bench' / f'{name}.py'
        if not path.is_file():
            raise FileNotFoundError(f'Required inference source is missing: {path}')
        source_files[_relative(path)] = digest(path)
    for source_root in EXTERNAL_SOURCE_DIRS:
        if not source_root.is_dir():
            raise FileNotFoundError(f'Required pinned inference source tree is missing: {source_root}')
        for path in source_root.rglob('*'):
            if path.is_file():
                source_files[_relative(path)] = digest(path)
    evaluator_path = ROOT / 'bench/quality_pnp_mug_report.py'
    source_files[_relative(evaluator_path)] = digest(evaluator_path)

    bank_path = CACHE / 'banks/mug-foundpose.pt'
    checkpoint_path = CACHE / 'checkpoints/gotrack_checkpoint.pt'
    for path in (bank_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required pinned pose asset is missing: {path}')
    checkpoint_hash = digest(checkpoint_path)
    if checkpoint_hash != inference['checkpoint_sha256']['gotrack_checkpoint.pt']:
        raise ValueError('GoTrack checkpoint hash does not match the pinned provenance')
    bank_hash = digest(bank_path)
    run_provenance = {**inference, 'foundpose_bank_sha256': bank_hash}

    if smoke_path is None:
        device = 'cuda'
        smoke_path = CACHE / 'smoke/mug-cuda-unlit-no-msaa.json'
    else:
        smoke_path = Path(smoke_path).resolve()
        try:
            smoke_relative = smoke_path.relative_to(CACHE.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError('Renderer smoke record must be inside the model-quality cache') from exc
        parts = Path(smoke_relative).stem.split('-')
        device = parts[1] if len(parts) > 1 else ''
    if not smoke_path.is_file():
        raise FileNotFoundError(f'Existing setup renderer smoke is required: {smoke_path}')
    smoke = json.loads(smoke_path.read_text(encoding='utf-8'))
    if not smoke.get('smoke_passed') or not smoke.get('unlit_templates') or not smoke.get('disable_multisampling'):
        raise ValueError('Existing unlit/no-MSAA setup smoke must have passed')
    if smoke.get('runtime', {}).get('device') != device:
        raise ValueError('Smoke device and requested experiment device do not match')

    frozen_files = {
        **input_file_hashes,
        _relative(segmentation_path): digest(segmentation_path),
        **mask_file_hashes,
        _relative(bank_path): bank_hash,
        _relative(checkpoint_path): checkpoint_hash,
        _relative(smoke_path): digest(smoke_path),
    }
    snapshot_tree = experiment_root / 'source-snapshot'
    source_snapshot_files = write_source_snapshot(source_files, snapshot_tree)

    result = dict(schema_version=1, object='mug', created_utc=datetime.now(timezone.utc).isoformat(),
        expected_frame_ids=list(MUG_FRAME_IDS), setup_frame_id=setup_id, native_resolution=resolution,
        device=device, run_mode='complete', prefix_lengths=list(PREFIX_LENGTHS),
        declared_branch_difference={'tracking_settings.pnp_use_extrinsic_guess': {'control': True, 'candidate': False}},
        required_settings=EXPECTED_SETTINGS, smoke_runtime=smoke['runtime'], inference_provenance=run_provenance,
        source_files=source_files, source_snapshot_files=source_snapshot_files, frozen_files=frozen_files,
        mask_sha256_by_frame=mask_hashes, segmentation_provenance=segmentation.get('provenance'),
        segmentation_manifest_sha256=digest(segmentation_path), setup_mask_sha256=mask_hashes[str(setup_id)],
        evaluator_sha256=source_files[_relative(evaluator_path)],
        reference_or_annotation_inputs=False,
        scope='Frozen source/setup/cache manifest for complete-mode paired original mug window; no reference poses or annotations are inference inputs.')
    target_manifest = experiment_root / 'experiment.json'
    with target_manifest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def changed_source_files(snapshot, root=ROOT):
    changed = []
    for relative, expected in snapshot.get('source_files', {}).items():
        path = Path(root) / Path(*PurePosixPath(relative).parts)
        if not path.is_file() or digest(path) != expected:
            changed.append(relative)
    return changed


def changed_frozen_files(snapshot, root=ROOT):
    changed = []
    for relative, expected in snapshot.get('frozen_files', {}).items():
        path = Path(root) / Path(*PurePosixPath(relative).parts)
        if not path.is_file() or digest(path) != expected:
            changed.append(relative)
    return changed


def validate_result(result, expected_ids, expected_complete, expected_setup_id=None):
    if not isinstance(result, dict):
        raise ValueError('Pose result must be a JSON object')
    if result.get('object') != 'mug' or result.get('mode') != 'complete':
        raise ValueError('Every run must use complete mode for the mug')
    if type(result.get('complete')) is not bool or result['complete'] is not expected_complete:
        label = 'full' if expected_complete else 'partial-prefix'
        raise ValueError(f'{label} result has incorrect complete flag')
    if 'status' in result and result['status'] != 'complete':
        raise ValueError(f"Run status must be 'complete', found {result['status']!r}")
    if result.get('stress_test') is not None:
        raise ValueError('Stress-test pose results cannot enter this comparison')
    if result.get('independent_accuracy_scored') is not False:
        raise ValueError('This comparison must remain unscored by independent annotations')
    if result.get('automatic') is not True or result.get('diagnostic_control') is not False:
        raise ValueError('Automatic complete-mode results are required')
    expected_ids = list(expected_ids)
    if result.get('frame_ids') != expected_ids:
        raise ValueError('Run frame_ids do not exactly match the requested source IDs')
    if expected_setup_id is not None:
        initialization = result.get('initialization')
        if not isinstance(initialization, dict) or initialization.get('frameId') != expected_setup_id:
            raise ValueError('Complete mode must initialize from the frozen setup frame only')
        if initialization.get('cameraFromObject') is not None:
            raise ValueError('Setup initialization must not publish a scored-frame pose seed')
    frames = result.get('frames')
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError('Every requested source frame must have an object record')
    if [frame.get('frameId') for frame in frames] != expected_ids:
        raise ValueError('Every requested source frame must be present exactly once and in order')
    settings = result.get('tracking_settings')
    if not isinstance(settings, dict):
        raise ValueError('Run tracking settings are required')
    for key, expected in EXPECTED_SETTINGS.items():
        if type(settings.get(key)) is not bool or settings[key] is not expected:
            raise ValueError(f'Run has an invalid frozen setting: {key}')
    if type(settings.get('pnp_use_extrinsic_guess')) is not bool:
        raise ValueError('Run must record the PnP extrinsic-guess boolean')

    for frame in frames:
        fid = frame['frameId']
        state = frame.get('pose_state')
        render_state = frame.get('render_state')
        mask_state = frame.get('mask_state')
        if state not in ('tracking', 'recovering', 'lost') or render_state not in ('visible', 'suppressed'):
            raise ValueError(f'Frame {fid} has an invalid pose or render state')
        pose = frame.get('cameraFromObject')
        if state == 'tracking':
            if pose is None or render_state != 'visible' or mask_state != 'available':
                raise ValueError(f'Accepted frame {fid} must publish a visible pose with an available mask')
            checked_pose(pose)
        else:
            if pose is not None or render_state != 'suppressed':
                raise ValueError(f'Non-tracking frame {fid} must suppress rendering and expose no pose')
        mask_sha = frame.get('mask_sha256')
        if not isinstance(mask_sha, str) or len(mask_sha) != 64 or any(c not in '0123456789abcdef' for c in mask_sha):
            raise ValueError(f'Frame {fid} is missing the cached mask SHA-256')
        if not isinstance(frame.get('mask_path'), str) or not frame['mask_path']:
            raise ValueError(f'Frame {fid} is missing its cached mask path')
        timings = frame.get('timings_ms', {})
        if not isinstance(timings, dict):
            raise ValueError(f'Frame {fid} timing record must be an object')
        if 'pose_total' in timings:
            value = timings['pose_total']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'Frame {fid} has an invalid pose latency')
    if not isinstance(result.get('provenance'), dict) or not isinstance(result.get('runtime'), dict):
        raise ValueError('Runtime and inference provenance are required')
    return result


def _accepted(frame):
    return frame.get('pose_state') == 'tracking' and frame.get('cameraFromObject') is not None


def _frame_subset(result, ids):
    selected = set(ids)
    return {**result,
        'frame_ids': [frame_id for frame_id in result['frame_ids'] if frame_id in selected],
        'frames': [frame for frame in result['frames'] if frame['frameId'] in selected]}


def _intervals(frames, predicate, include_reasons=False):
    groups = []
    current = []
    for frame in frames:
        if predicate(frame):
            if current and frame['frameId'] != current[-1]['frameId'] + 1:
                groups.append(current)
                current = []
            current.append(frame)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    rows = []
    for group in groups:
        row = dict(start_frame_id=group[0]['frameId'], end_frame_id=group[-1]['frameId'], count=len(group))
        if include_reasons:
            row['failure_reasons'] = sorted({frame['failure_reason'] for frame in group if frame.get('failure_reason')})
        rows.append(row)
    return rows


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(samples=0, median=None, p95=None, maximum=None)
    return dict(samples=int(len(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def summarize_branch(result):
    frames = result['frames']
    accepted = [frame for frame in frames if _accepted(frame)]
    lost = [frame for frame in frames if frame['pose_state'] == 'lost']
    recovering = [frame for frame in frames if frame['pose_state'] == 'recovering']
    suppressed = [frame for frame in frames if frame['render_state'] == 'suppressed']
    failure_reasons = {}
    for frame in frames:
        reason = frame.get('failure_reason')
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    all_latency = [frame['timings_ms']['pose_total'] for frame in frames
                   if 'pose_total' in frame.get('timings_ms', {})]
    accepted_latency = [frame['timings_ms']['pose_total'] for frame in accepted
                        if 'pose_total' in frame.get('timings_ms', {})]
    return dict(requested_source_frames=len(result['frame_ids']), source_frames=len(frames),
        run_complete=result.get('complete') is True, accepted_frames=len(accepted), failures=len(frames) - len(accepted),
        accepted_pose_availability=len(accepted) / len(frames) if frames else None,
        lost_frames=len(lost), recovering_frames=len(recovering),
        rendering_visible_frames=sum(frame['render_state'] == 'visible' for frame in frames),
        rendering_suppressed_frames=len(suppressed), suppressed_frame_ids=[frame['frameId'] for frame in suppressed],
        failure_reasons=failure_reasons,
        failure_intervals=_intervals(frames, lambda frame: frame['pose_state'] == 'lost', include_reasons=True),
        recovery_intervals=_intervals(frames, lambda frame: frame['pose_state'] == 'recovering'),
        latency_ms=dict(all_source_frames=dict(_distribution(all_latency), missing_source_frames=len(frames)-len(all_latency)),
            accepted_frames=dict(_distribution(accepted_latency), missing_accepted_frames=len(accepted)-len(accepted_latency))))


def _native_metrics(summary, height):
    scale = height / 720
    return {**summary,
        'source_frames': summary.get('processed', summary.get('processed_frames', 0)),
        'missing_source_frames': max(0, summary.get('processed', summary.get('processed_frames', 0))
                                     - summary.get('accepted', summary.get('accepted_frames', 0))),
        'missing_reference_frames': max(0, summary.get('processed', summary.get('processed_frames', 0))
                                        - summary.get('reference_frames', 0)),
        'missing_point_samples': max(0, summary['reference_visible_samples'] - summary['accepted_point_samples']),
        'median_native_pixels': None if summary['median_720'] is None else summary['median_720'] * scale,
        'p95_native_pixels': None if summary['p95_720'] is None else summary['p95_720'] * scale}


def area_metrics(results, original, mesh, points, normals, visibility_cache):
    refs = {frame['frameId']: frame for frame in original['referenceFrames']}
    k = np.asarray(original['cameraCalibration']).reshape(3, 3)
    width, height = original['nativeResolution']
    errors = []
    details = []
    visible_samples = 0
    reference_frames = 0
    missing_reference_ids = []
    accepted_count = 0
    for frame in results['frames']:
        reference_frame = refs.get(frame['frameId'])
        reference = None if reference_frame is None else reference_frame.get('cameraFromObject')
        accepted = _accepted(frame)
        accepted_count += int(accepted)
        if reference is None:
            missing_reference_ids.append(frame['frameId'])
            details.append(dict(frame_id=frame['frameId'], accepted=accepted,
                reference_pose_available=False, reference_visible_samples=0, accepted_samples=0,
                missing_samples=0, median_720=None, p95_720=None))
            continue
        reference_frames += 1
        reference_pose = checked_pose(reference)
        key = frame['frameId']
        if key not in visibility_cache:
            visibility_cache[key] = visibility(mesh, points, normals, reference_pose, k,
                                               original['nativeResolution'])
        visible, expected = visibility_cache[key]
        visible_count = int(visible.sum())
        visible_samples += visible_count
        values = []
        if accepted:
            predicted, _ = project(points, checked_pose(frame['cameraFromObject']), k)
            values = (np.linalg.norm(predicted[visible] - expected[visible], axis=1)
                      * 720 / height).tolist()
            errors.extend(values)
        details.append(dict(frame_id=frame['frameId'], accepted=accepted,
            reference_pose_available=True, reference_visible_samples=visible_count,
            accepted_samples=len(values), missing_samples=visible_count-len(values),
            median_720=float(np.median(values)) if values else None,
            p95_720=float(np.percentile(values, 95)) if values else None))
    values = np.asarray(errors, dtype=float)
    summary = dict(processed=len(results['frames']), accepted=accepted_count,
        source_frames=len(results['frames']), reference_frames=reference_frames,
        reference_visible_samples=visible_samples, accepted_point_samples=len(errors),
        sample_availability=len(errors)/visible_samples if visible_samples else None,
        median_720=float(np.median(values)) if len(values) else None,
        p95_720=float(np.percentile(values, 95)) if len(values) else None,
        details=details, missing_reference_frame_ids=missing_reference_ids)
    return _native_metrics(summary, original['nativeResolution'][1])


def legacy_vertex_metrics(results, original, mesh, visibility_cache):
    """Expanded form of quality_evaluate.secondary_pose_agreement()."""
    k = np.asarray(original['cameraCalibration']).reshape(3, 3)
    height, width = original['nativeResolution'][1], original['nativeResolution'][0]
    vertex_ids = np.linspace(0, len(mesh.positions) - 1, min(128, len(mesh.positions)), dtype=int)
    references = {frame['frameId']: frame['cameraFromObject'] for frame in original['referenceFrames']}
    sample_errors_native = []
    details = []
    reference_frames = 0
    reference_visible_samples = 0
    missing_reference_ids = []
    for frame in results['frames']:
        reference = references.get(frame['frameId'])
        if reference is None:
            missing_reference_ids.append(frame['frameId'])
            details.append(dict(frame_id=frame['frameId'], accepted=_accepted(frame),
                                reference_pose_available=False, reference_visible_samples=0,
                                accepted_samples=0, missing_samples=0))
            continue
        reference_frames += 1
        pose = checked_pose(reference)
        key = (id(mesh), frame['frameId'], pose.tobytes(), k.tobytes(), tuple(original['nativeResolution']))
        if key not in visibility_cache:
            visible_ids, expected = evaluation_truth(mesh, vertex_ids,
                (pose[:3, :3], pose[:3, 3]), k, width, height)
            visibility_cache[key] = (np.asarray(visible_ids, dtype=int), expected)
        visible_ids, expected = visibility_cache[key]
        reference_visible_samples += len(visible_ids)
        values = []
        accepted = _accepted(frame)
        if accepted and len(visible_ids):
            predicted, _ = project(mesh.positions[vertex_ids], checked_pose(frame['cameraFromObject']), k)
            values = np.linalg.norm(predicted[visible_ids] - expected[visible_ids], axis=1).tolist()
            sample_errors_native.extend(values)
        details.append(dict(frame_id=frame['frameId'], accepted=accepted, reference_pose_available=True,
            reference_visible_samples=len(visible_ids), accepted_samples=len(values),
            missing_samples=len(visible_ids) - len(values)))
    errors = np.asarray(sample_errors_native, dtype=float)
    errors_720 = errors * 720 / height
    return dict(processed_frames=len(results['frames']), source_frames=len(results['frames']),
        accepted_frames=sum(_accepted(frame) for frame in results['frames']),
        missing_source_frames=len(results['frames']) - sum(_accepted(frame) for frame in results['frames']),
        reference_frames=reference_frames, missing_reference_frames=len(results['frames']) - reference_frames,
        missing_reference_frame_ids=missing_reference_ids,
        reference_visible_samples=reference_visible_samples,
        accepted_point_samples=int(len(errors)), missing_point_samples=max(0, reference_visible_samples-len(errors)),
        sample_availability=len(errors)/reference_visible_samples if reference_visible_samples else None,
        median_native_pixels=float(np.median(errors)) if len(errors) else None,
        p95_native_pixels=float(np.percentile(errors, 95)) if len(errors) else None,
        median_720=float(np.median(errors_720)) if len(errors_720) else None,
        p95_720=float(np.percentile(errors_720, 95)) if len(errors_720) else None,
        vertex_sample_count=int(len(vertex_ids)), details=details,
        scope='Legacy 128 model-vertex secondary agreement with related-family estimated poses; not independent accuracy.')


def _orientation_diagnostics(results, original):
    references = {frame['frameId']: frame['cameraFromObject'] for frame in original['referenceFrames']}
    disagreements = []
    unavailable_reference_ids = []
    for frame in results['frames']:
        if not _accepted(frame):
            continue
        reference = references.get(frame['frameId'])
        if reference is None:
            unavailable_reference_ids.append(frame['frameId'])
            continue
        error = rotation_error(frame['cameraFromObject'], reference)
        if error >= 90:
            disagreements.append(dict(frame_id=frame['frameId'], rotation_degrees=error))
    return dict(frames_at_or_above_90_degrees=disagreements,
                frame_ids=[row['frame_id'] for row in disagreements],
                missing_estimated_reference_frame_ids=unavailable_reference_ids)


def _trace_prefix_status(traces, branch, shorter, longer):
    short = None if traces is None else traces.get(branch, {}).get(shorter)
    long = None if traces is None else traces.get(branch, {}).get(longer)
    if not isinstance(short, list) or not short or not isinstance(long, list) or not long:
        return dict(supplied=short is not None or long is not None,
                    passed=False, comparison=dict(reason='missing_or_empty_prefix_trace'))
    mismatch = compare_traces(short, long)
    return dict(supplied=True, passed=mismatch is None, comparison=mismatch)


def _run_metadata_checks(runs, experiment):
    issues = []
    all_results = [runs[branch][length] for branch in BRANCHES for length in PREFIX_LENGTHS]
    expected_provenance = experiment.get('inference_provenance')
    if not isinstance(expected_provenance, dict):
        issues.append('experiment snapshot is missing inference provenance')
    for index, result in enumerate(all_results):
        if result['provenance'] != expected_provenance:
            issues.append(f"{result['object']} {result['frame_ids'][-1]} provenance differs from the frozen snapshot")
        if result['runtime'].get('device') != experiment.get('device'):
            issues.append(f"{result['object']} {result['frame_ids'][-1]} runtime device differs from the frozen snapshot")
        runtime = result['runtime']
        threads = runtime.get('math_threads', {})
        if (runtime.get('deterministic_algorithms') is not True or
            runtime.get('cudnn_deterministic') is not True or
            runtime.get('cublas_workspace') != ':4096:8' or
            any(threads.get(name) != '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')) or
            runtime.get('precision') != 'float32' or runtime.get('batch_size') != 1 or
            runtime.get('iterations') != 5 or runtime.get('crop_size') != [280, 280]):
            issues.append(f"{result['object']} {result['frame_ids'][-1]} runtime is not the frozen deterministic GoTrack setup")
        if index and result['runtime'] != all_results[0]['runtime']:
            issues.append(f"{result['object']} {result['frame_ids'][-1]} runtime differs between phases")

    normalized = []
    for branch in BRANCHES:
        for length in PREFIX_LENGTHS:
            result = runs[branch][length]
            settings = result['tracking_settings']
            expected_pnp = branch == 'control'
            if type(settings.get('pnp_use_extrinsic_guess')) is not bool or settings.get('pnp_use_extrinsic_guess') is not expected_pnp:
                issues.append(f'{branch}-{length} has the wrong PnP boolean')
            if any(type(settings.get(key)) is not bool or settings.get(key) is not expected
                   for key, expected in EXPECTED_SETTINGS.items()):
                issues.append(f'{branch}-{length} changed a required fixed setting')
            normalized.append({key: value for key, value in settings.items() if key != 'pnp_use_extrinsic_guess'})
    if normalized and any(settings != normalized[0] for settings in normalized[1:]):
        issues.append('Paired runs differ in a setting other than the declared PnP boolean')
    if len(all_results) != len(STAGE_IDS) or any(result.get('mode') != 'complete' for result in all_results):
        issues.append('All six fresh complete-mode runs are required')

    expected_masks = experiment.get('mask_sha256_by_frame', {})
    for branch in BRANCHES:
        for length in PREFIX_LENGTHS:
            result = runs[branch][length]
            for frame in result['frames']:
                expected = expected_masks.get(str(frame['frameId']))
                if expected is None or frame['mask_sha256'] != expected:
                    issues.append(f"{branch}-{length} mask hash differs from frozen segmentation at {frame['frameId']}")
    return dict(passed=not issues, issues=issues)


def _prefix_checks(runs, traces=None):
    checks = {}
    for branch in BRANCHES:
        for shorter, longer in ((30, 120), (30, 240), (120, 240)):
            short = runs[branch][shorter]
            long = runs[branch][longer]
            runtime_equal = short['runtime'] == long['runtime']
            provenance_equal = short['provenance'] == long['provenance']
            settings_equal = short['tracking_settings'] == long['tracking_settings']
            poses_and_frames_equal = prefix_equal(short, long)
            trace = _trace_prefix_status(traces, branch, shorter, longer)
            checks[f'{branch}_{shorter}_vs_{longer}'] = dict(
                runtime_equal=runtime_equal, provenance_equal=provenance_equal,
                settings_equal=settings_equal, prefix_equal=poses_and_frames_equal,
                trace=trace, passed=runtime_equal and provenance_equal and settings_equal
                    and poses_and_frames_equal and trace['passed'])
    return checks


def _stage_lifecycle(stage_evidence, expected_output_hashes=None, experiment=None, runs=None,
                     trace_attestations=None, traces=None):
    issues = []
    expected_output_hashes = expected_output_hashes or {}
    if not isinstance(stage_evidence, dict) or stage_evidence.get('status') != 'complete':
        issues.append('harness did not record a complete six-stage run')
    stages = stage_evidence.get('stages', []) if isinstance(stage_evidence, dict) else []
    by_id = {stage.get('stage_id'): stage for stage in stages if isinstance(stage, dict)}
    if len(stages) != len(STAGE_IDS) or [stage.get('stage_id') for stage in stages if isinstance(stage, dict)] != list(STAGE_IDS):
        issues.append('terminal evidence does not contain the six expected phases in order')
    if isinstance(stage_evidence, dict) and stage_evidence.get('attempted_stage_ids') != list(STAGE_IDS):
        issues.append('harness did not record attempts in the required order')
    expected_source_files = experiment.get('source_files') if isinstance(experiment, dict) else None
    expected_frozen_files = experiment.get('frozen_files') if isinstance(experiment, dict) else None
    expected_provenance = experiment.get('inference_provenance') if isinstance(experiment, dict) else None
    if not isinstance(trace_attestations, dict) or set(trace_attestations) != set(STAGE_IDS):
        issues.append('all six terminal trace files must be hash-bound and present')
    if not isinstance(expected_source_files, dict):
        issues.append('frozen snapshot is missing source file hashes')
    if not isinstance(expected_frozen_files, dict):
        issues.append('frozen snapshot is missing input and checkpoint hashes')
    if not isinstance(expected_provenance, dict):
        issues.append('frozen snapshot is missing inference provenance')
    for stage_id in STAGE_IDS:
        stage = by_id.get(stage_id)
        if stage is None:
            continue
        if stage.get('status') != 'succeeded' or type(stage.get('exit_code')) is not int or stage['exit_code'] != 0:
            issues.append(f'{stage_id} has no terminal exit-code-zero evidence')
        if stage.get('terminal_exit_recorded') is not True:
            issues.append(f'{stage_id} lacks recorded terminal exit evidence')
        branch, length_text = stage_id.rsplit('-', 1)
        length = int(length_text)
        expected_ids = list(MUG_FRAME_IDS[:length])
        if stage.get('output') != f'{stage_id}.json':
            issues.append(f'{stage_id} records an unexpected output path')
        if stage.get('requested_frame_ids') != expected_ids:
            issues.append(f'{stage_id} does not record the exact requested frame IDs')
        if stage.get('expected_complete') is not (length == 240):
            issues.append(f'{stage_id} records an incorrect expected completion flag')
        if stage.get('pnp_use_extrinsic_guess') is not (branch == 'control'):
            issues.append(f'{stage_id} records the wrong branch setting')
        expected_trace_path = f'{stage_id}.trace.jsonl'
        trace_hash = stage.get('trace_sha256')
        trace_count = stage.get('trace_record_count')
        if stage.get('trace') != expected_trace_path:
            issues.append(f'{stage_id} records an unexpected trace path')
        if not isinstance(trace_hash, str) or len(trace_hash) != 64 or any(c not in '0123456789abcdef' for c in trace_hash):
            issues.append(f'{stage_id} has no valid terminal trace hash')
        if type(trace_count) is not int or trace_count < 1:
            issues.append(f'{stage_id} has no nonempty terminal trace count')
        attestation = None if not isinstance(trace_attestations, dict) else trace_attestations.get(stage_id)
        if not isinstance(attestation, dict):
            issues.append(f'{stage_id} trace file is missing or empty')
        else:
            if attestation.get('path') != expected_trace_path or attestation.get('sha256') != trace_hash:
                issues.append(f'{stage_id} loaded trace path/hash differs from terminal evidence')
            if type(attestation.get('record_count')) is not int or attestation.get('record_count') != trace_count:
                issues.append(f'{stage_id} loaded trace record count differs from terminal evidence')
        if traces is not None:
            stage_records = traces.get(branch, {}).get(length) if isinstance(traces.get(branch), dict) else None
            if not isinstance(stage_records, list) or len(stage_records) != trace_count or not stage_records:
                issues.append(f'{stage_id} parsed trace records differ from terminal evidence')
        if expected_source_files is not None and stage.get('source_files') != expected_source_files:
            issues.append(f'{stage_id} source hashes differ from the frozen snapshot')
        if expected_frozen_files is not None and stage.get('frozen_files') != expected_frozen_files:
            issues.append(f'{stage_id} input/checkpoint hashes differ from the frozen snapshot')
        if expected_provenance is not None and stage.get('inference_provenance') != expected_provenance:
            issues.append(f'{stage_id} inference provenance differs from the frozen snapshot')
        if runs is not None:
            result = runs[branch][length]
            if stage.get('runtime') != result.get('runtime'):
                issues.append(f'{stage_id} terminal runtime differs from its saved output')
            if stage.get('tracking_settings') != result.get('tracking_settings'):
                issues.append(f'{stage_id} terminal settings differ from its saved output')
            if stage.get('result_frame_ids') != result.get('frame_ids'):
                issues.append(f'{stage_id} terminal IDs differ from its saved output')
            if stage.get('result_complete') is not result.get('complete'):
                issues.append(f'{stage_id} terminal completion differs from its saved output')
        if expected_output_hashes and stage.get('output_sha256') != expected_output_hashes.get(stage_id):
            issues.append(f'{stage_id} output hash differs from its terminal evidence')
    return dict(passed=not issues, issues=issues, stages=stages)


def _gate_checks(control_stats, candidate_stats, common_control, common_candidate,
                 prefixes, metadata, lifecycle, snapshot_intact, orientation):
    control_own = control_stats['area_metrics']
    candidate_own = candidate_stats['area_metrics']
    checks = {
        'full_six_stage_lifecycle_exit_zero': lifecycle['passed'],
        'frozen_snapshot_intact': snapshot_intact,
        'paired_provenance_and_settings_match': metadata['passed'],
        'candidate_accepted_at_least_control': candidate_stats['accounting']['accepted_frames'] >= control_stats['accounting']['accepted_frames'],
        'candidate_accepted_at_least_216': candidate_stats['accounting']['accepted_frames'] >= 216,
        'common_area_median_not_worse': (common_control['median_720'] is not None and common_candidate['median_720'] is not None
            and common_candidate['median_720'] <= common_control['median_720']),
        'common_area_p95_strictly_better': (common_control['p95_720'] is not None and common_candidate['p95_720'] is not None
            and common_candidate['p95_720'] < common_control['p95_720']),
        'own_area_median_not_worse': (control_own['median_720'] is not None and candidate_own['median_720'] is not None
            and candidate_own['median_720'] <= control_own['median_720']),
        'own_area_p95_not_worse': (control_own['p95_720'] is not None and candidate_own['p95_720'] is not None
            and candidate_own['p95_720'] <= control_own['p95_720']),
        'no_new_90_degree_disagreement': not orientation['new_candidate_frame_ids'],
    }
    for key, value in prefixes.items():
        checks[f'prefix_{key}'] = value['passed']
    return checks


def build_report(runs, original, mesh, points, normals, experiment, stage_evidence,
                 traces=None, trace_attestations=None, snapshot_intact=True, expected_output_hashes=None,
                 expected_ids=MUG_FRAME_IDS):
    """Build metrics from saved outputs; never feeds references into inference."""
    ids = list(expected_ids)
    if ids != list(MUG_FRAME_IDS):
        raise ValueError('Full report is defined for the frozen mug IDs 827–1066')
    if original.get('name') != 'mug' or getattr(mesh, 'name', None) != 'mug':
        raise ValueError('Estimated reference data and model mesh must both identify the mug')
    if set(runs) != set(BRANCHES) or any(set(runs[branch]) != set(PREFIX_LENGTHS) for branch in BRANCHES):
        raise ValueError('All six control/candidate 30, 120 and 240 frame runs are required')
    if not isinstance(traces, dict) or any(
        not isinstance(traces.get(branch, {}).get(length), list) or not traces[branch][length]
        for branch in BRANCHES for length in PREFIX_LENGTHS
    ):
        raise ValueError('All six nonempty stage traces are required for a complete report')
    if not isinstance(trace_attestations, dict) or set(trace_attestations) != set(STAGE_IDS):
        raise ValueError('All six stage traces must be bound to terminal file hashes')
    for branch in BRANCHES:
        for length in PREFIX_LENGTHS:
            expected = ids[:length]
            validate_result(runs[branch][length], expected, expected_complete=(length == 240),
                            expected_setup_id=experiment.get('setup_frame_id'))

    metadata = _run_metadata_checks(runs, experiment)
    prefixes = _prefix_checks(runs, traces)
    lifecycle = _stage_lifecycle(stage_evidence, expected_output_hashes, experiment, runs,
                                 trace_attestations, traces)
    expected_masks = experiment.get('mask_sha256_by_frame', {})
    snapshot_intact = bool(snapshot_intact)

    branch_stats = {}
    legacy_cache = {}
    area_cache = {}
    for branch in BRANCHES:
        full = runs[branch][240]
        branch_stats[branch] = dict(
            accounting=summarize_branch(full),
            legacy_vertex_metrics=legacy_vertex_metrics(full, original, mesh, legacy_cache),
            area_metrics=area_metrics(full, original, mesh, points, normals, area_cache),
        )

    control_accepted = {frame['frameId'] for frame in runs['control'][240]['frames'] if _accepted(frame)}
    candidate_accepted = {frame['frameId'] for frame in runs['candidate'][240]['frames'] if _accepted(frame)}
    common_ids = [frame_id for frame_id in ids if frame_id in control_accepted and frame_id in candidate_accepted]
    common_results = {
        branch: _frame_subset(runs[branch][240], common_ids) for branch in BRANCHES
    }
    common_metrics = {}
    for branch in BRANCHES:
        common_metrics[branch] = dict(
            accepted_frames=len(common_ids),
            legacy_vertex_metrics=legacy_vertex_metrics(common_results[branch], original, mesh, legacy_cache),
            area_metrics=area_metrics(common_results[branch], original, mesh, points, normals, area_cache),
        )

    orientation = {branch: _orientation_diagnostics(runs[branch][240], original) for branch in BRANCHES}
    control_large = set(orientation['control']['frame_ids'])
    new_orientation_ids = sorted(set(orientation['candidate']['frame_ids']) - control_large)
    orientation_delta = dict(new_candidate_frame_ids=new_orientation_ids,
        control_frame_ids=sorted(control_large), candidate_frame_ids=sorted(orientation['candidate']['frame_ids']))
    control_suppressed = {frame['frameId'] for frame in runs['control'][240]['frames']
                          if frame['render_state'] == 'suppressed'}
    candidate_suppressed = {frame['frameId'] for frame in runs['candidate'][240]['frames']
                            if frame['render_state'] == 'suppressed'}
    suppression_delta = dict(
        control_suppressed_frame_ids=sorted(control_suppressed),
        candidate_suppressed_frame_ids=sorted(candidate_suppressed),
        candidate_only_suppressed_frame_ids=sorted(candidate_suppressed - control_suppressed),
        control_only_suppressed_frame_ids=sorted(control_suppressed - candidate_suppressed),
        control_suppressed_frames=len(control_suppressed), candidate_suppressed_frames=len(candidate_suppressed),
    )
    gates = _gate_checks(branch_stats['control'], branch_stats['candidate'],
        common_metrics['control']['area_metrics'], common_metrics['candidate']['area_metrics'],
        prefixes, metadata, lifecycle, snapshot_intact, orientation_delta)
    full_outputs_complete = all(runs[branch][240]['complete'] is True for branch in BRANCHES)
    comparison_complete = full_outputs_complete and lifecycle['passed']
    continuation_passed = comparison_complete and all(gates.values())

    return dict(schema_version=1, object='mug', scope='Fresh complete-mode original-window PnP initialization comparison. Estimated-reference metrics are secondary and are not an independent accuracy gate.',
        requested_source_frames=len(ids), source_frame_ids=ids,
        branch_accounting={branch: branch_stats[branch]['accounting'] for branch in BRANCHES},
        own_accepted_legacy_vertex_metrics={branch: branch_stats[branch]['legacy_vertex_metrics'] for branch in BRANCHES},
        own_accepted_fixed128_area_metrics={branch: branch_stats[branch]['area_metrics'] for branch in BRANCHES},
        common_accepted_frame_ids=common_ids, common_accepted_frames=len(common_ids),
        common_accepted_legacy_vertex_metrics={branch: common_metrics[branch]['legacy_vertex_metrics'] for branch in BRANCHES},
        common_accepted_fixed128_area_metrics={branch: common_metrics[branch]['area_metrics'] for branch in BRANCHES},
        rendering_suppression_contribution=suppression_delta,
        area_sampling=dict(sample_count=128, seed=0, method='stratified triangle-area quantiles and uniform barycentric samples',
            visibility='Reference-pose self-visibility with the same surface sampler for every branch; no hand ground truth.'),
        orientation_disagreements_at_or_above_90_degrees=orientation,
        new_candidate_large_orientation_disagreements=orientation_delta,
        paired_source_settings_checks=metadata, prefix_checks=prefixes, terminal_stage_evidence=lifecycle,
        frozen_snapshot_intact=snapshot_intact, frozen_snapshot_expected_mask_sha256=expected_masks,
        full_window_results_complete=full_outputs_complete, full_comparison_complete=comparison_complete,
        continuation_gate_checks=gates, continuation_gate_passed=continuation_passed,
        independent_accuracy_verified=False, overall_gate_passed=False,
        reference_or_annotation_inputs=False, annotations_loaded=False)


def _read_trace_file(path, stage_id, stage_record):
    path = Path(path)
    expected_name = f'{stage_id}.trace.jsonl'
    if stage_record.get('trace') != expected_name:
        raise ValueError(f'{stage_id} terminal trace path does not match its stage ID')
    if not path.is_file():
        raise FileNotFoundError(f'Missing required terminal trace: {path}')
    raw = path.read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise ValueError(f'{stage_id} trace is empty or truncated before its final newline')
    try:
        lines = raw.decode('utf-8').splitlines()
        records = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{stage_id} trace contains a truncated or invalid JSON record') from exc
    if not records or any(not isinstance(row, dict) or not isinstance(row.get('stage'), str) or
                          not row.get('stage') or not isinstance(row.get('arrays'), dict) or not row['arrays']
                          for row in records):
        raise ValueError(f'{stage_id} trace contains an empty or malformed record')
    actual_sha = digest(path)
    expected_sha = stage_record.get('trace_sha256')
    expected_count = stage_record.get('trace_record_count')
    if actual_sha != expected_sha:
        raise ValueError(f'{stage_id} trace file hash differs from terminal evidence')
    if type(expected_count) is not int or expected_count < 1 or len(records) != expected_count:
        raise ValueError(f'{stage_id} trace record count differs from terminal evidence')
    return records, dict(path=expected_name, sha256=actual_sha, record_count=len(records))


def _read_evaluation_inputs(root):
    root = Path(root)
    experiment = json.loads((root / 'experiment.json').read_text(encoding='utf-8'))
    evidence = json.loads((root / 'stage-evidence.json').read_text(encoding='utf-8'))
    runs = {branch: {} for branch in BRANCHES}
    traces = {branch: {} for branch in BRANCHES}
    trace_attestations = {}
    output_hashes = {}
    for branch in BRANCHES:
        for length in PREFIX_LENGTHS:
            stage_id = f'{branch}-{length}'
            output = root / f'{stage_id}.json'
            if not output.is_file():
                raise FileNotFoundError(f'Missing preserved stage output: {output}')
            runs[branch][length] = json.loads(output.read_text(encoding='utf-8'))
            output_hashes[stage_id] = digest(output)
            trace_path = root / f'{stage_id}.trace.jsonl'
            stage_rows = [stage for stage in evidence.get('stages', []) if stage.get('stage_id') == stage_id]
            if len(stage_rows) != 1:
                raise ValueError(f'{stage_id} must have exactly one terminal evidence row')
            traces[branch][length], trace_attestations[stage_id] = _read_trace_file(trace_path, stage_id, stage_rows[0])
    for stage in evidence.get('stages', []):
        stage_id = stage.get('stage_id') if isinstance(stage, dict) else None
        if stage_id in output_hashes and stage.get('output') != f'{stage_id}.json':
            raise ValueError(f'{stage_id} terminal evidence points at a different output path')
    return experiment, evidence, runs, traces, trace_attestations, output_hashes


def _write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    snapshot_parser = subparsers.add_parser('snapshot', help='freeze exact mug inference inputs and source hashes')
    snapshot_parser.add_argument('--bundle', type=Path, required=True)
    snapshot_parser.add_argument('--masks', type=Path, required=True)
    snapshot_parser.add_argument('--experiment-root', type=Path, required=True)
    snapshot_parser.add_argument('--smoke', type=Path)
    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate six preserved terminal runs')
    evaluate_parser.add_argument('--experiment-root', type=Path, required=True)
    evaluate_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.command == 'snapshot':
        snapshot = prepare_experiment_snapshot(args.bundle, args.masks, args.experiment_root, args.smoke)
        print(json.dumps(dict(snapshot_created=True, object='mug', source_files=len(snapshot['source_files']),
            cached_masks=len(snapshot['mask_sha256_by_frame']), frozen_files=len(snapshot['frozen_files'])), indent=2))
        return

    experiment, evidence, runs, traces, trace_attestations, output_hashes = _read_evaluation_inputs(args.experiment_root)
    original = next(obj for obj in json.loads((ROOT / 'artifacts/video-60/report.json').read_text(encoding='utf-8'))['objects']
                    if obj['name'] == 'mug')
    mesh = read_glb(CACHE / 'inputs/mug/object.glb', 'mug')
    points, normals, _, _ = sample_surface(mesh, count=128, seed=0)
    intact = not changed_source_files(experiment) and not changed_frozen_files(experiment)
    report = build_report(runs, original, mesh, points, normals, experiment, evidence, traces, trace_attestations,
                          snapshot_intact=intact, expected_output_hashes=output_hashes)
    _write_exclusive(args.output, report)
    print(json.dumps(dict(output=str(args.output), full_comparison_complete=report['full_comparison_complete'],
        continuation_gate_passed=report['continuation_gate_passed'],
        independent_accuracy_verified=report['independent_accuracy_verified'],
        overall_gate_passed=report['overall_gate_passed']), indent=2))


if __name__ == '__main__':
    main()
