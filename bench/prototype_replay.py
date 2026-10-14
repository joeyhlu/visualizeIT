"""Replay frozen dataset RGB frames through the local prototype service.

This records offline request/response evidence only. It sends no reference pose,
depth, visibility annotation, or mask, and it does not report live-camera FPS.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import ssl
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
SPRINT_ROOT = ROOT/'.cache'/'prototype-sprint'
DEFAULT_OUTPUT = SPRINT_ROOT/'service-replay-v1.json'
FRAME_COUNT = 110
MAX_OBJECTS = 5
MAX_REPLAY_FRAMES = 550
MAX_REQUEST_BYTES = 8*1024*1024
MAX_JPEG_BYTES = 6*1024*1024
MAX_RESPONSE_BYTES = 1024*1024
MAX_SOURCE_IMAGE_BYTES = 32*1024*1024
MAX_IMAGE_PIXELS = 3840*2160
REQUEST_TIMEOUT_SECONDS = 15.0
JPEG_QUALITY = 85
STALL_THRESHOLD_MS = 1000.0
FRAME_FIELDS = frozenset({
    'sessionId', 'frameId', 'logicalIndex', 'captureTimestampSeconds', 'timestampKind',
    'objectId', 'referenceId', 'calibrationId', 'width', 'height', 'rgbJPEGBase64',
})
POSE_STATES = frozenset({'tracking', 'recovering', 'limited', 'lost'})
SE3_CONVENTION = 'CV X right, Y down, Z forward; column vectors'


class ReplayError(RuntimeError):
    """Frozen replay inputs are inconsistent or outside the bounded replay contract."""


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON value: {value}')


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReplayError(f'Cannot read frozen JSON input: {Path(path).name}') from exc


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def _inside(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if path == root or root not in path.parents:
        raise ReplayError('Frozen resource escaped its declared directory.')
    return path


def _resource_path(root, spec):
    if not isinstance(spec, dict) or spec.get('available') is not True:
        raise ReplayError('RGB member is marked unavailable in the frozen manifest.')
    relative = spec.get('path')
    expected = spec.get('sha256')
    if (not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative
            or ':' in relative or not isinstance(expected, str)
            or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
        raise ReplayError('RGB resource must have a safe relative path and SHA-256.')
    member = PurePosixPath(relative)
    if member.is_absolute() or any(part in ('', '.', '..') for part in member.parts):
        raise ReplayError('RGB resource path is not a safe archive member.')
    target = _inside(root, Path(root).joinpath(*member.parts))
    if not target.is_file():
        raise ReplayError('RGB file is unavailable in the frozen cache.')
    if target.stat().st_size > MAX_SOURCE_IMAGE_BYTES:
        raise ReplayError('RGB source image exceeds the 32 MiB local-read limit.')
    if _sha256_file(target) != expected:
        raise ReplayError('RGB source hash differs from the frozen inference manifest.')
    return target


def _matrix_k(value):
    try:
        flat = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayError('Frame calibration has an invalid camera matrix.') from exc
    if len(flat) != 9 or not all(math.isfinite(item) for item in flat):
        raise ReplayError('Frame calibration must contain nine finite intrinsics values.')
    matrix = [flat[0:3], flat[3:6], flat[6:9]]
    if (matrix[0][0] <= 0 or matrix[1][1] <= 0
            or abs(matrix[0][1]) > 1e-10 or abs(matrix[1][0]) > 1e-10
            or any(abs(a-b) > 1e-10 for a, b in zip(matrix[2], [0., 0., 1.]))):
        raise ReplayError('Frame calibration is not supported pinhole intrinsics.')
    return flat


def _validate_frame_sequence(obj):
    onboarding = obj.get('onboarding')
    frames = obj.get('frames')
    if not isinstance(onboarding, list) or len(onboarding) != 10:
        raise ReplayError(f'Object {obj.get("objectId")} lacks its frozen 10-frame onboarding set.')
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ReplayError(f'Object {obj.get("objectId")} must replay all 110 scored frames.')
    onboarding_ids = [row.get('frameId') for row in onboarding]
    frame_ids = [row.get('frameId') for row in frames]
    if (any(type(value) is not int for value in onboarding_ids + frame_ids)
            or onboarding_ids != list(range(onboarding_ids[0], onboarding_ids[0] + 10))
            or frame_ids != list(range(frame_ids[0], frame_ids[0] + FRAME_COUNT))
            or onboarding_ids[-1] + 1 != frame_ids[0]):
        raise ReplayError(f'Object {obj.get("objectId")} has a nonconsecutive frozen timeline.')
    allowed = {'frameId', 'logicalIndex', 'timestampSeconds', 'timestampKind',
               'rgb', 'calibration', 'width', 'height', 'sessionId'}
    previous_k = None
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or set(row)-allowed:
            raise ReplayError('Inference frame contains evaluator or unsupported fields.')
        if (row.get('logicalIndex') != index + 10
                or row.get('timestampKind') != 'frame_index_no_fps'
                or row.get('timestampSeconds') is not None):
            raise ReplayError('Replay requires the frozen frame-index clock with null timestamps.')
        calibration = row.get('calibration')
        if not isinstance(calibration, dict) or 'cam_K' not in calibration:
            raise ReplayError('Inference frame has no native camera calibration.')
        current_k = _matrix_k(calibration['cam_K'])
        if previous_k is not None and any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                                           for a, b in zip(previous_k, current_k)):
            raise ReplayError('Per-object camera intrinsics vary; one frozen dataset calibration cannot represent them.')
        previous_k = current_k
    return onboarding_ids, frames


def _load_frozen_inputs(dataset, results):
    dataset = Path(dataset).resolve()
    results = Path(results).resolve()
    inference_path = dataset/'inference.json'
    freeze_path = results/'freeze.json'
    seal_path = results/'inference-seal.json'
    if not all(path.is_file() for path in (inference_path, freeze_path, seal_path)):
        raise ReplayError('Replay requires inference.json, freeze.json, and inference-seal.json.')
    inference = _read_json(inference_path)
    freeze = _read_json(freeze_path)
    seal = _read_json(seal_path)
    inference_hash = _sha256_file(inference_path)
    if freeze.get('inferenceManifestSHA256') != inference_hash:
        raise ReplayError('Inference manifest does not match the sealed freeze.')
    if seal.get('freezeSHA256') != _sha256_file(freeze_path):
        raise ReplayError('Inference freeze does not match the output seal.')
    if (inference.get('benchmarkId') != seal.get('benchmarkId')
            or inference.get('selectionDigest') != seal.get('selectionDigest')
            or inference.get('evaluatorManifestSHA256') != seal.get('evaluatorManifestSHA256')):
        raise ReplayError('Dataset and sealed inference result identities differ.')
    for key in ('evidenceKind', 'benchmarkMode', 'plannedObjects',
                'realUnseenObjects', 'tenObjectsEvaluated'):
        if key in freeze and freeze[key] != inference.get(key):
            raise ReplayError(f'Frozen run {key} differs from the inference manifest.')
        if key in seal and seal[key] != inference.get(key):
            raise ReplayError(f'Inference seal {key} differs from the inference manifest.')
    if seal.get('evaluatorOpenedDuringInference') is not False:
        raise ReplayError('Frozen inference seal does not prove evaluator isolation.')
    if inference.get('evidenceKind') != 'dataset_reference' or inference.get('benchmarkMode') != 'pilot':
        raise ReplayError('Offline service replay accepts only the frozen dataset-reference pilot.')
    objects = inference.get('objects')
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ReplayError('Offline pilot is bounded to one through five objects.')
    if (inference.get('plannedObjects') != len(objects)
            or inference.get('realUnseenObjects') != len(objects)
            or inference.get('tenObjectsEvaluated') is not False
            or inference.get('evaluationFrames') != FRAME_COUNT):
        raise ReplayError('Frozen manifest does not declare its pilot scope and complete scored window.')
    total_frames = len(objects)*FRAME_COUNT
    if total_frames > MAX_REPLAY_FRAMES:
        raise ReplayError('Offline replay exceeds its 550-frame cap.')
    sealed_rows = [row for row in seal.get('objects', []) if isinstance(row, dict)]
    sealed_ids = [row.get('objectId') for row in sealed_rows]
    object_ids = [row.get('objectId') for row in objects if isinstance(row, dict)]
    if (len(object_ids) != len(objects) or len(set(object_ids)) != len(object_ids)
            or len(sealed_ids) != len(set(sealed_ids)) or set(object_ids) != set(sealed_ids)):
        raise ReplayError('Inference objects differ from the sealed result objects.')

    asset_map = {}
    for obj in objects:
        object_id = obj.get('objectId')
        if type(object_id) is not int or object_id < 1:
            raise ReplayError('Inference object ID is invalid.')
        onboarding_ids, frames = _validate_frame_sequence(obj)
        asset_dir = results/f'object-{object_id:06d}'
        asset_json_path = _inside(results, asset_dir/'asset.json')
        if not asset_json_path.is_file():
            raise ReplayError(f'Frozen tracker asset is missing for object {object_id}.')
        asset = _read_json(asset_json_path)
        if (asset.get('objectId') != object_id or asset.get('name') != obj.get('name')
                or asset.get('evidenceKind') != 'dataset_reference'
                or asset.get('inferenceManifestSHA256') != inference_hash
                or asset.get('units') != 'metres'
                or asset.get('onboardingFrameIds') != onboarding_ids):
            raise ReplayError(f'Frozen tracker asset identity/onboarding mismatch for object {object_id}.')
        asset_hash = asset.get('assetSHA256')
        asset_name = asset.get('assetPath')
        if (not isinstance(asset_hash, str) or re.fullmatch(r'[0-9a-f]{64}', asset_hash) is None
                or not isinstance(asset_name, str) or Path(asset_name).name != asset_name
                or '/' in asset_name or '\\' in asset_name):
            raise ReplayError(f'Frozen tracker asset reference is invalid for object {object_id}.')
        asset_path = _inside(asset_dir, asset_dir/asset_name)
        if not asset_path.is_file() or _sha256_file(asset_path) != asset_hash:
            raise ReplayError(f'Frozen tracker asset hash mismatch for object {object_id}.')
        asset_map[object_id] = {
            'name': obj['name'], 'assetSHA256': asset_hash,
            'assetPath': str(asset_path), 'frames': frames,
        }
    return dataset, results, inference, freeze, seal, asset_map


def _service_target(service_url, ca_file=None):
    try:
        parsed = urlsplit(service_url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError('Service URL has an invalid authority or port.') from exc
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
        raise ValueError('Service URL must be an HTTP(S) loopback origin without credentials or path.')
    host = parsed.hostname.lower()
    if host != 'localhost':
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError('Service URL must use localhost or an explicit loopback IP.') from exc
        if address.version != 4 or not address.is_loopback:
            raise ValueError('Service URL must use an IPv4 loopback address.')
        host = str(address)
    if parsed.scheme == 'https':
        if ca_file is None:
            raise ValueError('HTTPS replay requires an exact --ca-file.')
        ca_path = Path(ca_file).expanduser().resolve()
        if not ca_path.is_file():
            raise ValueError('The specified HTTPS CA file does not exist.')
        context = ssl.create_default_context(cafile=str(ca_path))
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    else:
        if ca_file is not None:
            raise ValueError('--ca-file is valid only with an HTTPS service URL.')
        context = None
    return {
        'scheme': parsed.scheme, 'host': host,
        'port': port or (443 if parsed.scheme == 'https' else 80),
        'path': '/v1/frames', 'sslContext': context,
    }


def _make_packet(session_id, object_id, reference_id, calibration_id,
                 frame, width, height, jpeg_bytes):
    """Construct only the service's explicit RGB/identity whitelist."""
    packet = {
        'sessionId': session_id,
        'frameId': frame['frameId'],
        'logicalIndex': frame['logicalIndex'],
        'captureTimestampSeconds': None,
        'timestampKind': 'frame_index_no_fps',
        'objectId': object_id,
        'referenceId': reference_id,
        'calibrationId': calibration_id,
        'width': width,
        'height': height,
        'rgbJPEGBase64': base64.b64encode(jpeg_bytes).decode('ascii'),
    }
    if set(packet) != FRAME_FIELDS:
        raise ReplayError('Internal frame packet exceeded the service field whitelist.')
    return packet


def _encode_rgb(dataset, spec, quality=JPEG_QUALITY):
    from PIL import Image, UnidentifiedImageError
    prepare_started = time.perf_counter()
    path = _resource_path(dataset, spec)
    try:
        with Image.open(path) as source:
            if source.format not in ('PNG', 'JPEG'):
                raise ReplayError('Frozen RGB must be a PNG or JPEG image.')
            width, height = source.size
            if width <= 0 or height <= 0 or width*height > MAX_IMAGE_PIXELS:
                raise ReplayError('Frozen RGB exceeds the service image-pixel limit.')
            source.load()
            image = source.convert('RGB')
        preparation_ms = (time.perf_counter()-prepare_started)*1000.
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ReplayError('Frozen RGB could not be decoded.') from exc
    import io
    stream = io.BytesIO()
    encode_started = time.perf_counter()
    image.save(stream, format='JPEG', quality=quality, optimize=False)
    encode_ms = (time.perf_counter()-encode_started)*1000.
    jpeg = stream.getvalue()
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ReplayError('Encoded JPEG is empty or exceeds the service 6 MiB limit.')
    return width, height, jpeg, preparation_ms, encode_ms


def _finite_pose(value):
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(not isinstance(row, list) or len(row) != 4 for row in value):
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(float(item)) for row in value for item in row):
        return False
    m = [[float(item) for item in row] for row in value]
    if any(abs(a-b) > 1e-6 for a, b in zip(m[3], [0., 0., 0., 1.])):
        return False
    r = [row[:3] for row in m[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i]*r[k][j] for k in range(3))
            if abs(dot - (1. if i == j else 0.)) > 1e-4:
                return False
    det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
           - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
           + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    return abs(det-1.) <= 1e-4


def _interpret_response(payload, status, expected):
    if not isinstance(payload, dict):
        return {'responseIdentityValid': False, 'poseState': None,
                'failureReason': 'invalid_json_response', 'publicPoseValid': False,
                'freshValidPublicPose': False}
    identity = ('sessionId', 'objectId', 'referenceId', 'calibrationId',
                'frameId', 'logicalIndex', 'timestampKind', 'captureTimestampSeconds')
    identity_valid = all(payload.get(key) == expected[key] for key in identity)
    state = payload.get('poseState')
    state = state if isinstance(state, str) and state in POSE_STATES else None
    pose_valid = (state == 'tracking' and payload.get('units') == 'metres'
                  and payload.get('cameraConvention') == SE3_CONVENTION
                  and _finite_pose(payload.get('cameraFromObject')))
    failure = payload.get('failureReason')
    if status != 200:
        failure = f'http_{status}' if not isinstance(failure, str) else failure
    elif not identity_valid:
        failure = 'response_identity_mismatch'
    elif state is None:
        failure = 'invalid_pose_state'
    elif state == 'tracking' and not pose_valid:
        failure = 'invalid_public_pose'
    elif state != 'tracking' and payload.get('cameraFromObject') is not None:
        failure = 'nontracking_response_contains_pose'
    return {
        'responseIdentityValid': identity_valid,
        'poseState': state,
        'failureReason': failure,
        'publicPoseValid': bool(pose_valid),
        # Fresh means the response is identity-bound to this exact replay request.
        # It is not a claim that a live camera frame was freshly captured.
        'freshValidPublicPose': bool(status == 200 and identity_valid and pose_valid),
    }


def _exchange(target, body, timeout_seconds):
    connection_type = (http.client.HTTPSConnection if target['scheme'] == 'https'
                       else http.client.HTTPConnection)
    kwargs = {'timeout': timeout_seconds}
    if target['scheme'] == 'https':
        kwargs['context'] = target['sslContext']
    connection = connection_type(target['host'], target['port'], **kwargs)
    started = time.perf_counter()
    status = None
    try:
        connection.request('POST', target['path'], body=body,
                          headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                                   'Connection': 'close'})
        response = connection.getresponse()
        status = response.status
        raw = response.read(MAX_RESPONSE_BYTES+1)
        latency_ms = (time.perf_counter()-started)*1000.
        if len(raw) > MAX_RESPONSE_BYTES:
            return status, None, latency_ms, 'response_too_large', True
        try:
            payload = json.loads(raw.decode('utf-8'), parse_constant=_reject_constant)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return status, None, latency_ms, 'invalid_json_response', True
        return status, payload, latency_ms, None, True
    except (OSError, TimeoutError, socket.timeout, http.client.HTTPException, ssl.SSLError) as exc:
        return status, None, (time.perf_counter()-started)*1000., type(exc).__name__, False
    finally:
        connection.close()


def _percentile(values, fraction):
    rows = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not rows:
        return None
    index = (len(rows)-1)*fraction
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return rows[low]
    return rows[low] + (rows[high]-rows[low])*(index-low)


def _latency_summary(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {'count': len(values), 'p50Ms': _percentile(values, .50),
            'p95Ms': _percentile(values, .95), 'maxMs': max(values) if values else None}


def _rate_per_second(count, wall_seconds):
    if (wall_seconds is None or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(float(wall_seconds)) or wall_seconds <= 0):
        return None
    return count/float(wall_seconds)


def _availability(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'fraction': numerator/denominator if denominator else None}


def _progress_journal_path(output):
    output = Path(output)
    return output.with_name(output.stem+'.frames.jsonl')


def _append_progress_row(stream, sequence, replay_started, row):
    entry = {
        'sequence': sequence,
        'elapsedSinceReplayStartSeconds': max(0., time.perf_counter()-replay_started),
        **row,
    }
    stream.write(_strict_json_bytes(entry).decode('utf-8')+'\n')
    stream.flush()


def _validate_output_path(output, dataset, results):
    output = Path(output).expanduser().resolve()
    sprint_root = SPRINT_ROOT.resolve()
    if output.suffix.lower() != '.json' or output == sprint_root or sprint_root not in output.parents:
        raise ValueError('Replay report must be a .json file inside .cache/prototype-sprint.')
    for source_root in (Path(dataset).resolve(), Path(results).resolve()):
        if output == source_root or source_root in output.parents:
            raise ValueError('Replay report may not be written inside the dataset or result tree.')
    if output.exists():
        raise FileExistsError(f'Replay output already exists; refusing overwrite: {output.name}')
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError('Replay output parent is not a directory.')
    return output


def _validate_journal_path(output, dataset, results):
    journal = _progress_journal_path(output).expanduser().resolve()
    sprint_root = SPRINT_ROOT.resolve()
    if journal.suffix.lower() != '.jsonl' or sprint_root not in journal.parents:
        raise ValueError('Replay progress journal must be a .jsonl file inside .cache/prototype-sprint.')
    for source_root in (Path(dataset).resolve(), Path(results).resolve()):
        if journal == source_root or source_root in journal.parents:
            raise ValueError('Replay progress journal may not be written inside the dataset or result tree.')
    if journal.exists():
        raise FileExistsError(f'Replay progress journal already exists; refusing overwrite: {journal.name}')
    return journal


def _write_new_json(output, document, dataset, results):
    output = _validate_output_path(output, dataset, results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return output


def _summarize_object(frames, wall_seconds=None):
    state_counts = Counter(row['poseState'] for row in frames if row.get('poseState'))
    reason_counts = Counter(row['failureReason'] for row in frames
                            if row.get('failureReason'))
    status_counts = Counter(str(row['httpStatus']) for row in frames
                            if row.get('httpStatus') is not None)
    requests = [row for row in frames if row.get('requestAttempted')]
    responses = [row for row in frames if row.get('responseComplete')]
    network = [row['networkLatencyMs'] for row in requests
               if row.get('networkLatencyMs') is not None]
    end_to_end = [row['endToEndLatencyMs'] for row in frames
                  if row.get('endToEndLatencyMs') is not None]
    encode = [row['jpegEncodeLatencyMs'] for row in frames
              if row.get('jpegEncodeLatencyMs') is not None]
    preparation = [row['rgbPreparationLatencyMs'] for row in frames
                   if row.get('rgbPreparationLatencyMs') is not None]
    scheduled = len(frames)
    fresh_poses = sum(bool(row.get('freshValidPublicPose')) for row in frames)
    return {
        'scheduledFrames': scheduled,
        'requestAttempts': len(requests),
        'completedResponses': len(responses),
        'http200Responses': sum(row.get('httpStatus') == 200 for row in responses),
        'droppedFrames': sum(row.get('httpStatus') != 200 or not row.get('responseComplete')
                             for row in requests),
        'locallySkippedFrames': sum(not row.get('requestAttempted') for row in frames),
        'freshValidPublicPoses': fresh_poses,
        'replayWallSeconds': wall_seconds,
        'completedResponsesPerSecond': _rate_per_second(len(responses), wall_seconds),
        'freshValidPublicPosesPerSecond': _rate_per_second(fresh_poses, wall_seconds),
        'completedResponseAvailability': _availability(len(responses), scheduled),
        'allFrameAvailability': _availability(fresh_poses, scheduled),
        'http429Count': sum(row.get('httpStatus') == 429 for row in frames),
        'timeoutCount': sum(row.get('transportFailureType') in ('TimeoutError', 'socket.timeout')
                            for row in frames),
        'stallCount': sum((row.get('endToEndLatencyMs') or 0.) >= STALL_THRESHOLD_MS
                          for row in requests),
        'stallThresholdMs': STALL_THRESHOLD_MS,
        'httpStatusCounts': dict(sorted(status_counts.items())),
        'serviceStateCounts': dict(sorted(state_counts.items())),
        'failureReasonCounts': dict(sorted(reason_counts.items())),
        'rgbPreparationLatency': _latency_summary(preparation),
        'jpegEncodeLatency': _latency_summary(encode),
        'networkLatency': _latency_summary(network),
        'endToEndLatency': _latency_summary(end_to_end),
    }


def replay(dataset, results, service_url, output=DEFAULT_OUTPUT, *, ca_file=None,
           timeout_seconds=REQUEST_TIMEOUT_SECONDS, jpeg_quality=JPEG_QUALITY):
    """Sequentially replay all scored pilot RGB records once, retaining failures."""
    if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 120.:
        raise ValueError('Request timeout must be between 0.1 and 120 seconds.')
    if type(jpeg_quality) is not int or not 40 <= jpeg_quality <= 95:
        raise ValueError('JPEG quality must be an integer from 40 to 95.')
    dataset_root, results_root, inference, freeze, seal, assets = _load_frozen_inputs(dataset, results)
    output = _validate_output_path(output, dataset_root, results_root)
    journal_path = _validate_journal_path(output, dataset_root, results_root)
    target = _service_target(service_url, ca_file)
    all_rows = []
    object_results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open('x', encoding='utf-8', newline='\n') as journal_stream:
        replay_started = time.perf_counter()
        for object_id in sorted(assets):
            asset = assets[object_id]
            calibration_id = f'dataset-{object_id}'
            session_id = f'offline-replay-{inference["benchmarkId"]}-{object_id}-{uuid.uuid4().hex[:12]}'
            object_rows = []
            object_started = time.perf_counter()
            expected_size = None
            for frame in asset['frames']:
                started = time.perf_counter()
                row = {
                    'objectId': object_id, 'frameId': frame['frameId'],
                    'logicalIndex': frame['logicalIndex'], 'requestAttempted': False,
                    'responseComplete': False, 'httpStatus': None,
                    'poseState': None, 'failureReason': None,
                    'freshValidPublicPose': False, 'publicPoseValid': False,
                    'rgbPreparationLatencyMs': None, 'jpegEncodeLatencyMs': None,
                    'networkLatencyMs': None,
                    'endToEndLatencyMs': None, 'transportFailureType': None,
                }
                try:
                    width, height, jpeg, preparation_ms, encode_ms = _encode_rgb(
                        dataset_root, frame['rgb'], jpeg_quality)
                    row['rgbPreparationLatencyMs'] = preparation_ms
                    row['jpegEncodeLatencyMs'] = encode_ms
                    if expected_size is None:
                        expected_size = (width, height)
                    elif (width, height) != expected_size:
                        raise ReplayError('RGB resolution changed within one dataset calibration identity.')
                    packet = _make_packet(session_id, object_id, asset['assetSHA256'],
                        calibration_id, frame, width, height, jpeg)
                    body = _strict_json_bytes(packet)
                    if len(body) > MAX_REQUEST_BYTES:
                        raise ReplayError('Serialized frame exceeds the service 8 MiB request limit.')
                except (OSError, ValueError, ReplayError, TypeError, KeyError) as exc:
                    row['failureReason'] = f'local_input_or_encoding:{type(exc).__name__}:{str(exc)[:200]}'
                    row['endToEndLatencyMs'] = (time.perf_counter()-started)*1000.
                    object_rows.append(row)
                    all_rows.append(row)
                    _append_progress_row(journal_stream, len(all_rows)-1, replay_started, row)
                    continue

                row['requestAttempted'] = True
                http_status, payload, network_ms, transport_failure, response_complete = _exchange(
                    target, body, timeout_seconds)
                row['httpStatus'] = http_status
                row['networkLatencyMs'] = network_ms
                row['transportFailureType'] = transport_failure
                row['responseComplete'] = bool(response_complete and isinstance(payload, dict))
                row['endToEndLatencyMs'] = (time.perf_counter()-started)*1000.
                if payload is None:
                    row['failureReason'] = transport_failure or 'response_unavailable'
                else:
                    interpreted = _interpret_response(payload, http_status, packet)
                    row.update(interpreted)
                    if not row['failureReason'] and http_status != 200:
                        row['failureReason'] = f'http_{http_status}'
                object_rows.append(row)
                all_rows.append(row)
                _append_progress_row(journal_stream, len(all_rows)-1, replay_started, row)

            object_wall_seconds = time.perf_counter()-object_started
            object_results.append({
                'objectId': object_id, 'name': asset['name'],
                'referenceId': asset['assetSHA256'], 'calibrationId': calibration_id,
                **_summarize_object(object_rows, object_wall_seconds), 'frames': object_rows,
            })
        replay_wall_seconds = time.perf_counter()-replay_started

    summary = _summarize_object(all_rows, replay_wall_seconds)
    report = {
        'schemaVersion': 1,
        'mode': 'offline_dataset_service_replay',
        'label': 'Offline recorded dataset RGB replay through the local prototype service.',
        'evidenceKind': inference['evidenceKind'],
        'benchmarkMode': inference['benchmarkMode'],
        'benchmarkId': inference['benchmarkId'],
        'selectionDigest': inference['selectionDigest'],
        'inferenceManifestSHA256': freeze['inferenceManifestSHA256'],
        'inferenceFreezeSHA256': seal['freezeSHA256'],
        'serviceOrigin': f"{target['scheme']}://{target['host']}:{target['port']}",
        'endpointPath': target['path'],
        'clockKind': 'frame_index_no_fps',
        'timestampSeconds': None,
        'neural_updates_per_second': 0,
        'liveCameraFPSMeasured': False,
        'noLiveCameraFPSClaim': True,
        'freshPoseMeaning': 'Response identity matches this replay frame and the public SE(3) pose validates; this does not establish a fresh live camera capture.',
        'timeoutSeconds': timeout_seconds,
        'jpegQuality': jpeg_quality,
        'timingScope': 'Measured wall time covers the sequential replay loop from first frame preparation through final progress-journal append; it excludes frozen-input validation and final report serialization.',
        'replayWallSeconds': replay_wall_seconds,
        'completedResponsesPerSecond': summary['completedResponsesPerSecond'],
        'freshValidPublicPosesPerSecond': summary['freshValidPublicPosesPerSecond'],
        'progressJournal': journal_path.name,
        'progressJournalRows': len(all_rows),
        'totalScheduledFrames': len(all_rows),
        'maximumScheduledFrames': MAX_REPLAY_FRAMES,
        'objects': object_results,
        'summary': summary,
    }
    output_path = _write_new_json(output, report, dataset_root, results_root)
    return {
        'status': 'complete', 'output': str(output_path),
        'totalScheduledFrames': len(all_rows),
        'requestAttempts': summary['requestAttempts'],
        'completedResponses': summary['completedResponses'],
        'freshValidPublicPoses': summary['freshValidPublicPoses'],
        'replayWallSeconds': replay_wall_seconds,
        'completedResponsesPerSecond': summary['completedResponsesPerSecond'],
        'freshValidPublicPosesPerSecond': summary['freshValidPublicPosesPerSecond'],
        'allFrameAvailability': summary['allFrameAvailability'],
        'progressJournal': str(journal_path),
        'droppedFrames': summary['droppedFrames'],
        'http429Count': summary['http429Count'],
        'timeoutCount': summary['timeoutCount'],
        'neural_updates_per_second': 0,
        'noLiveCameraFPSClaim': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Frozen pilot dataset containing inference.json and RGB members.')
    parser.add_argument('--results', type=Path, required=True,
                        help='Frozen results directory containing freeze/seal and object asset.json files.')
    parser.add_argument('--url', required=True,
                        help='HTTP(S) loopback service origin; requests go only to /v1/frames.')
    parser.add_argument('--ca-file', type=Path,
                        help='Exact CA file required for HTTPS; invalid for plain HTTP.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Fresh private JSON report path inside .cache/prototype-sprint.')
    parser.add_argument('--timeout-seconds', type=float, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument('--jpeg-quality', type=int, default=JPEG_QUALITY)
    args = parser.parse_args(argv)
    receipt = replay(args.dataset, args.results, args.url, args.output,
                     ca_file=args.ca_file, timeout_seconds=args.timeout_seconds,
                     jpeg_quality=args.jpeg_quality)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt


if __name__ == '__main__':
    main()
