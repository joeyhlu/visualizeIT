"""Serve the local RGB-only pose prototype over bounded HTTP(S).

The default listener is loopback-only. Private LAN listeners require an explicit
TLS certificate and key; public and wildcard binds are rejected. No endpoint
accepts evaluation labels, depth, masks, or reference poses.

Example (after benchmark onboarding has produced tracking assets)::

    python -B -S tools/prototype_bootstrap.py bench.prototype_service \
        --assets .cache/prototype-sprint/results-v1 \
        --calibrations .cache/prototype-sprint/calibrations.json
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import ipaddress
import io
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
import re
import ssl
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlsplit
import zipfile

import numpy as np
from PIL import Image, UnidentifiedImageError

from .prototype_tracking import FrameInputV1, PrototypeTrackerV1, is_proper_rotation, project_points_cv


ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_JPEG_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 3840 * 2160
MAX_ASSET_BYTES = 256 * 1024 * 1024
MAX_ASSET_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_WIREFRAME_SEGMENTS = 384
MAX_SESSIONS = 8
MAX_HTTP_WORKERS = 8
SESSION_IDLE_SECONDS = 30 * 60
WORKER_SOCKET_TIMEOUT_SECONDS = 10.
MAX_STATIC_BYTES = 64 * 1024 * 1024
PUBLIC_DEMO_EXTENSIONS = {'.html', '.css', '.js', '.mjs', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.woff', '.woff2'}
FRAME_FIELDS = {
    'sessionId', 'frameId', 'logicalIndex', 'captureTimestampSeconds', 'timestampKind',
    'objectId', 'referenceId', 'calibrationId', 'width', 'height', 'rgbJPEGBase64',
    'cameraFrameId', 'cameraMediaTime',
}
REQUIRED_FRAME_FIELDS = FRAME_FIELDS-{'cameraFrameId', 'cameraMediaTime'}
CALIBRATION_KINDS = {'physical_measured', 'dataset_reference', 'synthetic'}
TRACKING_STATES = {'tracking', 'recovering', 'limited', 'lost'}
ASSET_ARRAYS = {
    'referencePoints', 'referenceDescriptors', 'modelPositions', 'positions',
    'normals', 'triangles', 'uv', 'benchFromSource', 'schemaVersion',
}


def _reject_json_constant(value):
    raise ValueError(f'Nonfinite JSON number is forbidden: {value}')


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite JSON number is forbidden.')
    return number


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_json_constant,
                          parse_float=_finite_json_float)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read JSON registry: {Path(path).name}') from exc


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_int(value, label, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{label} must be an integer in range.')
    return value


def _identifier(value, label, maximum=128):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{label} must be a nonempty bounded string.')
    return value


def _matrix_k(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
            matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
            abs(matrix[0, 1]) > 1e-10 or abs(matrix[1, 0]) > 1e-10 or
            not np.allclose(matrix[2], [0., 0., 1.], atol=1e-10, rtol=0) or
            abs(float(np.linalg.det(matrix))) <= 1e-12):
        raise ValueError('Calibration K must be finite nonsingular pinhole intrinsics.')
    return matrix.copy()


def _inside(path, root):
    return path == root or root in path.parents


@dataclass(frozen=True)
class AssetRecord:
    object_id: int
    name: str
    source_basis: str
    evidence_kind: str
    reference_id: str
    metadata_path: Path
    asset_path: Path
    reference_points: np.ndarray
    reference_descriptors: np.ndarray
    model_positions: np.ndarray
    triangles: np.ndarray
    wire_edges: np.ndarray

    def public(self):
        return {'objectId': self.object_id, 'referenceId': self.reference_id, 'name': self.name,
                'sourceBasis': self.source_basis, 'evidenceKind': self.evidence_kind}


@dataclass(frozen=True)
class CalibrationRecord:
    calibration_id: str
    width: int
    height: int
    k: np.ndarray
    kind: str
    object_id: int | None = None
    name: str | None = None

    def public(self):
        result = {
            'calibrationId': self.calibration_id,
            'width': self.width,
            'height': self.height,
            'kind': self.kind,
            'provenance': self.kind,
        }
        if self.object_id is not None:
            result['objectId'] = self.object_id
        if self.name:
            result['name'] = self.name
        return result


def _wire_edges(triangles):
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges) > MAX_WIREFRAME_SEGMENTS:
        keep = np.linspace(0, len(edges) - 1, MAX_WIREFRAME_SEGMENTS, dtype=np.int64)
        edges = edges[keep]
    return edges


def load_asset_registry(assets_root):
    """Load only direct ``object-*/asset.json`` children and verify each NPZ hash."""
    root = Path(assets_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Asset registry root must be an existing directory.')
    records = {}
    for listed_metadata in sorted(root.glob('object-*/asset.json')):
        metadata_path = listed_metadata.resolve()
        directory = listed_metadata.parent.resolve()
        if not _inside(metadata_path, root) or not _inside(directory, root):
            raise ValueError('Asset registry entry escaped its fixed root.')
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError('Asset metadata must be a JSON object.')
        object_id = _json_int(metadata.get('objectId'), 'asset objectId', 1)
        if directory.name != f'object-{object_id:06d}':
            raise ValueError('Asset directory and object identity disagree.')
        if object_id in records:
            raise ValueError('Duplicate object identity in asset registry.')
        name = _identifier(metadata.get('name'), 'asset name', 160)
        source_basis = _identifier(metadata.get('sourceBasis'), 'asset sourceBasis', 200)
        evidence_kind = _identifier(metadata.get('evidenceKind', 'dataset_reference'),
                                    'asset evidenceKind', 64)
        if evidence_kind not in ('dataset_reference', 'synthetic_control'):
            raise ValueError('Asset evidenceKind must be dataset_reference or synthetic_control.')
        if metadata.get('units') != 'metres':
            raise ValueError('Asset coordinates must explicitly use metres.')
        asset_name = metadata.get('assetPath')
        if (not isinstance(asset_name, str) or not asset_name or
                Path(asset_name).name != asset_name or '/' in asset_name or '\\' in asset_name):
            raise ValueError('Asset path must be a single file inside its object directory.')
        expected_hash = metadata.get('assetSHA256')
        if not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_hash):
            raise ValueError('Asset metadata must contain a SHA-256 digest.')
        asset_path = (directory / asset_name).resolve()
        if not _inside(asset_path, directory) or not asset_path.is_file():
            raise ValueError('Asset archive is missing or escaped its object directory.')
        if asset_path.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError('Asset archive exceeds the service size limit.')
        actual_hash = _sha256_file(asset_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError('Asset archive SHA-256 does not match asset.json.')
        try:
            with zipfile.ZipFile(asset_path) as archive:
                names = set(archive.namelist())
                if names != {name + '.npy' for name in ASSET_ARRAYS}:
                    raise ValueError('Asset archive has an unexpected array set.')
                if sum(info.file_size for info in archive.infolist()) > MAX_ASSET_UNCOMPRESSED_BYTES:
                    raise ValueError('Asset archive expands beyond the service size limit.')
            with np.load(asset_path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in ASSET_ARRAYS}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith('Asset archive'):
                raise
            raise ValueError('Asset archive is unreadable or malformed.') from exc
        if (arrays['schemaVersion'].shape != () or
                not np.issubdtype(arrays['schemaVersion'].dtype, np.integer) or
                int(arrays['schemaVersion']) != 1):
            raise ValueError('Asset archive version is unsupported.')
        refs = arrays['referencePoints']
        desc = arrays['referenceDescriptors']
        model = arrays['modelPositions']
        positions = arrays['positions']
        normals = arrays['normals']
        triangles = arrays['triangles']
        uv = arrays['uv']
        transform = arrays['benchFromSource']
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 4:
            raise ValueError('Asset geometry or reference arrays have invalid shapes.')
        if (refs.ndim != 2 or refs.shape[1:] != (3,) or len(refs) < 12 or
                desc.shape != (len(refs), 32) or desc.dtype != np.uint8 or
                model.ndim != 2 or model.shape[1:] != (3,) or len(model) != len(positions) or len(model) < 4 or
                normals.shape != positions.shape or uv.shape != (len(positions), 2) or
                triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) == 0 or
                not np.issubdtype(triangles.dtype, np.integer) or
                (triangles < 0).any() or (triangles >= len(positions)).any() or
                transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0)):
            raise ValueError('Asset geometry or reference arrays have invalid shapes.')
        numeric_arrays = (refs, model, positions, normals, uv, transform)
        if any(not np.issubdtype(array.dtype, np.number) or
               np.issubdtype(array.dtype, np.complexfloating) for array in numeric_arrays):
            raise ValueError('Asset geometry arrays must be real numeric arrays.')
        if any(not np.isfinite(array).all() for array in (refs, model, positions, normals, uv, transform)):
            raise ValueError('Asset arrays contain nonfinite values.')
        basis = transform[:3, :3]
        if (abs(float(np.linalg.det(basis))) <= 1e-12 or
                not np.allclose(basis@basis.T, np.eye(3), atol=1e-6, rtol=0) or
                not np.allclose(model@basis.T+transform[:3, 3], positions, atol=1e-7, rtol=1e-7)):
            raise ValueError('Asset source/bench vertex mapping is inconsistent.')
        for array in (refs, desc, model, triangles):
            array.setflags(write=False)
        records[object_id] = AssetRecord(
            object_id, name, source_basis, evidence_kind, actual_hash, metadata_path, asset_path,
            refs, desc, model, triangles, _wire_edges(triangles),
        )
    return records


def load_calibration_registry(calibration_path):
    """Read the versioned measured/synthetic calibration allowlist."""
    if calibration_path is None:
        return {}
    document = _read_json(calibration_path)
    if (not isinstance(document, dict) or document.get('schemaVersion') != 1 or
            not isinstance(document.get('calibrations'), list)):
        raise ValueError('Calibration registry must use schemaVersion 1 and a calibrations list.')
    records = {}
    allowed = {'calibrationId', 'width', 'height', 'K', 'cam_K', 'kind', 'objectId', 'name'}
    for row in document['calibrations']:
        if not isinstance(row, dict) or set(row)-allowed:
            raise ValueError('Calibration entry contains unsupported fields.')
        calibration_id = _identifier(row.get('calibrationId'), 'calibrationId', 128)
        width = _json_int(row.get('width'), 'calibration width', 1, MAX_IMAGE_PIXELS)
        height = _json_int(row.get('height'), 'calibration height', 1, MAX_IMAGE_PIXELS)
        if width*height > MAX_IMAGE_PIXELS:
            raise ValueError('Calibration image dimensions exceed the service pixel limit.')
        kind = row.get('kind')
        if not isinstance(kind, str) or kind not in CALIBRATION_KINDS:
            raise ValueError('Calibration kind must explicitly state its provenance.')
        if 'K' in row and 'cam_K' in row:
            raise ValueError('Calibration entry must provide only one intrinsics matrix.')
        if 'K' not in row and 'cam_K' not in row:
            raise ValueError('Calibration entry requires K.')
        k = _matrix_k(row.get('K', row.get('cam_K')))
        object_id = row.get('objectId')
        if object_id is not None:
            object_id = _json_int(object_id, 'calibration objectId', 1)
        name = row.get('name')
        if name is not None:
            name = _identifier(name, 'calibration name', 160)
        if calibration_id in records:
            raise ValueError('Duplicate calibration identity.')
        records[calibration_id] = CalibrationRecord(calibration_id, width, height, k, kind,
                                                     object_id, name)
    return records


@dataclass
class _Session:
    object_id: int
    reference_id: str
    tracker: object | None = None
    calibration_id: str | None = None
    last_frame_id: int | None = None
    last_logical_index: int | None = None
    last_timestamp: float | None = None
    clock_kind: str | None = None
    last_camera_frame_id: int | None = None
    last_camera_media_time: float | None = None
    last_access: float = 0.

    def reset(self):
        if self.tracker is not None:
            try:
                self.tracker.reset_temporal()
            except Exception:
                self.tracker = None


class PrototypeService:
    """Single-frame inference with nonblocking admission and bounded HTTP workers."""

    def __init__(self, assets_root, calibrations_path=None, demo_root=None, client_path=None,
                 *, tracker_factory=PrototypeTrackerV1, max_sessions=MAX_SESSIONS):
        self.assets_root = Path(assets_root).expanduser().resolve()
        self.assets = load_asset_registry(self.assets_root)
        self.calibrations = load_calibration_registry(calibrations_path)
        for calibration in self.calibrations.values():
            if calibration.object_id is not None and calibration.object_id not in self.assets:
                raise ValueError('Calibration references an object without a loaded asset.')
        self.demo_root = Path(demo_root).expanduser().resolve() if demo_root is not None else None
        if self.demo_root is not None and not self.demo_root.is_dir():
            raise ValueError('Demo root must be an existing directory.')
        self.client_path = (Path(client_path).expanduser().resolve() if client_path is not None
                            else ROOT/'prototype'/'client.html')
        if not self.client_path.is_file() or self.client_path.stat().st_size > MAX_STATIC_BYTES:
            raise ValueError('Prototype browser client is unavailable or too large.')
        self.tracker_factory = tracker_factory
        self.max_sessions = _json_int(max_sessions, 'max_sessions', 1, 64)
        self.sessions = OrderedDict()
        self.lock = threading.RLock()
        self.frame_admission = threading.Lock()
        self.admission_lock = threading.Lock()
        self.pending_reset_sessions = set()
        self.pending_reset_all = False
        self.frame_requests = 0
        self.accepted_frame_requests = 0
        self.busy_429_responses = 0

    def status(self, server_metrics=None):
        with self.admission_lock:
            frame_metrics = {
                'frameRequests': self.frame_requests,
                'acceptedFrameRequests': self.accepted_frame_requests,
                'busy429Responses': self.busy_429_responses,
            }
        worker_metrics = {
            'connectionAdmissionDrops': 0,
            'activeWorkers': 0,
            'maxWorkers': MAX_HTTP_WORKERS,
        }
        if server_metrics is not None:
            worker_metrics.update(server_metrics)
        return {
            'apiVersion': 'prototype-service/1',
            'assets': [asset.public() for _, asset in sorted(self.assets.items())],
            'calibrations': [entry.public() for _, entry in sorted(self.calibrations.items())],
            **frame_metrics,
            **worker_metrics,
            'states': {
                'pose': ['lost', 'limited', 'recovering', 'tracking'],
                'mask': ['unavailable'],
                'render': ['suppressed', 'diagnostic_wireframe'],
                'occlusion': 'unsupported',
            },
        }

    def _queue_busy_reset_locked(self, packet):
        session_id = None
        if isinstance(packet, dict):
            try:
                session_id = _identifier(packet.get('sessionId'), 'sessionId')
            except ValueError:
                pass
        self.busy_429_responses += 1
        if session_id is None or self.pending_reset_all:
            return
        if session_id in self.pending_reset_sessions:
            return
        if len(self.pending_reset_sessions) < MAX_SESSIONS:
            self.pending_reset_sessions.add(session_id)
        else:
            self.pending_reset_sessions.clear()
            self.pending_reset_all = True

    def _apply_pending_resets(self):
        with self.admission_lock:
            reset_all = self.pending_reset_all
            sessions = tuple(self.pending_reset_sessions)
            self.pending_reset_all = False
            self.pending_reset_sessions.clear()
        if not reset_all and not sessions:
            return
        with self.lock:
            if reset_all:
                for state in self.sessions.values():
                    state.reset()
            else:
                for session_id in sessions:
                    state = self.sessions.get(session_id)
                    if state is not None:
                        state.reset()

    def _new_result(self, packet, reason, started, calibration=None, *, pose_state='lost',
                    fresh_camera_frame_verified=False):
        def echoed(key, kind):
            if not isinstance(packet, dict):
                return None
            value = packet.get(key)
            if kind == 'text':
                return value if isinstance(value, str) and len(value) <= 128 else None
            if kind == 'integer':
                return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**53-1 else None
            if kind == 'number':
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                try:
                    result = float(value)
                except (OverflowError, ValueError):
                    return None
                return result if math.isfinite(result) else None
            return None

        result = {
            'frameId': echoed('frameId', 'integer'),
            'sessionId': echoed('sessionId', 'text'),
            'objectId': echoed('objectId', 'integer'),
            'referenceId': echoed('referenceId', 'text'),
            'calibrationId': echoed('calibrationId', 'text'),
            'logicalIndex': echoed('logicalIndex', 'integer'),
            'captureTimestampSeconds': echoed('captureTimestampSeconds', 'number'),
            'timestampKind': echoed('timestampKind', 'text'),
            'cameraFrameId': echoed('cameraFrameId', 'integer'),
            'cameraMediaTime': echoed('cameraMediaTime', 'number'),
            'freshCameraFrameVerified': bool(fresh_camera_frame_verified),
            'units': 'metres',
            'cameraConvention': 'CV X right, Y down, Z forward; column vectors',
            'objectBasis': 'Original right-handed source-object axes',
            'poseState': pose_state,
            'maskState': 'unavailable',
            'renderState': 'suppressed',
            'failureReason': reason,
            'cameraFromObject': None,
            'projectionSegments': None,
            'occlusionState': 'unsupported',
            'processingSeconds': max(0., time.perf_counter()-started),
            'neuralUpdatesPerSecond': 0,
        }
        if calibration is not None:
            result['calibrationKind'] = calibration.kind
        return result

    def _session_for(self, session_id, object_id, reference_id):
        current = self.sessions.get(session_id)
        if current is not None:
            if current.object_id != object_id or current.reference_id != reference_id:
                current.reset()
                current.last_access = time.monotonic()
                self.sessions.move_to_end(session_id)
                return current, 'identity_mismatch'
            current.last_access = time.monotonic()
            self.sessions.move_to_end(session_id)
            return current, None
        now = time.monotonic()
        expired = [key for key, state in self.sessions.items()
                   if now-state.last_access > SESSION_IDLE_SECONDS]
        for key in expired:
            self.sessions.pop(key).reset()
        while len(self.sessions) >= self.max_sessions:
            _, oldest = self.sessions.popitem(last=False)
            oldest.reset()
        state = _Session(object_id, reference_id, last_access=now)
        self.sessions[session_id] = state
        return state, None

    @staticmethod
    def _frame_clock(state, packet):
        try:
            frame_id = _json_int(packet.get('frameId'), 'frameId', 0, 2**53-1)
            logical_index = _json_int(packet.get('logicalIndex'), 'logicalIndex', 0, 2**53-1)
        except ValueError:
            state.reset()
            return 'invalid_frame_order', False
        if ((state.last_frame_id is not None and frame_id <= state.last_frame_id) or
                (state.last_logical_index is not None and logical_index <= state.last_logical_index)):
            state.reset()
            return 'out_of_order_frame', False
        # Advance the high-watermarks before validating any remaining payload. A
        # malformed frame cannot be replayed later as if it were new.
        state.last_frame_id = frame_id
        state.last_logical_index = logical_index
        clock_kind = packet.get('timestampKind')
        if clock_kind not in ('physical', 'frame_index_no_fps'):
            state.reset()
            return 'unsupported_clock_kind', False
        if state.clock_kind is not None and clock_kind != state.clock_kind:
            state.reset()
            return 'clock_kind_changed', False
        state.clock_kind = clock_kind
        fresh_camera_frame = False
        if clock_kind == 'physical':
            stamp = packet.get('captureTimestampSeconds')
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
                state.reset()
                return 'invalid_timestamp', False
            try:
                stamp = float(stamp)
            except (OverflowError, ValueError):
                state.reset()
                return 'invalid_timestamp', False
            if not math.isfinite(stamp):
                state.reset()
                return 'invalid_timestamp', False
            if state.last_timestamp is not None and stamp <= state.last_timestamp:
                state.reset()
                return 'out_of_order_timestamp', False
            state.last_timestamp = stamp
        elif packet.get('captureTimestampSeconds') is not None:
            state.reset()
            return 'invalid_clock_metadata', False

        has_camera_frame = 'cameraFrameId' in packet
        has_media_time = 'cameraMediaTime' in packet
        if has_camera_frame or has_media_time:
            if clock_kind != 'physical':
                state.reset()
                return 'camera_frame_metadata_not_applicable', False
        if has_camera_frame:
            try:
                camera_frame_id = _json_int(packet['cameraFrameId'], 'cameraFrameId', 0, 2**53-1)
            except (KeyError, ValueError):
                state.reset()
                return 'invalid_camera_frame_metadata', False
            if state.last_camera_frame_id is not None and camera_frame_id <= state.last_camera_frame_id:
                state.reset()
                return 'out_of_order_camera_frame', False
            state.last_camera_frame_id = camera_frame_id
        if has_media_time:
            try:
                camera_media_time = packet['cameraMediaTime']
                if isinstance(camera_media_time, bool) or not isinstance(camera_media_time, (int, float)):
                    raise ValueError('Invalid cameraMediaTime')
                camera_media_time = float(camera_media_time)
                if not math.isfinite(camera_media_time) or camera_media_time < 0:
                    raise ValueError('Invalid cameraMediaTime')
            except (KeyError, ValueError, OverflowError):
                state.reset()
                return 'invalid_camera_frame_metadata', False
            if state.last_camera_media_time is not None and camera_media_time <= state.last_camera_media_time:
                state.reset()
                return 'out_of_order_camera_frame', False
            state.last_camera_media_time = camera_media_time
        if has_camera_frame != has_media_time:
            state.reset()
            return 'invalid_camera_frame_metadata', False
        if has_camera_frame:
            fresh_camera_frame = True
        return None, fresh_camera_frame

    @staticmethod
    def _decode_rgb(packet, calibration):
        width = _json_int(packet.get('width'), 'frame width', 1, MAX_IMAGE_PIXELS)
        height = _json_int(packet.get('height'), 'frame height', 1, MAX_IMAGE_PIXELS)
        if width*height > MAX_IMAGE_PIXELS:
            raise ValueError('frame_too_large')
        if width != calibration.width or height != calibration.height:
            raise ValueError('calibration_resolution_mismatch')
        encoded = packet.get('rgbJPEGBase64')
        if not isinstance(encoded, str) or not encoded or len(encoded) > ((MAX_JPEG_BYTES+2)//3)*4:
            raise ValueError('invalid_jpeg_payload')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('invalid_jpeg_payload') from exc
        if not raw or len(raw) > MAX_JPEG_BYTES:
            raise ValueError('invalid_jpeg_payload')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'JPEG' or image.size != (width, height):
                    raise ValueError('calibration_resolution_mismatch')
                image.load()
                return np.asarray(image.convert('RGB'), dtype=np.uint8).copy(order='C')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('invalid_jpeg_payload') from exc

    def _pose_segments(self, asset, pose, calibration):
        matrix = np.asarray(pose, dtype=np.float64)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
                not is_proper_rotation(matrix[:3, :3])):
            raise ValueError('invalid_tracking_pose')
        if not len(asset.wire_edges):
            return matrix.tolist(), []
        # Project only the deterministic sparse edge set. The mesh outline is a
        # diagnostic overlay; it is never promoted to an observed object mask.
        edge_vertices = asset.model_positions[asset.wire_edges]
        flat = edge_vertices.reshape(-1, 3)
        pixels, _ = project_points_cv(flat, matrix[:3, :3], matrix[:3, 3], calibration.k)
        segments = np.concatenate((pixels[0::2], pixels[1::2]), axis=1)
        if not np.isfinite(segments).all():
            raise ValueError('invalid_wireframe_projection')
        return matrix.tolist(), segments.tolist()

    def process_frame(self, packet):
        started = time.perf_counter()
        with self.admission_lock:
            self.frame_requests += 1
            admitted = self.frame_admission.acquire(blocking=False)
            if admitted:
                self.accepted_frame_requests += 1
            else:
                # Keep the rejected request's reset queued atomically with its
                # failed admission decision. The next frame admitted after the
                # active inference releases the gate must observe this reset.
                self._queue_busy_reset_locked(packet)
        if not admitted:
            return 429, self._new_result(packet, 'service_busy', started)
        try:
            self._apply_pending_resets()
            return self._process_frame_admitted(packet, started)
        finally:
            self.frame_admission.release()

    def _process_frame_admitted(self, packet, started):
        if not isinstance(packet, dict):
            return 400, {'error': 'invalid_json_object', 'failureReason': 'invalid_frame_packet'}
        try:
            session_id = _identifier(packet.get('sessionId'), 'sessionId')
        except ValueError:
            return 400, self._new_result(packet, 'invalid_identity', started)
        try:
            object_id = _json_int(packet.get('objectId'), 'objectId', 1)
            reference_id = _identifier(packet.get('referenceId'), 'referenceId')
        except ValueError:
            # The session binding is known even when the remaining identity is
            # malformed. Clear its active tracker while retaining sequence
            # high-watermarks so corrected packets must reacquire without
            # making an old frame acceptable again.
            with self.lock:
                state = self.sessions.get(session_id)
                if state is not None:
                    state.reset()
            return 400, self._new_result(packet, 'invalid_identity', started)

        with self.lock:
            state, identity_error = self._session_for(session_id, object_id, reference_id)
            if identity_error:
                return 200, self._new_result(packet, identity_error, started)
            clock_error, fresh_camera_frame = self._frame_clock(state, packet)
            if clock_error:
                result = self._new_result(packet, clock_error, started,
                                          fresh_camera_frame_verified=False)
                return 200, result
            if not REQUIRED_FRAME_FIELDS.issubset(packet) or set(packet)-FRAME_FIELDS:
                state.reset()
                return 400, self._new_result(packet, 'unsupported_frame_fields', started)

            asset = self.assets.get(object_id)
            if asset is None or asset.reference_id != reference_id:
                state.reset()
                return 200, self._new_result(packet, 'asset_identity_mismatch', started)
            calibration_id = packet.get('calibrationId')
            if not isinstance(calibration_id, str) or not calibration_id:
                state.reset()
                return 200, self._new_result(packet, 'calibration_required', started)
            calibration = self.calibrations.get(calibration_id)
            if calibration is None:
                state.reset()
                return 200, self._new_result(packet, 'calibration_unknown', started)
            if calibration.object_id is not None and calibration.object_id != object_id:
                state.reset()
                return 200, self._new_result(packet, 'calibration_object_mismatch', started, calibration,
                                              fresh_camera_frame_verified=False)
            clock_kind = packet['timestampKind']
            camera_fresh_ok = bool(fresh_camera_frame and clock_kind == 'physical' and
                                    calibration.kind == 'physical_measured')
            if ((clock_kind == 'physical' and calibration.kind != 'physical_measured') or
                    (clock_kind == 'frame_index_no_fps' and calibration.kind == 'physical_measured') or
                    (asset.evidence_kind == 'synthetic_control' and
                     not (clock_kind == 'frame_index_no_fps' and calibration.kind == 'synthetic'))):
                state.reset()
                return 200, self._new_result(packet, 'calibration_clock_mismatch', started, calibration,
                                              fresh_camera_frame_verified=False)
            if state.calibration_id is None:
                state.calibration_id = calibration_id
            elif state.calibration_id != calibration_id:
                state.reset()
                return 200, self._new_result(packet, 'calibration_identity_mismatch', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)

            try:
                rgb = self._decode_rgb(packet, calibration)
            except ValueError as exc:
                state.reset()
                return 200, self._new_result(packet, str(exc), started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)

            if state.tracker is None:
                try:
                    state.tracker = self.tracker_factory(
                        asset.reference_points, asset.reference_descriptors, asset.model_positions,
                        session_id=session_id, object_id=object_id, reference_id=reference_id,
                    )
                except Exception:
                    state.reset()
                    return 200, self._new_result(packet, 'tracker_initialization_failed', started, calibration,
                                                  fresh_camera_frame_verified=camera_fresh_ok)
            frame = FrameInputV1(
                session_id, object_id, reference_id,
                packet['frameId'], packet['logicalIndex'], packet['captureTimestampSeconds'],
                packet['timestampKind'], rgb, calibration.k, (calibration.width, calibration.height),
            )
            try:
                output = state.tracker.update(frame)
                manifest = output.manifest() if hasattr(output, 'manifest') else output
            except Exception:
                state.reset()
                return 200, self._new_result(packet, 'tracking_failed', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            if not isinstance(manifest, dict):
                state.reset()
                return 200, self._new_result(packet, 'invalid_tracker_output', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            identity = (manifest.get('sessionId'), manifest.get('objectId'), manifest.get('referenceId'),
                        manifest.get('frameId'), manifest.get('logicalIndex'))
            expected_identity = (session_id, object_id, reference_id, packet['frameId'], packet['logicalIndex'])
            if identity != expected_identity:
                state.reset()
                return 200, self._new_result(packet, 'tracker_identity_mismatch', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            pose_state = manifest.get('state')
            if not isinstance(pose_state, str) or pose_state not in TRACKING_STATES:
                state.reset()
                return 200, self._new_result(packet, 'invalid_tracker_state', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            result = self._new_result(packet, manifest.get('failureReason'), started, calibration,
                                      pose_state=pose_state,
                                      fresh_camera_frame_verified=camera_fresh_ok)
            if pose_state != 'tracking':
                return 200, result
            pose = manifest.get('cameraFromObject')
            if pose is None:
                state.reset()
                return 200, self._new_result(packet, 'missing_tracking_pose', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            try:
                pose_matrix, segments = self._pose_segments(asset, pose, calibration)
            except (TypeError, ValueError, np.linalg.LinAlgError):
                state.reset()
                return 200, self._new_result(packet, 'invalid_tracking_pose', started, calibration,
                                              fresh_camera_frame_verified=camera_fresh_ok)
            result.update(
                poseState='tracking',
                renderState='diagnostic_wireframe' if segments else 'suppressed',
                cameraFromObject=pose_matrix,
                projectionSegments=segments if segments else None,
                failureReason=None,
            )
            result['processingSeconds'] = max(0., time.perf_counter()-started)
            return 200, result

    def static_file(self, request_path):
        """Resolve only the configured browser client or files under --demo."""
        split = urlsplit(request_path)
        if split.query or split.fragment:
            return None
        if split.path in ('/', '/index.html'):
            path = self.client_path
            root = path.parent
        elif split.path.startswith('/demo/') and self.demo_root is not None:
            relative = unquote(split.path[len('/demo/'):]) or 'index.html'
            if '\\' in relative:
                return None
            parts = PurePosixPath(relative).parts
            if not parts or any(part in ('..', '.') or part.startswith('.') for part in parts):
                return None
            path = (self.demo_root.joinpath(*parts)).resolve()
            root = self.demo_root
            if not _inside(path, root):
                return None
            if path.suffix.lower() == '.json' and parts != ('gallery.json',):
                return None
        else:
            return None
        try:
            if not path.is_file() or path.stat().st_size > MAX_STATIC_BYTES:
                return None
            if path.suffix.lower() not in PUBLIC_DEMO_EXTENSIONS and path.name != 'gallery.json':
                return None
            content = path.read_bytes()
        except OSError:
            return None
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if content_type == 'text/html':
            content_type += '; charset=utf-8'
        return content_type, content


def validate_bind_security(host, cert_path=None, key_path=None):
    """Permit loopback HTTP or private/link-local TLS only; reject public binds."""
    if bool(cert_path) != bool(key_path):
        raise ValueError('TLS requires both --cert and --key.')
    if host.lower() == 'localhost':
        address = ipaddress.ip_address('127.0.0.1')
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError('Bind to localhost or an explicit IPv4/IPv6 address.') from exc
    if address.version != 4:
        raise ValueError('The prototype service currently supports IPv4 listeners only.')
    if address.is_unspecified or address.is_multicast or address.is_global:
        raise ValueError('Only loopback or an explicit private/link-local address is allowed.')
    if not address.is_loopback and not (address.is_private or address.is_link_local):
        raise ValueError('Non-loopback binds must be private or link-local.')
    if not address.is_loopback and not cert_path:
        raise ValueError('A non-loopback listener requires --cert and --key TLS files.')
    if cert_path and (not Path(cert_path).is_file() or not Path(key_path).is_file()):
        raise ValueError('TLS certificate and key files must exist.')
    return address


class _PrototypeHTTPServer(HTTPServer):
    allow_reuse_address = True
    request_queue_size = 4

    def __init__(self, address, service):
        self.service = service
        self.tls_context = None
        self._worker_slots = threading.BoundedSemaphore(MAX_HTTP_WORKERS)
        self._worker_lock = threading.Lock()
        self._active_workers = 0
        self._connection_admission_drops = 0
        super().__init__(address, _RequestHandler)

    def process_request(self, request, client_address):
        if not self._worker_slots.acquire(blocking=False):
            with self._worker_lock:
                self._connection_admission_drops += 1
            self.shutdown_request(request)
            return
        with self._worker_lock:
            self._active_workers += 1
        try:
            worker = threading.Thread(
                target=self._run_admitted_request,
                args=(request, client_address),
                daemon=True,
            )
            worker.start()
        except Exception:
            self._finish_worker(request)
            self.handle_error(request, client_address)

    def _run_admitted_request(self, request, client_address):
        connection = request
        try:
            if self.tls_context is not None:
                connection.settimeout(WORKER_SOCKET_TIMEOUT_SECONDS)
                connection = self.tls_context.wrap_socket(connection, server_side=True)
            self.finish_request(connection, client_address)
        except OSError:
            # Includes slow/invalid TLS handshakes and interrupted clients.
            pass
        except Exception:
            self.handle_error(connection, client_address)
        finally:
            self._finish_worker(connection)

    def _finish_worker(self, connection):
        try:
            try:
                self.shutdown_request(connection)
            except OSError:
                pass
        finally:
            with self._worker_lock:
                self._active_workers -= 1
            self._worker_slots.release()

    def worker_metrics(self):
        with self._worker_lock:
            return {
                'connectionAdmissionDrops': self._connection_admission_drops,
                'activeWorkers': self._active_workers,
                'maxWorkers': MAX_HTTP_WORKERS,
            }


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'VisualizeItPrototype/1'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(10.)

    def log_message(self, format_string, *args):
        # Avoid logging camera/session content or creating an unbounded log file.
        return

    def _allowed_host(self):
        raw = self.headers.get('Host', '')
        host = raw.rsplit(':', 1)[0] if ':' in raw and not raw.startswith('[') else raw
        if host.startswith('[') and ']' in host:
            host = host[1:host.index(']')]
        if host.lower() == 'localhost':
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return bool(not address.is_unspecified and not address.is_global and
                    (address.is_loopback or address.is_private or address.is_link_local))

    def _headers(self, content_type, length):
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
        self.send_header('Content-Security-Policy', "default-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; form-action 'none'; base-uri 'none'")
        self.send_header('Connection', 'close')

    def _send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self._headers('application/json; charset=utf-8', len(body))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_bytes(self, content_type, body):
        self.send_response(200)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        if not self._allowed_host():
            return self._send_json(403, {'error': 'host_not_allowed'})
        path = urlsplit(self.path).path
        if path == '/healthz':
            return self._send_json(200, {'status': 'ok', 'assetsLoaded': len(self.server.service.assets),
                                         'calibrationsLoaded': len(self.server.service.calibrations)})
        if path == '/health':
            return self._send_json(200, {'status': 'ok', 'assetsLoaded': len(self.server.service.assets),
                                         'calibrationsLoaded': len(self.server.service.calibrations)})
        if path == '/v1/status':
            return self._send_json(200, self.server.service.status(self.server.worker_metrics()))
        static = self.server.service.static_file(self.path)
        if static is not None:
            return self._send_bytes(*static)
        return self._send_json(404, {'error': 'not_found'})

    def do_POST(self):
        if not self._allowed_host():
            return self._send_json(403, {'error': 'host_not_allowed'})
        if urlsplit(self.path).path != '/v1/frames':
            return self._send_json(404, {'error': 'not_found'})
        if self.headers.get('Transfer-Encoding'):
            return self._send_json(400, {'error': 'transfer_encoding_not_supported'})
        media_type = self.headers.get_content_type()
        if media_type != 'application/json':
            return self._send_json(415, {'error': 'application_json_required'})
        raw_length = self.headers.get('Content-Length')
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            return self._send_json(411, {'error': 'content_length_required'})
        if length <= 0:
            return self._send_json(400, {'error': 'empty_request'})
        if length > MAX_REQUEST_BYTES:
            return self._send_json(413, {'error': 'request_too_large'})
        try:
            body = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            return
        if len(body) != length:
            return self._send_json(400, {'error': 'incomplete_request'})
        try:
            packet = json.loads(body.decode('utf-8'), parse_constant=_reject_json_constant,
                                parse_float=_finite_json_float)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return self._send_json(400, {'error': 'invalid_json'})
        status, result = self.server.service.process_frame(packet)
        return self._send_json(status, result)


def create_server(host, port, service, cert_path=None, key_path=None):
    address = validate_bind_security(host, cert_path, key_path)
    port = _json_int(port, 'port', 0, 65535)
    server = _PrototypeHTTPServer((str(address), port), service)
    if cert_path:
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
            server.tls_context = context
        except Exception:
            server.server_close()
            raise
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='fixed root containing object-XXXXXX/asset.json and its SHA-pinned NPZ')
    parser.add_argument('--calibrations', type=Path,
                        help='optional version-1 calibration registry; absent means all poses are suppressed')
    parser.add_argument('--demo', type=Path,
                        help='optional explicit static report directory served under /demo/')
    parser.add_argument('--client', type=Path, default=ROOT/'prototype'/'client.html')
    parser.add_argument('--host', default='127.0.0.1', help='localhost, loopback, or explicit private IPv4')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--cert', type=Path, help='TLS certificate PEM (required for private LAN bind)')
    parser.add_argument('--key', type=Path, help='TLS private key PEM (required with --cert)')
    args = parser.parse_args(argv)
    service = PrototypeService(args.assets, args.calibrations, args.demo, args.client)
    server = create_server(args.host, args.port, service, args.cert, args.key)
    scheme = 'https' if args.cert else 'http'
    print(f'Prototype service listening at {scheme}://{args.host}:{server.server_port}/', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
