"""Build a private, static visual report from sealed prototype benchmark results.

This is a post-seal visualization step. It never invokes the tracker and never
passes evaluator annotations into inference.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from contextlib import closing, nullcontext
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from .prototype_output_budget import CompactOutputBudget, CompactBudgetExceeded

MAX_GALLERY_BYTES = 30 * 1024 * 1024
GALLERY_METADATA_RESERVE_BYTES = 2 * 1024 * 1024
PANEL_WIDTH = 320
PANEL_MAX_HEIGHT = 220
FIXED_SAMPLE_INDICES = (0, 36, 73, 109)
TRIAL_NAMES = ('natural', 'interrupted')


class GalleryError(RuntimeError):
    """The sealed inputs cannot support a trustworthy gallery."""


class GalleryBudgetExceeded(GalleryError):
    """Writing another gallery asset would exceed the private 30 MiB cap."""


def _json_bytes(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + '\n'
    # Keep JSON safe if a consumer later embeds it in an HTML script element.
    text = (text.replace('&', '\\u0026').replace('<', '\\u003c')
            .replace('>', '\\u003e').replace('\u2028', '\\u2028')
            .replace('\u2029', '\\u2029'))
    return text.encode('utf-8')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _inside(root, path, *, allow_root=False):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if (path == root and not allow_root) or root not in path.parents and path != root:
        raise GalleryError(f'Path escapes its declared root: {path.name}')
    return path


def _verified_resource(root, spec, role='rgb'):
    from .prototype_resource_store import ResourceStore
    if isinstance(root, ResourceStore):
        return root.open_binary(spec, role)
    if not isinstance(spec, dict) or not spec.get('available', True):
        raise GalleryError('Dataset resource is unavailable.')
    relative = spec.get('path')
    expected = spec.get('sha256')
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise GalleryError('Dataset resource has no pinned path/hash.')
    path = _inside(root, Path(root).joinpath(*PurePosixPath(relative).parts))
    if not path.is_file() or _sha256(path) != expected:
        raise GalleryError(f'Dataset resource hash mismatch: {relative}')
    return path


def sample_plan(frames):
    """Choose stable timeline anchors plus the first failure and recovery."""
    if len(frames) != 110:
        raise ValueError('A gallery trial must retain its complete 110-frame timeline.')
    tags = {}
    for index in FIXED_SAMPLE_INDICES:
        tags.setdefault(index, []).append(f'fixed index {index}')
    failure = next((i for i, row in enumerate(frames) if row.get('state') != 'tracking'), None)
    if failure is not None:
        tags.setdefault(failure, []).append('first nontracking frame')
    recovery = next((i for i in range(1, len(frames))
                     if frames[i - 1].get('state') != 'tracking'
                     and frames[i].get('state') == 'tracking'), None)
    if recovery is not None:
        tags.setdefault(recovery, []).append('first tracking frame after loss')
    return [{'index': i, 'frameId': frames[i].get('frameId'), 'tags': labels}
            for i, labels in sorted(tags.items())]


def timeline_rows(frames, metrics):
    """Keep one public row for every scheduled frame, including failures."""
    if len(frames) != 110:
        raise ValueError('A gallery trial must retain all 110 frames.')
    per_frame = {row.get('frameId'): row for row in metrics.get('perFrameAttachment', [])
                 if isinstance(row, dict)}
    rows = []
    for index, frame in enumerate(frames):
        score = per_frame.get(frame.get('frameId'), {})
        row = {
            'index': index,
            'logicalIndex': frame.get('logicalIndex'),
            'frameId': frame.get('frameId'),
            'state': frame.get('state', 'unavailable'),
            'failureReason': frame.get('failureReason'),
            'forcedInterruption': bool(frame.get('forcedInterruption', False)),
            'pairedPoints': score.get('pairedPoints'),
            'medianErrorPixels': _finite_number(score.get('medianErrorPixels')),
            'p95ErrorPixels': _finite_number(score.get('p95ErrorPixels')),
            'unavailableReason': score.get('unavailableReason'),
        }
        rows.append(row)
    return rows


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    return value if math.isfinite(value) else None


def estimated_pose_for_frame(frame):
    """Return only a current tracking pose; never reuse a pose after loss."""
    if frame.get('state') != 'tracking':
        return None
    return frame.get('cvCameraFromSourceObject')


def physical_occlusion_mask(triangle_ids, rendered_depth, measured_depth):
    """Mask rendered pixels behind measured scene depth; never use eval masks."""
    import numpy as np
    projected = np.asarray(triangle_ids) >= 0
    if measured_depth is None:
        return projected, 'unoccluded_depth_unavailable'
    measured_depth = np.asarray(measured_depth)
    rendered_depth = np.asarray(rendered_depth)
    if projected.shape != measured_depth.shape or projected.shape != rendered_depth.shape:
        raise ValueError('Render, triangle map, and measured depth dimensions differ.')
    measured = np.isfinite(measured_depth) & (measured_depth > 0)
    tolerance = np.maximum(.005, np.where(measured, measured_depth * .01, 0))
    projected &= measured & (rendered_depth <= measured_depth + tolerance)
    return projected, 'measured_depth_occlusion'


def _check_budget(current_bytes, next_bytes, limit=MAX_GALLERY_BYTES):
    if current_bytes < 0 or next_bytes < 0 or current_bytes + next_bytes > limit:
        raise GalleryBudgetExceeded('Private generated gallery would exceed 30 MiB.')


def _safe_metrics(value):
    """Copy scalar report data without leaking pose/annotation payloads."""
    if isinstance(value, dict):
        return {str(key): _safe_metrics(child) for key, child in value.items()
                if key not in ('evaluationPose', 'evaluationMask', 'pose', 'cameraFromObject',
                               'cvCameraFromSourceObject', 'cam_R_m2c', 'cam_t_m2c')}
    if isinstance(value, list):
        return [_safe_metrics(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _binding(obj):
    return {
        'objectId': obj['objectId'], 'name': obj['name'], 'model': obj['model'],
        'frames': [{key: row[key] for key in ('frameId', 'calibration', 'rgb')}
                   for row in obj['frames']],
    }


def _evidence_contract(report, seal, object_count):
    kind = report.get('evidenceKind', 'dataset_reference')
    if seal.get('evidenceKind', kind) != kind:
        raise GalleryError('Report and inference seal evidence kinds differ.')
    real_objects = report.get('realUnseenObjects')
    if seal.get('realUnseenObjects', real_objects) != real_objects:
        raise GalleryError('Report and inference seal real-object counts differ.')
    if kind == 'synthetic_control':
        if object_count != 1 or real_objects != 0:
            raise GalleryError('Synthetic control must contain one smoke object and zero real unseen objects.')
        if report.get('tenObjectsEvaluated') is not False:
            raise GalleryError('Synthetic control must explicitly mark ten-object evaluation false.')
        if report.get('benchmarkGatePassed') is True or report.get('benchmarkGateEligible') is True:
            raise GalleryError('Synthetic control cannot pass or qualify the real benchmark gate.')
        return {
            'evidenceKind': kind,
            'datasetName': str(report.get('datasetName', report.get('dataset', 'Synthetic fixture'))),
            'objectIdLabel': str(report.get('objectIdLabel', 'Synthetic control')),
            'title': 'Synthetic runtime smoke gallery',
            'provenance': 'Synthetic control only: zero real unseen objects; no real-object accuracy claim.',
            'scopeNote': 'One synthetic object for runtime smoke only. It is not a new physical identity or a benchmark result.',
            'renderNote': 'The panels visualize synthetic fixture RGB, original fixture geometry, and its supplied synthetic pose control. They are not real-object evidence.',
            'realUnseenObjects': 0,
            'benchmarkMode': 'synthetic_control',
            'plannedObjects': 1,
            'benchmarkGateEligible': False,
            'tenObjectsEvaluated': False,
        }
    if kind not in ('dataset_reference', 'ycbv_dataset_reference'):
        raise GalleryError(f'Unsupported gallery evidence kind: {kind}')
    mode = report.get('benchmarkMode', seal.get('benchmarkMode', 'full'))
    if seal.get('benchmarkMode', mode) != mode:
        raise GalleryError('Report and inference seal benchmark modes differ.')
    if mode == 'pilot':
        if not (1 <= object_count <= 9) or real_objects != object_count:
            raise GalleryError('Dataset-reference pilot count must equal its 1–9 real objects.')
        if report.get('tenObjectsEvaluated') is not False:
            raise GalleryError('Dataset-reference pilot must explicitly mark ten-object evaluation false.')
        if seal.get('tenObjectsEvaluated', False) is not False:
            raise GalleryError('Inference seal must mark ten-object evaluation false for a pilot.')
        if report.get('plannedObjects', object_count) != object_count:
            raise GalleryError('Dataset-reference pilot planned-object count differs from its contents.')
        if seal.get('plannedObjects', object_count) != object_count:
            raise GalleryError('Inference seal planned-object count differs from its contents.')
        if report.get('benchmarkGatePassed') is True or report.get('benchmarkGateEligible') is True:
            raise GalleryError('A dataset-reference pilot cannot pass or qualify the ten-object benchmark gate.')
        dataset_name = str(report.get('datasetName', report.get('dataset', 'BOP YCB-Video')))
        return {
            'evidenceKind': kind,
            'datasetName': dataset_name,
            'objectIdLabel': str(report.get('objectIdLabel', 'YCBV object ID')),
            'title': str(report.get('galleryTitle', 'Recorded dataset-reference pilot gallery')),
            'provenance': str(report.get('provenance', report.get('evidenceLabel',
                f'{dataset_name} dataset-reference pilot; {object_count} real unseen objects; not the ten-object benchmark.'))),
            'scopeNote': f'Pilot dataset-reference scores for {object_count} eligible real objects. The ten-object benchmark gate is not evaluated; independent human accuracy and live camera performance are unverified.',
            'renderNote': 'Checkerboard geometry is rendered from the original model. Measured depth is used only for physical occlusion; evaluation masks are not used as inferred masks.',
            'realUnseenObjects': object_count,
            'benchmarkMode': 'pilot',
            'plannedObjects': object_count,
            'benchmarkGateEligible': False,
            'tenObjectsEvaluated': False,
        }
    if mode == 'followup' and (report.get('datasetName') != 'HB' or seal.get('datasetName') != 'HB'
                               or report.get('objectIdLabel') != 'HB ID'):
        raise GalleryError('Followup mode requires validated HB provenance')
    if mode not in ('full', 'followup'):
        raise GalleryError(f'Unsupported dataset-reference benchmark mode: {mode}')
    if object_count != 10 or real_objects not in (None, 10):
        raise GalleryError('Dataset-reference gallery requires ten eligible real objects.')
    if report.get('tenObjectsEvaluated') is False:
        raise GalleryError('Dataset-reference report does not contain ten evaluated objects.')
    dataset_name = str(report.get('datasetName', report.get('dataset', 'BOP YCB-Video')))
    return {
        'evidenceKind': kind,
        'datasetName': dataset_name,
        'objectIdLabel': str(report.get('objectIdLabel', 'YCBV object ID')),
        'title': str(report.get('galleryTitle', 'Recorded prototype benchmark gallery')),
        'provenance': str(report.get('provenance', report.get('evidenceLabel',
            f'{dataset_name} dataset-reference labels; no independent human accuracy claim.'))),
        'scopeNote': 'Dataset-reference scores only; independent human accuracy and live camera performance are unverified.',
        'renderNote': 'Checkerboard geometry is rendered from the original model. Measured depth is used only for physical occlusion; evaluation masks are not used as inferred masks.',
        'realUnseenObjects': 10 if real_objects is None else real_objects,
        'benchmarkMode': mode,
        'plannedObjects': 10,
        'benchmarkGateEligible': bool(report.get('benchmarkGateEligible', True)),
        'tenObjectsEvaluated': True,
    }


def _validate_output(results, dataset, output):
    results = Path(results).resolve()
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    if output == results or results not in output.parents:
        raise GalleryError('Gallery output must be a new child of the private results directory.')
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise GalleryError('Gallery output may not overlap the source dataset cache.')
    if output.exists():
        raise GalleryError('Gallery output already exists; preserving existing generated artifacts.')
    return results, dataset, output


def _load_sealed_inputs(dataset, results):
    """Authenticate inference and every trace before opening evaluator labels."""
    freeze_path = results / 'freeze.json'
    seal_path = results / 'inference-seal.json'
    report_path = results / 'report.json'
    inference_path = dataset / 'inference.json'
    for path in (freeze_path, seal_path, report_path, inference_path,
                 dataset / 'evaluator.json'):
        if not path.is_file():
            raise GalleryError(f'Missing sealed benchmark input: {path.name}')
    freeze = _read_json(freeze_path)
    seal = _read_json(seal_path)
    report = _read_json(report_path)
    if _sha256(freeze_path) != seal.get('freezeSHA256'):
        raise GalleryError('Inference freeze hash does not match the output seal.')
    if seal.get('resourceStorage'):
        if freeze.get('resourceStorage') != seal['resourceStorage']:
            raise GalleryError('Compact storage differs from inference freeze.')
        if _sha256(results / 'resource-access.json') != seal.get('resourceAccessSHA256'):
            raise GalleryError('Inference access log changed after seal.')
    if _sha256(inference_path) != freeze.get('inferenceManifestSHA256'):
        raise GalleryError('Inference dataset manifest changed after the run.')
    if (seal.get('evaluatorManifestSHA256') != freeze.get('evaluatorManifestSHA256')
            or report.get('evaluatorManifestSHA256') != seal.get('evaluatorManifestSHA256')):
        raise GalleryError('Evaluation labels are not bound to the inference seal.')
    if report.get('inferenceSealSHA256') != _sha256(seal_path):
        raise GalleryError('Evaluation report does not bind to the inference seal.')

    # Verify every trace before reading the evaluator manifest or any labels.
    trial_hashes = seal.get('trialHashes')
    if not isinstance(trial_hashes, dict) or not trial_hashes:
        raise GalleryError('Inference seal contains no pinned trial traces.')
    verified_trials = {}
    for relative, expected in trial_hashes.items():
        path = _inside(results, results.joinpath(*PurePosixPath(relative).parts))
        if not path.is_file() or _sha256(path) != expected:
            raise GalleryError(f'Sealed trial trace hash mismatch: {relative}')
        verified_trials[relative] = path

    inference = _read_json(inference_path)
    if inference.get('benchmarkId') != seal.get('benchmarkId'):
        raise GalleryError('Inference manifest benchmark ID does not match the seal.')
    if inference.get('selectionDigest') != seal.get('selectionDigest'):
        raise GalleryError('Inference manifest selection digest does not match the seal.')
    if inference.get('evaluatorManifestSHA256') != seal.get('evaluatorManifestSHA256'):
        raise GalleryError('Inference manifest does not pin the evaluator manifest.')
    evaluator_path = dataset / 'evaluator.json'
    if _sha256(evaluator_path) != seal.get('evaluatorManifestSHA256'):
        raise GalleryError('Evaluator manifest changed after the inference seal.')
    evaluator = _read_json(evaluator_path)
    if (evaluator.get('benchmarkId') != seal.get('benchmarkId')
            or evaluator.get('selectionDigest') != seal.get('selectionDigest')):
        raise GalleryError('Evaluator manifest belongs to a different frozen selection.')

    inference_objects = {obj['objectId']: obj for obj in inference.get('objects', [])}
    evaluator_objects = {obj['objectId']: obj for obj in evaluator.get('objects', [])}
    report_objects = {obj['objectId']: obj for obj in report.get('objects', [])}
    seal_objects = {obj['objectId']: obj for obj in seal.get('objects', [])}
    if len(inference_objects) != len(inference.get('objects', [])):
        raise GalleryError('Inference manifest contains duplicate object IDs.')
    evidence = _evidence_contract(report, seal, len(inference_objects))
    for manifest_name, manifest in (('inference', inference), ('evaluator', evaluator)):
        for key in ('evidenceKind', 'benchmarkMode', 'realUnseenObjects',
                    'plannedObjects', 'tenObjectsEvaluated'):
            if key == 'tenObjectsEvaluated' and evidence['benchmarkMode'] == 'followup':
                continue  # Pre-run HB manifests explicitly retain False; the sealed report owns completion.
            if key in manifest and manifest[key] != evidence[key]:
                raise GalleryError(f'{manifest_name.title()} manifest {key} differs from the sealed report contract.')
    if not (set(inference_objects) == set(evaluator_objects) == set(report_objects) == set(seal_objects)):
        raise GalleryError('Inference, evaluation, report, and seal object sets differ.')
    if report.get('attemptedObjects') != len(inference_objects):
        raise GalleryError('Evaluation report object count differs from the sealed run.')

    bindings = {row['objectId']: row for row in seal.get('geometryBindings', [])}
    if set(bindings) != set(inference_objects):
        raise GalleryError('Inference seal geometry bindings are incomplete.')
    for object_id, inf_obj in inference_objects.items():
        if len(inf_obj.get('frames', [])) != 110 or len(inf_obj.get('onboarding', [])) != 10:
            raise GalleryError(f'Object {object_id} does not have the frozen 10/110 split.')
        eval_obj = evaluator_objects[object_id]
        if len(eval_obj.get('frames', [])) != 110:
            raise GalleryError(f'Evaluator object {object_id} has an incomplete timeline.')
        if (_binding(inf_obj) != bindings[object_id]
                or _binding(eval_obj) != bindings[object_id]
                or inf_obj.get('referenceId') != eval_obj.get('referenceId')):
            raise GalleryError(f'Object {object_id} evaluator geometry differs from inference.')
    return {
        'freeze': freeze, 'seal': seal, 'report': report,
        'inferenceObjects': inference_objects, 'evaluatorObjects': evaluator_objects,
        'reportObjects': report_objects, 'sealObjects': seal_objects,
        'evidence': evidence,
        'trialPaths': verified_trials,
    }


def _load_rgb(dataset, spec):
    import numpy as np
    from PIL import Image
    path = _verified_resource(dataset, spec)
    with (closing(path) if hasattr(path, 'close') else nullcontext()), Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def _load_depth(dataset, record):
    import numpy as np
    from PIL import Image
    path = _verified_resource(dataset, record['depth'], 'evaluation_depth')
    with (closing(path) if hasattr(path, 'close') else nullcontext()), Image.open(path) as image:
        depth = np.asarray(image).astype(np.float64)
    scale = float(record['calibration']['depth_scale']) * .001
    if not math.isfinite(scale) or scale <= 0:
        raise GalleryError('Invalid measured-depth scale.')
    return depth * scale


def _camera_from_matrix(matrix):
    import numpy as np
    value = np.asarray(matrix, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise GalleryError('Estimated frame pose is not a finite proper metre SE(3) pose.')
    return value[:3, :3], value[:3, 3]


def _pose_from_annotation(record):
    import numpy as np
    pose = record['evaluationPose']
    rotation = np.asarray(pose['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(pose['cam_t_m2c'], dtype=np.float64).reshape(3) * .001
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise GalleryError('Dataset pose has nonfinite values.')
    u, _, vt = np.linalg.svd(rotation)
    repaired = u @ vt
    if np.linalg.det(repaired) < 0 or np.linalg.norm(rotation - repaired) > .001:
        raise GalleryError('Dataset pose is not near a proper rotation.')
    return repaired, translation


def _resize_panel(image):
    from PIL import Image
    width, height = image.size
    scale = min(PANEL_WIDTH / width, PANEL_MAX_HEIGHT / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _overlay(rgb, mesh, transform, rotation, translation, calibration, depth):
    """Render checkerboard geometry over RGB, using measured depth for occlusion."""
    import numpy as np
    from PIL import Image
    from .prototype_benchmark import cv_render_camera
    from .renderer import render

    height, width = rgb.shape[:2]
    camera = cv_render_camera(rotation, translation, calibration, transform, width, height)
    raster = render(mesh, camera, cull=True, shade=True)
    try:
        projected, depth_state = physical_occlusion_mask(raster.triangle,
                                                          raster.depth, depth)
    except ValueError as exc:
        raise GalleryError(str(exc)) from exc
    result = rgb.copy()
    if projected.any():
        rendered = raster.color.astype(np.float32)
        result[projected] = np.clip(.50 * rgb[projected] + .50 * rendered[projected],
                                    0, 255).astype(np.uint8)
    return Image.fromarray(result, mode='RGB'), depth_state


def _contact_sheet(rgb, estimated, reference,
                   reference_label='Dataset-pose reference control',
                   input_label='Captured RGB',
                   estimated_label='Estimated design render'):
    from PIL import Image, ImageDraw
    source = Image.fromarray(rgb, mode='RGB')
    panels = [source if item is None else item for item in (None, estimated, reference)]
    labels = (input_label, estimated_label, reference_label)
    resized = [_resize_panel(panel) for panel in panels]
    panel_height = max(panel.height for panel in resized)
    label_height = 24
    canvas = Image.new('RGB', (PANEL_WIDTH * 3, panel_height + label_height), (20, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(resized, labels)):
        x = index * PANEL_WIDTH
        draw.text((x + 5, 5), label, fill=(235, 245, 240))
        canvas.paste(panel, (x, label_height))
    buffer = io.BytesIO()
    canvas.save(buffer, format='JPEG', quality=74, optimize=True)
    return buffer.getvalue()


def _result_index(results, frames):
    by_id = {row.get('frameId'): row for row in frames}
    if len(by_id) != len(frames):
        raise GalleryError('Trial trace contains duplicate frame IDs.')
    if [row.get('frameId') for row in frames] != [row.get('frameId') for row in results]:
        raise GalleryError('Inference trace does not match the frozen 110-frame sequence.')
    return by_id


def _used_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob('*') if path.is_file())


def _write_limited(root, relative, data, limit=MAX_GALLERY_BYTES, guard=None):
    root = Path(root).resolve()
    target = _inside(root, root.joinpath(*PurePosixPath(relative).parts))
    if target.exists():
        raise GalleryError(f'Gallery asset already exists: {target.name}')
    _check_budget(_used_bytes(root), len(data), limit)
    if guard:
        guard.write_bytes(target, data)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    if temporary.exists():
        raise GalleryError(f'Retained gallery partial exists: {temporary.name}')
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def _html_document():
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prototype benchmark gallery</title>
<style>
:root{color-scheme:dark;background:#111a1a;color:#e8f2ee;font:15px system-ui,sans-serif}body{max-width:1200px;margin:auto;padding:18px}
h1{margin:4px 0 8px}.note{color:#b9ccc6;line-height:1.5}.object{border-top:1px solid #456158;padding:18px 0}.trial{margin:14px 0;padding:12px;background:#1a2927;border-radius:10px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}.metric{background:#243734;padding:8px;border-radius:6px}.metric b{display:block;color:#9cdbc0;font-size:12px}
.timeline{display:grid;grid-template-columns:repeat(22,minmax(8px,1fr));gap:3px;margin:12px 0}.tick{height:13px;border-radius:2px;background:#ba8058}.tick.tracking{background:#61bd90}.tick.recovering{background:#dfbd5b}.tick.unavailable{background:#745f89}
.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.sample{background:#101817;padding:8px;border-radius:8px;min-width:0}.sample img{display:block;width:100%;height:auto}.sample p{margin:7px 0;color:#c6d9d2}.unavailable{color:#f1bd9b}
@media(max-width:650px){body{padding:12px}.timeline{grid-template-columns:repeat(11,minmax(9px,1fr))}}
</style></head><body>
<h1 id="pageTitle">Recorded benchmark gallery</h1>
<p id="datasetLabel" class="note">Dataset and evidence labels load from the sealed report.</p>
<p id="scopeNote" class="note">Report provenance and evidence scope load from the sealed report. A failed tracking frame keeps its actual tracker input and suppresses the estimated-pose render; forced-interruption input is blank, with the recording reserved for the labeled dataset-pose reference control. Gallery generation runs only after inference is sealed. no live camera FPS claim is made.</p>
<p id="status" role="status">Loading sealed report…</p><main id="gallery"></main>
<script>
'use strict';
const status=document.getElementById('status'), gallery=document.getElementById('gallery');
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=String(text);if(cls)n.className=cls;return n;};
function metricLabel(key,evidenceKind,benchmarkMode){if(evidenceKind==='synthetic_control'){return ({attachmentGatePassed:'Synthetic smoke check',recoveryGatePassed:'Synthetic recovery smoke check',allFrameAvailability:'Synthetic frame availability',eligibleVisibleFrameAvailability:'Synthetic visible-frame availability',medianErrorPixels:'Synthetic median error (px)',p95ErrorPixels:'Synthetic P95 error (px)'}[key]||key);}const labels={allFrameAvailability:'All-frame availability',eligibleVisibleFrameAvailability:'Visible-frame availability',trackingFrames:'Tracking frames',scheduledFrames:'Scheduled frames',medianErrorPixels:'Median paired error (px)',p95ErrorPixels:'P95 paired error (px)',pairedPoints:'Paired landmarks',pairedFrames:'Frames with paired landmarks',attachmentGatePassed:benchmarkMode==='pilot'?'Pilot gate':'Dataset-reference gate',recoveryGatePassed:'Recovery gate',recoveryObservableFrames:'Recovery delay (visible frames)',nativeMedianErrorPixels:'Native median error (px)',nativeP95ErrorPixels:'Native P95 error (px)',medianProcessingSeconds:'Median offline processing (s)',p95ProcessingSeconds:'P95 offline processing (s)'};return labels[key]||key;}
function format(value,key){if(value===null||value===undefined)return 'unavailable';if(typeof value==='boolean')return /gate|check/i.test(String(key||''))?(value?'passed':'not passed'):String(value);if(typeof value==='number')return Number.isInteger(value)?String(value):value.toFixed(3);return String(value);}
function render(data){document.title=data.title;document.getElementById('pageTitle').textContent=data.title;document.getElementById('datasetLabel').textContent=`Dataset: ${data.datasetName} · Evidence: ${data.evidenceKind} · Mode: ${data.benchmarkMode} · Real unseen objects: ${data.realUnseenObjects} · Benchmark eligible: ${data.benchmarkGateEligible}`;document.getElementById('scopeNote').textContent=[data.provenance,data.scopeNote,data.renderNote,'no live camera FPS claim is made.'].filter(Boolean).join(' ');status.textContent=`${data.objects.length} attempted objects · ${data.sampleCount} selected visual samples · ${data.generatedBytes} total generated bytes`;
for(const object of data.objects){const section=el('section',undefined,'object');section.append(el('h2',`${object.name} · ${data.objectIdLabel} ${object.objectId}`));section.append(el('p',`Reference ID ${object.referenceId} · ${object.identityLabel}`,'note'));
for(const trial of object.trials){const block=el('section',undefined,'trial');block.append(el('h3',`${trial.name} trial`));block.append(el('p',trial.processingLabel,'note'));
const metrics=el('div',undefined,'metrics');for(const [key,value] of Object.entries(trial.metrics)){if(key==='perFrameAttachment'||key==='provenance'||(value!==null&&typeof value==='object'))continue;const item=el('div',undefined,'metric');item.append(el('b',metricLabel(key,data.evidenceKind,data.benchmarkMode)));item.append(el('span',format(value,key)));metrics.append(item);}block.append(metrics);
const timeline=el('div',undefined,'timeline');timeline.setAttribute('aria-label',`${trial.name} all 110 frame states`);
for(const row of trial.timeline){const state=String(row.state||'unavailable');const tick=el('span',undefined,`tick ${['tracking','recovering'].includes(state)?state:'unavailable'}`);tick.title=`Frame ${row.frameId}: ${state}${row.pairedPoints!==null&&row.pairedPoints!==undefined?` · ${row.pairedPoints} paired landmarks`:''}${row.medianErrorPixels!==null?` · median ${format(row.medianErrorPixels)} px`:''}${row.p95ErrorPixels!==null?` · P95 ${format(row.p95ErrorPixels)} px`:''}${row.failureReason?` · ${row.failureReason}`:''}${row.unavailableReason?` · ${row.unavailableReason}`:''}`;tick.setAttribute('aria-label',tick.title);timeline.append(tick);}block.append(timeline);
const samples=el('div',undefined,'samples');for(const sample of trial.samples){const card=el('article',undefined,'sample');card.append(el('p',`Frame ${sample.frameId} · ${sample.tags.join(', ')}`));if(sample.imagePath){const image=document.createElement('img');image.src=sample.imagePath;image.alt=`${sample.inputPanelLabel||'Captured RGB'}, ${sample.estimatedPanelLabel||'estimated design render'}, and ${sample.referencePanelLabel||'supplied-pose control'} for frame ${sample.frameId}`;card.append(image);}else{card.append(el('p',`Sample unavailable: ${sample.unavailableReason||'asset could not be generated'}`,'unavailable'));}
card.append(el('p',`Tracker input: ${sample.inputPanelLabel||'Captured RGB'} · Tracking state: ${sample.state} · Estimated render: ${sample.estimatedRenderState} · Depth: ${sample.depthOcclusion||'unavailable'}`));if(sample.forcedBlankTrackerInput)card.append(el('p','Forced interruption: zero RGB is shown as tracker input, estimated pose is suppressed, and original recorded RGB with measured depth is used only for the dataset-pose reference control. Evaluation masks are not used.','note'));samples.append(card);}block.append(samples);section.append(block);}gallery.append(section);}}
fetch('./gallery.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Gallery data is unavailable');return r.json();}).then(render).catch(error=>{status.textContent=`${error.message}. Serve this private directory over a local HTTP server.`;});
</script></body></html>'''


@dataclass
class _RendererContext:
    dataset: Path
    mesh: object | None
    transform: object | None
    calibration: dict
    rgb_cache: dict
    depth_cache: dict


def _load_trace(path):
    value = _read_json(path)
    frames = value.get('frames')
    if not isinstance(frames, list) or len(frames) != 110:
        raise GalleryError(f'Sealed trace does not contain all 110 frames: {path.name}')
    if [row.get('frameId') for row in frames] != list(range(frames[0]['frameId'], frames[0]['frameId'] + 110)):
        raise GalleryError(f'Sealed trace is not consecutive: {path.name}')
    return value


def _sample_asset(context, inference_obj, evaluator_obj, trace_frames, sample,
                  trial_name, object_id, evidence_kind, write_asset):
    import numpy as np
    from PIL import Image
    from .ply_model import read_ply

    index = sample['index']
    inf_record = inference_obj['frames'][index]
    eval_record = evaluator_obj['frames'][index]
    frame = trace_frames[index]
    forced_blank_input = frame.get('forcedInterruption') is True
    output = {
        'index': index, 'frameId': frame.get('frameId'), 'tags': sample['tags'],
        'state': frame.get('state', 'unavailable'),
        'failureReason': frame.get('failureReason'),
        'estimatedRenderState': 'suppressed_nontracking',
        'depthOcclusion': None, 'imagePath': None, 'unavailableReason': None,
        'forcedBlankTrackerInput': forced_blank_input,
        'inputPanelLabel': ('Forced blank tracker input (zero RGB)' if forced_blank_input
                            else 'Captured RGB'),
        'estimatedPanelLabel': ('Estimated design render (suppressed)' if forced_blank_input
                                else 'Estimated design render'),
        'referenceRgbSource': ('original_recorded_rgb' if forced_blank_input
                               else 'captured_rgb'),
        'depthUseScope': ('dataset-pose reference control only' if forced_blank_input
                          else 'estimated/reference visualization'),
        'evaluationMaskUse': 'not used',
        'referencePanelLabel': ('Synthetic-pose control' if evidence_kind == 'synthetic_control'
                                else 'Dataset-pose reference control'),
    }
    try:
        rgb_spec = inf_record['rgb']
        if rgb_spec != eval_record['rgb'] or inf_record['calibration'] != eval_record['calibration']:
            raise GalleryError('Inference/evaluator frame identity changed after seal.')
        cache_key = (object_id, frame['frameId'])
        if cache_key not in context.rgb_cache:
            context.rgb_cache[cache_key] = _load_rgb(context.dataset, rgb_spec)
        rgb = context.rgb_cache[cache_key]
        if context.mesh is None:
            raise GalleryError('Original source model could not be loaded.')

        # Forced-interruption frames were sent to the tracker as zero RGB. Keep
        # the original recording only for the explicit supplied-pose control.
        tracker_rgb = np.zeros_like(rgb) if forced_blank_input else rgb

        depth = None
        try:
            if cache_key not in context.depth_cache:
                context.depth_cache[cache_key] = _load_depth(context.dataset, eval_record)
            depth = context.depth_cache[cache_key]
        except (OSError, ValueError, KeyError, TypeError, GalleryError):
            output['depthOcclusion'] = 'unavailable'

        estimated = None
        pose = estimated_pose_for_frame(frame)
        if forced_blank_input:
            output['estimatedRenderState'] = 'suppressed_forced_blank_input'
        elif frame.get('state') == 'tracking':
            if pose is None:
                output['estimatedRenderState'] = 'unavailable_tracking_pose_missing'
            else:
                try:
                    rotation, translation = _camera_from_matrix(pose)
                    estimated, depth_state = _overlay(tracker_rgb, context.mesh, context.transform,
                        rotation, translation, inf_record['calibration'], depth)
                    output['estimatedRenderState'] = 'rendered'
                    output['depthOcclusion'] = depth_state
                except (GalleryError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                    output['estimatedRenderState'] = 'unavailable_render_error'
                    output['estimatedRenderFailure'] = f'{type(exc).__name__}: {exc}'

        reference = None
        try:
            rotation, translation = _pose_from_annotation(eval_record)
            reference, reference_depth_state = _overlay(rgb, context.mesh, context.transform,
                rotation, translation, eval_record['calibration'], depth)
            output['referenceRenderState'] = 'rendered'
            if output['depthOcclusion'] is None:
                output['depthOcclusion'] = reference_depth_state
        except (GalleryError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            output['referenceRenderState'] = 'unavailable_reference_error'
            output['referenceRenderFailure'] = f'{type(exc).__name__}: {exc}'

        # A missing estimate falls back to the actual tracker input. On forced
        # interruption this is the zero image; the source recording is only in
        # the labeled dataset-pose reference control.
        sheet = _contact_sheet(tracker_rgb, estimated, reference,
            output['referencePanelLabel'], output['inputPanelLabel'],
            output['estimatedPanelLabel'])
        relative = f'assets/object-{object_id:06d}-{trial_name}-frame-{frame["frameId"]:06d}.jpg'
        try:
            write_asset(relative, sheet)
            output['imagePath'] = relative
        except CompactBudgetExceeded:
            raise
        except GalleryBudgetExceeded as exc:
            output['unavailableReason'] = str(exc)
        if output['imagePath'] is None and output['unavailableReason'] is None:
            output['unavailableReason'] = 'render panels unavailable'
    except CompactBudgetExceeded:
        raise
    except (OSError, ValueError, KeyError, TypeError, GalleryError) as exc:
        output['unavailableReason'] = f'{type(exc).__name__}: {exc}'
        output['estimatedRenderState'] = 'suppressed_or_unavailable'
        output['referenceRenderState'] = 'unavailable'
    return output


def generate(dataset, results, output=None):
    """Build a post-seal local gallery under ``results/demo`` by default.

    The function only reads the sealed inference/evaluation artifacts and writes
    static JSON/HTML/JPEG assets. It never calls the tracker or modifies inputs.
    """
    results_arg = Path(results).resolve()
    dataset_arg = Path(dataset).resolve()
    output_arg = Path(output).resolve() if output is not None else results_arg / 'demo'
    results_root, dataset_root, output_root = _validate_output(
        results_arg, dataset_arg, output_arg)
    loaded = _load_sealed_inputs(dataset_root, results_root)
    from .prototype_resource_store import provider, ResourceStore
    evaluator_manifest = _read_json(dataset_root / 'evaluator.json')
    if evaluator_manifest.get('resourceStorage') != loaded['seal'].get('resourceStorage'):
        raise GalleryError('Gallery storage does not match inference seal')
    inputs = provider(dataset_root, evaluator_manifest, 'gallery')
    guard = CompactOutputBudget(dataset_root, results_root) if isinstance(inputs, ResourceStore) else None

    from .ply_model import read_ply

    guard.mkdir(output_root) if guard else output_root.mkdir(parents=True, exist_ok=False)
    sample_count = 0
    objects = []
    evidence = loaded['evidence']
    report_label = evidence['provenance']

    def write_asset(relative, raw):
        _write_limited(output_root, relative, raw,
                       MAX_GALLERY_BYTES - GALLERY_METADATA_RESERVE_BYTES, guard)

    for object_id in sorted(loaded['inferenceObjects']):
        inference_obj = loaded['inferenceObjects'][object_id]
        evaluator_obj = loaded['evaluatorObjects'][object_id]
        report_obj = loaded['reportObjects'][object_id]
        seal_obj = loaded['sealObjects'][object_id]
        model_error = None
        mesh = transform = None
        try:
            from .prototype_benchmark import resource
            model_path = resource(inputs, inference_obj['model'])
            scan, _, transform = read_ply(model_path, inference_obj['name'])
            mesh = scan.original
        except (OSError, ValueError, KeyError, TypeError) as exc:
            model_error = f'{type(exc).__name__}: {exc}'

        renderer = _RendererContext(inputs, mesh, transform,
                                    inference_obj['frames'][0]['calibration'], {}, {})
        object_row = {
            'objectId': object_id, 'name': inference_obj['name'],
            'referenceId': inference_obj['referenceId'],
            'identityLabel': report_label,
            'modelAvailable': mesh is not None,
            'modelUnavailableReason': model_error,
            'trials': [],
        }
        for trial_name in TRIAL_NAMES:
            relative = seal_obj.get('trials', {}).get(trial_name)
            if not relative or relative not in loaded['trialPaths']:
                raise GalleryError(f'Seal has no {trial_name} trace for object {object_id}.')
            trial_path = loaded['trialPaths'][relative]
            trace = _load_trace(trial_path)
            frames = trace['frames']
            if [row.get('frameId') for row in frames] != [row.get('frameId') for row in inference_obj['frames']]:
                raise GalleryError(f'{trial_name} trace frame IDs differ from inference input.')
            _result_index(evaluator_obj['frames'], frames)
            report_metrics = report_obj.get('trials', {}).get(trial_name)
            if not isinstance(report_metrics, dict):
                report_metrics = {'evaluationUnavailableReason': 'sealed report has no trial metrics'}
            samples = []
            for planned in sample_plan(frames):
                sample = _sample_asset(renderer, inference_obj, evaluator_obj,
                    frames, planned, trial_name, object_id,
                    evidence['evidenceKind'], write_asset)
                if isinstance(inputs, ResourceStore):
                    renderer.rgb_cache.clear()
                    renderer.depth_cache.clear()
                sample['timelineState'] = sample['state']
                samples.append(sample)
                sample_count += 1
            score = _safe_metrics(report_metrics)
            score['provenance'] = report_label
            timeline = timeline_rows(frames, report_metrics)
            timing_keys = ('medianProcessingSeconds', 'p95ProcessingSeconds',
                           'processingScope', 'sourceFrameRate', 'liveInferenceFPSMeasured')
            timing = {key: trace[key] for key in timing_keys if key in trace}
            visible_metrics = dict(score)
            for key in ('medianProcessingSeconds', 'p95ProcessingSeconds'):
                if key in timing:
                    visible_metrics[key] = timing[key]
            object_row['trials'].append({
                'name': trial_name,
                'metrics': visible_metrics,
                'timeline': timeline,
                'samples': samples,
                'processingLabel': 'Offline per-frame wrapper timing only; no live camera FPS claim. '
                                   + str(timing.get('processingScope', 'Timing scope unavailable.')),
                'timing': _safe_metrics(timing),
            })
        objects.append(object_row)

    gallery = {
        'schemaVersion': 1,
        'title': evidence['title'],
        'datasetName': evidence['datasetName'],
        'objectIdLabel': evidence['objectIdLabel'],
        'evidenceKind': evidence['evidenceKind'],
        'benchmarkMode': evidence['benchmarkMode'],
        'realUnseenObjects': evidence['realUnseenObjects'],
        'plannedObjects': evidence['plannedObjects'],
        'attemptedObjects': loaded['report'].get('attemptedObjects', len(objects)),
        'tenObjectsEvaluated': bool(loaded['report'].get('tenObjectsEvaluated', False)),
        'benchmarkGateEligible': evidence['benchmarkGateEligible'],
        'benchmarkId': loaded['seal']['benchmarkId'],
        'selectionDigest': loaded['seal']['selectionDigest'],
        'provenance': report_label,
        'scopeNote': evidence['scopeNote'],
        'renderNote': evidence['renderNote'],
        'trackerIsolation': 'Generated only after validating the inference output seal and evaluator manifest hash. No evaluator fields are sent to the tracker.',
        'objects': objects,
        'sampleCount': sample_count,
        'assetBudgetBytes': MAX_GALLERY_BYTES,
        'generatedBytesBeforeManifest': _used_bytes(output_root),
    }
    if isinstance(inputs, ResourceStore):
        inputs.assert_sealed()
    html_bytes = _html_document().encode('utf-8')
    asset_bytes = _used_bytes(output_root)
    planned_total = asset_bytes + len(_json_bytes(gallery)) + len(html_bytes)
    for _ in range(8):
        gallery['generatedBytes'] = planned_total
        serialized_gallery = _json_bytes(gallery)
        exact_total = asset_bytes + len(serialized_gallery) + len(html_bytes)
        if exact_total == planned_total:
            break
        planned_total = exact_total
    else:
        raise GalleryError('Could not settle the generated gallery byte count.')
    _write_limited(output_root, 'gallery.json', serialized_gallery, guard=guard)
    _write_limited(output_root, 'index.html', html_bytes, guard=guard)
    return {
        'status': 'complete', 'output': str(output_root),
        'objectCount': len(objects), 'sampleCount': sample_count,
        'unavailableSamples': sum(not sample['imagePath'] for obj in objects
                                  for trial in obj['trials'] for sample in trial['samples']),
        'generatedBytes': _used_bytes(output_root),
        'assetBudgetBytes': MAX_GALLERY_BYTES,
        'benchmarkId': loaded['seal']['benchmarkId'],
        'evidenceKind': evidence['evidenceKind'],
        'realUnseenObjects': evidence['realUnseenObjects'],
        'provenance': evidence['provenance'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Sealed dataset directory containing inference.json and evaluator.json.')
    parser.add_argument('--results', type=Path, required=True,
                        help='Sealed inference/evaluation result directory.')
    parser.add_argument('--output', type=Path,
                        help='New private gallery directory; defaults to <results>/demo.')
    args = parser.parse_args(argv)
    receipt = generate(args.dataset, args.results, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt


if __name__ == '__main__':
    main()
