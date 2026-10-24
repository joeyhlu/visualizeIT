"""Input-only actual-output parity snapshots and read-only legacy observers.

Use ``observe_runtime`` around authorized masks/capture execution, then
``snapshot_capture`` on each retained output. No labels/generation payloads are
opened. CLI compares snapshot JSON files; missing evidence is a failure.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

SCHEMA = 'gotrack_larger_preparation_parity_snapshot_v1'
REQUIRED = {'id', 'input_pins', 'status', 'failure_reason', 'mask', 'detector',
    'prompt_box_xyxy', 'candidates', 'crops', 'geometry'}


def _array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'buif' or not np.all(np.isfinite(a)):
        raise ValueError('nonfinite/non-numeric parity array: ' + name)
    return a


def _equal_array(a, b, name, atol=0.):
    a, b = _array(a, name), _array(b, name)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('array shape/dtype differs: ' + name)
    if not (np.array_equal(a, b) if atol == 0 else np.allclose(a, b, rtol=0., atol=atol)):
        raise ValueError('array values differ: ' + name)


def _arrays(entry):
    """Read exact pinned NPZ arrays or in-memory stub arrays (never pickle)."""
    if not isinstance(entry, dict) or not entry:
        raise ValueError('missing array evidence')
    if set(entry) == {'pin', 'keys'}:
        from .quality_adaptation_data import sha256
        pin = entry['pin']; path = Path(pin['path'])
        if path.is_symlink() or path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError('parity array pin differs')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(entry['keys']):
                raise ValueError('missing/foreign parity array key')
            return {key: data[key].copy() for key in data.files}
    return {key: _array(value, key) for key, value in entry.items()}


def _metadata(doc):
    if (doc.get('schema') != SCHEMA or doc.get('test_payload_reads') != 0
            or doc.get('annotation_inputs') != [] or doc.get('fatal_error')):
        raise ValueError('parity snapshot not complete/input-only')
    rows = doc.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('missing requested parity rows')
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != REQUIRED:
            raise ValueError('missing/foreign parity row evidence')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('duplicate/invalid parity row')
        ids.append(row['id'])
        if row['status'] not in ('available', 'unavailable'):
            raise ValueError('invalid parity availability')
        if not isinstance(row['input_pins'], dict) or set(row['input_pins']) != {'input', 'video', 'mesh', 'bank'}:
            raise ValueError('missing exact input pin evidence')
        for pin in row['input_pins'].values():
            if pin is None and row['status'] == 'unavailable':
                continue
            if (not isinstance(pin, dict) or set(pin) != {'sha256', 'bytes'} or
                    not isinstance(pin['sha256'], str) or len(pin['sha256']) != 64 or
                    any(c not in '0123456789abcdef' for c in pin['sha256']) or type(pin['bytes']) is not int or pin['bytes'] <= 0):
                raise ValueError('invalid parity input pin')
        if row['status'] == 'unavailable' and not row['failure_reason']:
            raise ValueError('unavailable parity row missing failure reason')
    if doc.get('requested_ids') != ids:
        raise ValueError('requested row inventory differs or missing rows')
    return rows


def compare_runs(reference, candidate):
    """Fail closed on missing rows, unavailable mismatch or nonfinite evidence."""
    left, right = _metadata(reference), _metadata(candidate)
    if reference.get('device') != candidate.get('device') or reference.get('device') not in ('cpu', 'cuda'):
        raise ValueError('parity device differs')
    if reference['requested_ids'] != candidate['requested_ids']:
        raise ValueError('requested parity identity/order differs')
    available = 0
    for a, b in zip(left, right):
        identity = a['id']
        for key in ('id', 'input_pins', 'status', 'failure_reason', 'prompt_box_xyxy'):
            if a[key] != b[key]:
                raise ValueError('parity differs: ' + identity + '/' + key)
        # Mask evidence may be available even when recovery/capture failed.
        if (a['mask'] is None) != (b['mask'] is None):
            raise ValueError('mask availability differs: ' + identity)
        if a['mask'] is not None:
            if a['mask'] != b['mask']:
                raise ValueError('mask pixels differ: ' + identity)
            if a['detector'] is None or b['detector'] is None:
                raise ValueError('missing observed-mask association evidence: ' + identity)
        da, db = a['detector'], b['detector']
        if (da is None) != (db is None):
            raise ValueError('association availability differs: ' + identity)
        if da is not None:
            for key in ('scores', 'selected_proposal', 'reason'):
                if key not in da or key not in db:
                    raise ValueError('missing association evidence: ' + identity + '/' + key)
            _equal_array(da['scores'], db['scores'], identity + '/association_scores')
            if da['selected_proposal'] != db['selected_proposal'] or da['reason'] != db['reason']:
                raise ValueError('proposal identity differs: ' + identity)
        if a['candidates'] is not None or b['candidates'] is not None:
            if a['candidates'] is None or b['candidates'] is None:
                raise ValueError('candidate evidence unavailable: ' + identity)
            _equal_array(a['candidates'], b['candidates'], identity + '/ordered_candidates')
        if a['status'] == 'available':
            available += 1
            if a['mask'] is None or da is None or a['candidates'] is None:
                raise ValueError('missing available-row inference evidence: ' + identity)
            for kind in ('crops', 'geometry'):
                aa, bb = _arrays(a[kind]), _arrays(b[kind])
                from .quality_adaptation_data import CROP_KEYS
                from .quality_adaptation_larger import GEOMETRY_KEYS
                expected = CROP_KEYS if kind == 'crops' else GEOMETRY_KEYS
                if set(aa) != set(bb) or set(aa) != set(expected):
                    raise ValueError('missing/foreign array keys: ' + identity + '/' + kind)
                for key in aa:
                    tolerance = 1e-4 if key == 'actual_flow' else 1e-6 if key == 'actual_confidence' else 0.
                    _equal_array(aa[key], bb[key], identity + '/' + key, tolerance)
        elif any(a[k] is not None or b[k] is not None for k in ('crops', 'geometry')):
            raise ValueError('unavailable row contains substituted arrays')
    # Equal failures are retained evidence, never a successful numerical validation.
    if not available:
        raise ValueError('no available rows; numerical parity not established')
    return dict(schema='gotrack_larger_preparation_parity_result_v1', passed=True,
        requested=len(left), available=available, unavailable=len(left) - available,
        flow_atol=1e-4, confidence_atol=1e-6, test_payload_reads=0, annotation_inputs=[])


def compare_aba(first_a, b, second_a):
    """Compare whole-run variant replays on identical inputs.

    This checks replay parity. Actual persistent A(image A)->B(image B)->A state
    diagnostics are optimized.masks_aba_diagnostic/recovery_aba_diagnostic.
    """
    return dict(a_b=compare_runs(first_a, b), a_a_repeat=compare_runs(first_a, second_a))


def _pixel_pin(image):
    image = np.ascontiguousarray(image)
    return dict(shape=list(image.shape), dtype=str(image.dtype), sha256=hashlib.sha256(image.tobytes()).hexdigest())


def _detector_evidence(recorded, observed):
    """An observer augments legacy diagnostics; it cannot replace contradictory evidence."""
    if observed is None:
        return recorded
    if recorded is not None and any(key not in observed or observed[key] != value for key, value in recorded.items()):
        raise ValueError('observer detector differs from recorded producer evidence')
    return observed


def snapshot_capture(root, *, observations=None, prepared_root=None):
    """Authenticate actual plans/receipts/crops and collect only inference inputs.

    ROOT29 legacy observations come from ``observe_runtime``. ROOT30 has native
    candidate diagnostics but legacy association scores still need the observer.
    The returned manifest references pinned crop files rather than copying data.
    """
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_data import read_json, descriptor
    from .quality_runner import read_input, read_rgb
    from .vision import cv2
    root = Path(root)
    plan, states = larger.read_capture(root)
    receipt = read_json(root / 'capture-receipt.json')
    observations = observations or dict(detections=[], candidates=[])
    if observations.get('test_payload_reads', 0) != 0 or observations.get('annotation_inputs', []) != []:
        raise ValueError('observer used annotation/TEST inputs')
    diagnostics_path = root / 'preparation-diagnostics.json'
    native = read_json(diagnostics_path)['candidates'] if diagnostics_path.exists() else []
    candidates_by_id = {row['id']: row['poses'] for row in native}
    detections = {row['input_sha256']: row['detector'] for row in observations['detections']}
    candidates_by_rgb = {row['rgb_sha256']: row['poses'] for row in observations['candidates']}
    if len(detections) != len(observations['detections']) or len(candidates_by_rgb) != len(observations['candidates']):
        raise ValueError('ambiguous duplicate observer identity')
    by_id = {}
    for physical in plan['rows']:
        pins = {}
        for key in ('input', 'video', 'mesh', 'bank'):
            pin = physical[key]
            if pin is None:
                pins[key] = None
                continue
            larger.pinned_file(pin)
            pins[key] = {field: pin[field] for field in ('sha256', 'bytes')}
        mask = box = None
        observed = detections.get(physical['input']['sha256']) if physical['input'] else None
        detector = observed
        if prepared_root is not None:
            mask_receipt = Path(prepared_root) / f"mask-{int(physical['object_id']):06d}-{physical['view_index']:02d}.json"
            if mask_receipt.exists():
                diagnostics = read_json(mask_receipt).get('diagnostics', {})
                detector = _detector_evidence(diagnostics.get('detector'), detector)
                box = diagnostics.get('prompt_box_xyxy', box)
        if physical['mask'] is not None:
            image = cv2.imread(str(larger.pinned_file(physical['mask'])), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('unreadable observed mask')
            mask = _pixel_pin(image)
            producer = read_json(larger.pinned_file(physical['mask_index']))
            larger.validate_observed_mask(physical, producer, plan)
            detector = _detector_evidence(producer['detector'], detector)
            box = producer['prompt_box_xyxy']
        poses = candidates_by_id.get(physical['id'])
        if poses is None and physical['observation_status'] == 'available':
            bundle = Path(physical['input']['path']).parent
            rgb = read_rgb(bundle, read_input(bundle), physical['frame_id'])
            poses = candidates_by_rgb.get(_pixel_pin(rgb)['sha256'])
        by_id[physical['id']] = dict(input_pins=pins, mask=mask, detector=detector,
            prompt_box_xyxy=box, candidates=poses)
    rows = []
    for state in states:
        physical_id = state['id'].rsplit(':iteration', 1)[0]
        row = dict(by_id[physical_id], id=state['id'], status=state['status'],
            failure_reason=state['failure_reason'], crops=None, geometry=None)
        if state['status'] == 'available':
            for kind, key, keys in (('crops', 'crops', larger.CROP_KEYS), ('geometry', 'capture', larger.GEOMETRY_KEYS)):
                path = descriptor(root, state[key], verify=True)
                row[kind] = dict(pin=dict(path=path.as_posix(), bytes=path.stat().st_size,
                    sha256=larger.sha256(path)), keys=sorted(keys))
        rows.append(row)
    return dict(schema=SCHEMA, device=receipt.get('device', 'cpu'), rows=rows,
        requested_ids=[row['id'] for row in rows], fatal_error=receipt.get('fatal_error'),
        physical_frames_requested=len(plan['rows']), refinement_states_requested=len(rows),
        test_payload_reads=0, annotation_inputs=[])


@contextmanager
def observe_runtime(output, *, timings=None):
    """Observe legacy scores/candidates without changing results or source bytes.

    Call only within a cloud stage already admitted by the parent supervisor.
    The observer records recover's returned candidate order, not reference poses.
    It is intentionally scoped to a single sequential worker.
    """
    from . import quality_cnos as cnos, quality_foundpose as foundpose
    from .quality_adaptation_data import sha256
    evidence = dict(detections=[], candidates=[], test_payload_reads=0, annotation_inputs=[])
    original_detect, original_recover = cnos.detect, foundpose.FoundPoseRecovery.recover
    original_numpy = cnos.np
    captured = []
    class NumpyObserver:
        def __getattr__(self, name):
            return getattr(original_numpy, name)
        def argsort(self, values, *args, **kwargs):
            captured[:] = list(values)
            return original_numpy.argsort(values, *args, **kwargs)
    def detect(bundle, rgb, *args, **kwargs):
        captured.clear()
        mask, detector = original_detect(bundle, rgb, *args, **kwargs)
        scores = list(captured)
        ranked = original_numpy.argsort(scores)[::-1]
        observed = dict(detector, scores=scores,
            selected_proposal=(int(ranked[0]) if len(ranked) else None))
        evidence['detections'].append(dict(input_sha256=sha256(Path(bundle) / 'input.json'), detector=observed))
        return mask, detector
    def recover(self, frame, mask, *args, **kwargs):
        poses = original_recover(self, frame, mask, *args, **kwargs)
        evidence['candidates'].append(dict(rgb_sha256=_pixel_pin(frame.rgb)['sha256'],
            poses=[np.asarray(pose).tolist() for pose in poses]))
        return poses
    output = Path(output)
    if output.exists():
        raise FileExistsError('observer evidence exists; preserve prior repeat')
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(cnos, 'np', NumpyObserver()))
            stack.enter_context(patch.object(cnos, 'detect', detect))
            stack.enter_context(patch.object(foundpose.FoundPoseRecovery, 'recover', recover))
            yield evidence
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, sort_keys=True, allow_nan=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('snapshot')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--observations', type=Path)
    p.add_argument('--prepared-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('compare')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--repeat-reference', type=Path)
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    if args.command == 'snapshot':
        if args.output.exists():
            raise FileExistsError('parity snapshot exists')
        result = snapshot_capture(args.root, observations=read(args.observations) if args.observations else None,
            prepared_root=args.prepared_root)
        args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False), encoding='utf-8')
    else:
        result = (compare_aba(read(args.reference), read(args.candidate), read(args.repeat_reference))
            if args.repeat_reference else compare_runs(read(args.reference), read(args.candidate)))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
