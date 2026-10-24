"""Prepare explicit frozen normalized views, automatic masks and standard banks.

No acquisition or model download is implemented. Each expensive command is an
explicit CPU-only stage under the larger controller's shared watchdog. Generation
truth stays outside observed bundles and is not read by assemble/capture/extract.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from . import quality_adaptation_larger as larger
from .quality_adaptation_data import canonical_hash, read_json, require_hash, sha256
from .quality_adaptation_runtime import cloud_guard, tree_bytes, write_json

SCHEMA = 'gotrack_larger_preparation_spec_v1'


@contextmanager
def legacy_cnos_timing(timings, view):
    """Time unchanged CNOS call boundaries without retaining model instances."""
    from unittest.mock import patch
    from . import quality_cnos as cnos
    cnos.upstream_path()
    from utils import fastsam_util
    segmentor_class = fastsam_util.FastSAM
    original_init = segmentor_class.__init__
    original_generate = segmentor_class.generate_detection_proposals
    original_descriptor = cnos.load_descriptor
    def initialize(self, *args, **kwargs):
        with timings.step('fastsam_load', view=view, temperature='cold'):
            original_init(self, *args, **kwargs)
    def generate(self, *args, **kwargs):
        with timings.step('fastsam_infer', view=view, temperature='cold'):
            return original_generate(self, *args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(segmentor_class, '__init__', initialize))
        stack.enter_context(patch.object(segmentor_class, 'generate_detection_proposals', generate))
        def descriptor(*args, **kwargs):
            with timings.step('dino_load', view=view, temperature='cold'):
                model = original_descriptor(*args, **kwargs)
            model_class = type(model)
            original_forward = model_class.forward_features
            calls = 0
            def forward(self, *a, **kw):
                nonlocal calls
                temperature = 'cold' if calls == 0 else 'warm'
                calls += 1
                with timings.step('dino_infer', view=view, temperature=temperature):
                    return original_forward(self, *a, **kw)
            stack.enter_context(patch.object(model_class, 'forward_features', forward))
            return model
        stack.enter_context(patch.object(cnos, 'load_descriptor', descriptor))
        yield


def file_pin(path):
    path = Path(path)
    return dict(path=path.as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


def validate_spec(doc, *, verify_files=False):
    larger._variant_keys(doc, ('schema', 'runtime_contract', 'source_contract', 'assets', 'views', 'sealed_test', 'limits'), 'preparation spec')
    if doc['schema'] != SCHEMA or doc['limits'] != larger.LIMITS:
        raise ValueError('preparation schema/resource bounds differ')
    for key in ('runtime_contract', 'source_contract'):
        larger._pin_shape(doc[key], source=True)
    larger._test_metadata(doc['sealed_test'])
    if not isinstance(doc['assets'], list) or not 2 <= len(doc['assets']) <= larger.LIMITS['max_assets']:
        raise ValueError('bounded multi-object asset inventory required')
    assets, ownership = {}, {}
    for asset in doc['assets']:
        larger._keys(asset, ('object_id', 'split', 'physical_object_id', 'mesh'), 'preparation asset')
        obj, split = asset['object_id'], asset['split']
        if (not isinstance(obj, str) or not obj.isdecimal() or str(int(obj)) != obj or obj == '7'
                or obj in assets or split not in ('train', 'val')
                or asset['physical_object_id'] != f'hot3d:obj_{int(obj):06d}'):
            raise ValueError('duplicate/TEST/invalid asset')
        larger._pin_shape(asset['mesh'])
        for identity in (asset['physical_object_id'], asset['mesh']['sha256']):
            if ownership.setdefault(identity, split) != split:
                raise ValueError('cross-split physical/mesh acquisition identity')
        assets[obj] = asset
    if set(a['split'] for a in assets.values()) != {'train', 'val'}:
        raise ValueError('TRAIN and VAL owners required')
    if not isinstance(doc['views'], list) or not 2 <= len(doc['views']) <= larger.LIMITS['max_physical_frames']:
        raise ValueError('invalid bounded view inventory')
    seen, ids, owners, states = set(), set(), {}, 0
    for view in doc['views']:
        larger._keys(view, ('object_id', 'view_index', 'iterations', 'sequence_id', 'capture_group'), 'preparation view')
        key = (view['object_id'], view['view_index'])
        if (view['object_id'] not in assets or type(view['view_index']) is not int or not 0 <= view['view_index'] < 24
                or key in seen or not isinstance(view['sequence_id'], str) or not view['sequence_id']
                or view['sequence_id'] in ids or not isinstance(view['capture_group'], str) or not view['capture_group']):
            raise ValueError('invalid/duplicate frozen normalized view')
        iterations = view['iterations']
        if (not isinstance(iterations, list) or not iterations or iterations != sorted(set(iterations))
                or any(type(i) is not int or i not in (0, 1, 3) for i in iterations)):
            raise ValueError('invalid prescribed iterations')
        split = assets[view['object_id']]['split']
        if owners.setdefault(view['capture_group'], split) != split:
            raise ValueError('cross-split capture identity')
        seen.add(key); ids.add(view['sequence_id']); states += len(iterations)
    if not 2 <= states <= larger.LIMITS['max_states'] or {v['object_id'] for v in doc['views']} != set(assets):
        raise ValueError('frozen state count/asset coverage differs')
    if verify_files:
        larger.verify_source_contract(doc)
        for asset in doc['assets']:
            larger.pinned_file(asset['mesh'])
    return doc


def frozen(root, device=None, variant=None):
    root = Path(root)
    doc = validate_spec(read_json(root / 'spec.json'))
    receipt = read_json(root / 'freeze.json')
    if (receipt.get('spec_sha256') != sha256(root / 'spec.json')
            or receipt.get('split_sha256') != sha256(root / 'split.json')
            or receipt.get('views_sha256') != canonical_hash(doc['views'])):
        raise ValueError('frozen preparation/source/split mutation')
    larger.verify_source_contract(doc)
    larger.preparation_variant(doc, variant)
    larger.preparation_variant(receipt, larger.preparation_variant(doc))
    if device is not None and larger.validate_device(device) != receipt.get('device', 'cpu'):
        raise ValueError('execution device differs from frozen preparation variant')
    return doc


def freeze(spec, expected, output, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    larger.validate_device(device)
    raw = Path(spec).read_bytes()
    if len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != require_hash(expected):
        raise ValueError('preparation spec byte hash differs')
    doc = validate_spec(larger.inference._loads_unique(raw, 'preparation spec'), verify_files=True)
    larger.preparation_variant(doc, variant)
    output = larger._safe_output(output)
    larger.resources(output)
    from .quality_ranker_data import freeze_split_manifest
    assets = {a['object_id']: a for a in doc['assets']}
    records = []
    for view in doc['views']:
        asset = assets[view['object_id']]
        records.append(dict(physical_object_id=asset['physical_object_id'],
            sequence_id=view['sequence_id'], scene_id=view['capture_group'],
            split='validation' if asset['split'] == 'val' else 'train',
            mesh_sha256=asset['mesh']['sha256'], frame_id=0, source_frame_id=0,
            background_id='procedural:gray', background_kind='procedural', provenance_kind='synthetic'))
    split = freeze_split_manifest(dict(records=records),
        label_policy=dict(max_rotation_error_deg=10., max_translation_error_m=.02))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'spec.json').write_bytes(raw)
    write_json(output, 'split.json', split)
    write_json(output, 'freeze.json', dict(spec_sha256=expected, split_sha256=sha256(output / 'split.json'),
        views_sha256=canonical_hash(doc['views']), device=device, execution_variant=device + '-v1', preparation_variant=variant, split_frozen_before_generation=True,
        physical_frames=len(doc['views']), states=sum(len(v['iterations']) for v in doc['views']), test_payload_reads=0))
    return dict(frozen=True, physical_frames=len(records))


def _select(doc, object_id, view_index=None):
    if object_id not in {a['object_id'] for a in doc['assets']}:
        raise ValueError('object outside frozen plan')
    rows = [v for v in doc['views'] if v['object_id'] == object_id and (view_index is None or v['view_index'] == view_index)]
    if not rows:
        raise ValueError('view outside frozen plan')
    return rows


def _directory(root, view):
    # Sequence identity stays metadata; directories use validated numeric IDs.
    return Path(root) / 'views' / f"obj-{int(view['object_id']):06d}-view-{view['view_index']:02d}"


def render(root, object_id, view_index=None, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    root = Path(root); doc = frozen(root, device, variant)
    from .quality_ranker_data import read_split_manifest
    from .quality_ranker_synthetic import _render_bundle
    split = read_split_manifest(root / 'split.json')
    asset = next(a for a in doc['assets'] if a['object_id'] == object_id)
    results = []
    with larger.stage_lock():
        for view in _select(doc, object_id, view_index):
            larger.resources(root)
            target = _directory(root, view)
            group = next(g for g in split['groups'] if g['sequence_id'] == view['sequence_id'])
            try:
                result = _render_bundle(larger.pinned_file(asset['mesh']), target, group,
                    frames=1, width=640, height=480, fps=10., mesh_units='metres',
                    observationally_ambiguous=True, cloud_execution=True,
                    split_manifest_path=root / 'split.json', view_mode='normalized24-v1', view_index=view['view_index'])
                # Producer records its own output pins. Subsequent input-only
                # assembly copies metadata and never opens generation truth.
                receipt = dict(status='available', input=file_pin(target / 'bundle/input.json'),
                    video=file_pin(target / 'bundle/synthetic.mp4'), mesh=file_pin(target / 'bundle/object.glb'),
                    generation=file_pin(target / 'generation.json'), result=result)
            except Exception as exc:
                receipt = dict(status='unavailable', failure_reason=f'{type(exc).__name__}: {exc}'[:1024])
            root.mkdir(parents=True, exist_ok=True)
            name = f"render-{int(object_id):06d}-{view['view_index']:02d}.json"
            if (root / name).exists():
                raise FileExistsError('render attempt receipt already exists; preserve failed attempts')
            write_json(root, name, receipt)
            results.append(dict(view_index=view['view_index'], **receipt))
    return results


def bank(root, object_ids, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    root = Path(root); doc = frozen(root, device, variant)
    # Authenticate EVERY selected asset before loading any model.
    meshes = {}
    for obj in object_ids:
        _select(doc, obj)
        asset = next(a for a in doc['assets'] if a['object_id'] == obj)
        meshes[obj] = larger.pinned_file(asset['mesh'])
    # Verify existing checkpoint and all immutable upstream bytes before imports.
    # Existing loader uses the pinned checkpoint name below; no fetch is invoked.
    from .quality_gotrack import load_network, TexturedRenderer
    from .quality_adaptation_runtime import verify_checkpoint
    verify_checkpoint()
    torch = larger.configure_torch(device)
    from .quality_foundpose import build_bank
    results, network = [], None
    with larger.stage_lock():
        try:
            network = load_network(device).eval()
            for obj in object_ids:
                _select(doc, obj)
                asset = next(a for a in doc['assets'] if a['object_id'] == obj)
                mesh = meshes[obj]
                target = root / 'banks' / f'obj-{int(obj):06d}.pt'
                if target.exists() or target.with_suffix('.part').exists():
                    raise FileExistsError('bank output/partial exists; no replacement')
                renderer = None
                began = time.time()
                try:
                    larger.resources(root)
                    renderer = TexturedRenderer(mesh, 'synthetic_' + asset['mesh']['sha256'][:16], unlit=True)
                    state = build_bank(renderer, network.backbone, device, target)
                    if state.get('template_count') != 798 or state.get('asset_sha256') != asset['mesh']['sha256']:
                        raise ValueError('bank does not bind standard 798 templates/original asset')
                    result = dict(status='available', device=device, bank=file_pin(target), template_count=798,
                        elapsed_seconds=time.time() - began, asset_sha256=asset['mesh']['sha256'])
                    state = None
                except Exception as exc:
                    result = dict(status='unavailable', failure_reason=f'{type(exc).__name__}: {exc}'[:1024], elapsed_seconds=time.time() - began)
                finally:
                    if renderer is not None:
                        renderer.close()
                write_json(root, f'bank-{int(obj):06d}.json', result)
                results.append(result)
        finally:
            if network is not None:
                network.cpu()
            network = None
            larger.cleanup_device(device, torch)
    return results




def _existing_models():
    """Authenticate pre-existing weights without any acquisition helper."""
    from .quality_assets import CACHE, MODELS
    for name in ('gotrack_checkpoint.pt', 'sam2.1_hiera_base_plus.pt', 'FastSAM-x.pt', 'dinov2_vitl14_pretrain.pth'):
        path = CACHE / 'checkpoints' / name
        if not path.is_file() or path.is_symlink() or sha256(path) != MODELS[name]['sha256']:
            raise ValueError('missing/changed existing model: ' + name)
    return CACHE


def cnos_bank(root, object_id, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    root = Path(root); doc = frozen(root, device, variant)
    view = _select(doc, object_id)[0]
    bundle = _directory(root, view) / 'bundle'
    render_receipt = read_json(root / f"render-{int(object_id):06d}-{view['view_index']:02d}.json")
    for key in ('input', 'video', 'mesh'):
        larger.pinned_file(render_receipt[key])
    _existing_models()
    target = root / 'banks' / f'obj-{int(object_id):06d}-cnos.npz'
    if target.exists():
        raise FileExistsError('CNOS bank exists; preserve original')
    from .quality_cnos import build_descriptors
    with larger.stage_lock():
        # build_descriptors closes its renderer and returns, releasing DINO.
        torch = larger.configure_torch(device)
        try:
            build_descriptors(bundle, device=device, output=target)
        finally:
            larger.cleanup_device(device, torch)
    result = dict(status='available', device=device, bank=file_pin(target), templates=57,
        mesh_sha256=next(a['mesh']['sha256'] for a in doc['assets'] if a['object_id'] == object_id))
    write_json(root, f'cnos-{int(object_id):06d}.json', result)
    return result


def masks(root, object_id, view_index=None, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    root = Path(root); doc = frozen(root, device, variant)
    if variant == 'amortized-v1':
        from .quality_adaptation_larger_optimized import masks_phased
        return masks_phased(root, object_id, view_index, cloud_execution=cloud_execution, device=device)
    model_cache = _existing_models()
    results = []
    timings = larger.CaptureTimings(device)
    with larger.stage_lock():
        for view in _select(doc, object_id, view_index):
            name = f"mask-{int(object_id):06d}-{view['view_index']:02d}.json"
            if (root / name).exists():
                raise FileExistsError('mask attempt receipt exists; preserve failure')
            target = _directory(root, view)
            render_receipt = read_json(root / f"render-{int(object_id):06d}-{view['view_index']:02d}.json")
            device_ready, torch = False, None
            detection = prompt_box_xyxy = None
            try:
                for key in ('input', 'video', 'mesh'):
                    larger.pinned_file(render_receipt[key])
                bank = larger.pinned_file(read_json(root / f'cnos-{int(object_id):06d}.json')['bank'])
                from .quality_runner import read_input, read_rgb
                from .vision import cv2
                import numpy as np
                torch = larger.configure_torch(device)
                timings.torch = torch
                device_ready = True
                bundle = target / 'bundle'
                manifest = read_input(bundle)
                rgb = read_rgb(bundle, manifest, 0)
                from .quality_cnos import detect
                with timings.step('cnos_total', view=view['sequence_id']), legacy_cnos_timing(timings, view['sequence_id']):
                    initial_mask, detection = detect(bundle, rgb, device=device, bank_path=bank)
                larger.cleanup_device(device, torch)
                if initial_mask is None:
                    raise ValueError('CNOS observed initial association unavailable')
                yy, xx = np.nonzero(initial_mask)
                if len(xx) < 1:
                    raise ValueError('empty observed CNOS association')
                prompt_box_xyxy = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                # SAM2 receives an observed CNOS bounding box, never renderer truth.
                # CNOS/FastSAM models have been released before SAM2 construction.
                sys.path.insert(0, str(model_cache / 'sources/sam2'))
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from .quality_sam2 import deterministic_position_grid
                from sam2.modeling.position_encoding import PositionEmbeddingRandom
                original_grid = PositionEmbeddingRandom.forward
                predictor = None
                try:
                    PositionEmbeddingRandom.forward = deterministic_position_grid
                    with timings.step('sam2_load', view=view['sequence_id'], temperature='cold'):
                        model = build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',
                            str(model_cache / 'checkpoints/sam2.1_hiera_base_plus.pt'), device=device, apply_postprocessing=False)
                        predictor = SAM2ImagePredictor(model)
                    with timings.step('sam2_infer', view=view['sequence_id'], temperature='cold'), torch.inference_mode():
                        predictor.set_image(rgb)
                        predictions, scores, _ = predictor.predict(
                            box=np.array([xx.min(), yy.min(), xx.max(), yy.max()], dtype=np.float32), multimask_output=True)
                    selected = np.asarray(predictions[int(np.argmax(scores))], bool)
                finally:
                    predictor = model = None
                    PositionEmbeddingRandom.forward = original_grid
                    with timings.step('sam2_release', view=view['sequence_id']):
                        larger.cleanup_device(device, torch)
                if selected.shape != rgb.shape[:2] or not np.any(selected):
                    raise ValueError('SAM2 observed mask unavailable/empty')
                output = target / 'observed-masks'
                output.mkdir(parents=True, exist_ok=False)
                if not cv2.imwrite(str(output / '0.png'), selected.astype(np.uint8) * 255):
                    raise OSError('SAM2 mask write failed')
                mask_pin = file_pin(output / '0.png')
                producer = dict(schema='gotrack_larger_observed_sam_mask_v1', frame_id=0,
                    input_sha256=render_receipt['input']['sha256'], mesh_sha256=render_receipt['mesh']['sha256'],
                    video_sha256=render_receipt['video']['sha256'], source_contract_sha256=doc['source_contract']['sha256'],
                    producer_sha256=sha256(Path(__file__)), device=device,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    tf32=False if device == 'cuda' else None, cnos_bank=file_pin(bank),
                    automatic=True, mask_origin='observed_CNOS_box_then_SAM2_image_predictor',
                    prompt_box_xyxy=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                    detector=detection, mask=mask_pin, test_payload_reads=0, annotation_inputs=[])
                write_json(output, 'results.json', producer)
                result = dict(status='available', device=device, mask_index=file_pin(output / 'results.json'), mask=mask_pin)
            except Exception as exc:
                result = dict(status='unavailable', failure_reason=f'{type(exc).__name__}: {exc}'[:1024])
            finally:
                if device_ready:
                    larger.cleanup_device(device, torch)
            if detection is not None:
                result['diagnostics'] = dict(detector=detection)
                if prompt_box_xyxy is not None:
                    result['diagnostics']['prompt_box_xyxy'] = prompt_box_xyxy
            write_json(root, name, result)
            results.append(result)
    write_json(root, f"masks-legacy-{int(object_id):06d}-{_select(doc, object_id, view_index)[0]['view_index']:02d}.json",
        dict(timings=timings.report(), test_payload_reads=0, annotation_inputs=[]))
    return results


def assemble(root, output=None, *, cloud_execution=False, device='cpu', variant='legacy-v1'):
    cloud_guard(cloud_execution)
    root = Path(root); doc = frozen(root, device, variant)
    plan = {key: doc[key] for key in ('runtime_contract', 'source_contract', 'sealed_test', 'limits')}
    if 'preparation_variant' in doc:
        plan['preparation_variant'] = doc['preparation_variant']
    plan.update(schema=larger.SCHEMA, assets=[], rows=[])
    by_obj = {}
    for asset in doc['assets']:
        bank_path = root / f"bank-{int(asset['object_id']):06d}.json"
        receipt = read_json(bank_path) if bank_path.is_file() else dict(status='unavailable')
        item = dict(asset, original_mesh_sha256=asset['mesh']['sha256'],
            bank=receipt.get('bank') if receipt['status'] == 'available' else None)
        plan['assets'].append(item); by_obj[item['object_id']] = item
    generations = []
    for view in doc['views']:
        asset = by_obj[view['object_id']]
        render_path = root / f"render-{int(view['object_id']):06d}-{view['view_index']:02d}.json"
        mask_path = root / f"mask-{int(view['object_id']):06d}-{view['view_index']:02d}.json"
        render = read_json(render_path) if render_path.is_file() else dict(status='unavailable', failure_reason='render_not_attempted')
        mask = read_json(mask_path) if mask_path.is_file() else dict(status='unavailable', failure_reason='mask_not_attempted')
        okay = render['status'] == mask['status'] == 'available' and asset['bank'] is not None
        reason = None if okay else render.get('failure_reason') or mask.get('failure_reason') or 'bank_unavailable'
        row = dict(view, split=asset['split'], id=view['sequence_id'] + ':frame0', frame_id=0, timestamp_s=0.,
            input=render.get('input'), video=render.get('video'), mesh=render.get('mesh') or asset['mesh'],
            mask_index=mask.get('mask_index'), mask=mask.get('mask'), bank=asset['bank'],
            observation_status='available' if okay else 'unavailable', failure_reason=reason)
        plan['rows'].append(row)
        if render['status'] == 'available':
            generations.append(dict(sequence_id=view['sequence_id'], generation=render['generation']))
    larger.validate_plan(plan)  # No generation/poses payload is read here.
    destination = root if output is None else larger._safe_output(output)
    if destination != root:
        destination.mkdir(parents=True, exist_ok=False)
    for name in ('input-plan.json', 'label-plan.json'):
        if (destination / name).exists():
            raise FileExistsError('assembled plan is immutable')
    write_json(destination, 'input-plan.json', plan)
    write_json(destination, 'label-plan.json', dict(schema=larger.LABEL_SCHEMA,
        split_manifest=file_pin(root / 'split.json'), generations=generations))
    return dict(plan=file_pin(destination / 'input-plan.json'), labels=file_pin(destination / 'label-plan.json'),
        requested=len(larger.state_rows(plan)), unavailable_physical=sum(r['observation_status'] != 'available' for r in plan['rows']))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('freeze', 'render', 'bank', 'cnos-bank', 'masks', 'assemble'):
        p = sub.add_parser(command)
        p.add_argument('--cloud-execution', action='store_true')
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        p.add_argument('--variant', choices=larger.PREPARATION_VARIANTS, default='legacy-v1')
        p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
        if command == 'freeze':
            p.add_argument('--spec', type=Path, required=True)
            p.add_argument('--spec-sha256', required=True)
            p.add_argument('--output', type=Path, required=True)
        else:
            p.add_argument('--root', type=Path, required=True)
        if command == 'assemble':
            p.add_argument('--output', type=Path)
        if command in ('render', 'bank', 'cnos-bank', 'masks'):
            p.add_argument('--object-id', required=True)
        if command in ('render', 'masks'):
            p.add_argument('--view-index', type=int)
    args = parser.parse_args(argv)
    cloud_guard(args.cloud_execution)
    variant_args = {} if args.variant == 'legacy-v1' else dict(variant=args.variant)
    if args.command == 'freeze':
        result = freeze(args.spec, args.spec_sha256, args.output, cloud_execution=True, device=args.device, variant=args.variant)
    elif args.command == 'assemble':
        result = assemble(args.root, args.output, cloud_execution=True, device=args.device, **variant_args)
    elif not args._worker:
        doc = frozen(args.root, args.device, args.variant)
        rows = _select(doc, args.object_id, getattr(args, 'view_index', None))
        result = larger.supervise([sys.executable, '-B', '-m', 'bench.quality_adaptation_larger_prepare', *argv, '--_worker'],
            scratch=args.root, requested=rows, device=args.device)
    elif args.command == 'render':
        result = render(args.root, args.object_id, args.view_index, cloud_execution=True, device=args.device, **variant_args)
    elif args.command == 'bank':
        result = bank(args.root, [args.object_id], cloud_execution=True, device=args.device, **variant_args)
    elif args.command == 'cnos-bank':
        result = cnos_bank(args.root, args.object_id, cloud_execution=True, device=args.device, **variant_args)
    else:
        result = masks(args.root, args.object_id, args.view_index, cloud_execution=True, device=args.device, variant=args.variant)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
