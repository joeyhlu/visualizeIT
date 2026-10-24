"""Opt-in preparation amortization. Imports do not load models or read labels.

The mutable renderer, refiner, clock and capture sink always belong to one view.
Only immutable object search data and one mask-phase model are shared.
"""
from __future__ import annotations
from contextlib import contextmanager
import gc
from pathlib import Path
import time
import numpy as np
from .quality_adaptation_data import sha256, read_json
from .quality_adaptation_runtime import cloud_guard, write_json, tree_bytes

VARIANT = 'amortized-v1'
PROPOSAL_BYTES_PER_VIEW = 200 * 640 * 480 * 4 + 200 * 4 * 4 + 4096


class Timings:
    """Synchronized wall timings; instrumentation overhead is included."""
    def __init__(self, device='cpu', torch=None):
        self.device, self.torch, self.events = device, torch, []

    def sync(self):
        if self.device == 'cuda' and self.torch is not None:
            self.torch.cuda.synchronize()

    @contextmanager
    def step(self, name, *, view=None, temperature='unspecified'):
        self.sync(); began = time.perf_counter(); status = 'available'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            self.sync()
            self.events.append(dict(step=name, view=view, temperature=temperature,
                seconds=time.perf_counter() - began, status=status))

    def report(self):
        result = dict(events=self.events, timing_overhead='synchronization and instrumentation included',
            aggregation='events can be nested; do not sum as stage wall time',
            process_tree_rss_source='external stage supervision receipt',
            cuda_synchronized=self.device == 'cuda', observed_counts={})
        for event in self.events:
            name = event['step']
            result['observed_counts'][name] = result['observed_counts'].get(name, 0) + 1
        if self.device == 'cuda' and self.torch is not None:
            result['cuda_max_memory_allocated_bytes'] = self.torch.cuda.max_memory_allocated()
        return result


def select_proposal(scores, masks, bank_sha256):
    """Exact no-prior quality_cnos.detect selection, including reverse argsort ties."""
    scores = list(scores)
    if not scores or len(scores) != len(masks) or not np.all(np.isfinite(scores)):
        raise ValueError('missing/nonfinite proposal scores')
    ranked = np.argsort(scores)[::-1]; best = ranked[0]
    diagnostics = dict(score=scores[best], proposals=len(scores), reason=None,
        bank_sha256=bank_sha256, scores=scores, selected_proposal=int(best))
    if scores[best] < .25:
        return None, dict(diagnostics, reason='cnos_low_object_identity_score')
    if len(ranked) > 1 and scores[ranked[1]] >= scores[best] * .95:
        a, b = masks[best] > .5, masks[ranked[1]] > .5
        if np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)) < .5:
            return None, dict(diagnostics, reason='cnos_ambiguous_object_identity')
    return (masks[best] > .5).astype(np.uint8) * 255, diagnostics


def run_mask_phases(views, scratch, backend, *, reserve, persist):
    """Model-free lifecycle core. Backend owns model mechanics, never truth inputs.

    Proposal masks keep their original floating dtype until DINO has consumed them.
    Only paths, boxes and small diagnostics remain between phases. A phase load
    failure accounts for every affected view; no row is dropped.
    """
    scratch = Path(scratch)
    rows = [dict(view=view, status='pending', failure_reason=None) for view in views]
    peak, cleanup_errors, scratch_created = 0, [], False
    timings = backend.timings

    def fail(row, error):
        row.update(status='unavailable', failure_reason=f'{type(error).__name__}: {error}'[:1024])

    def release(phase):
        try:
            with timings.step(phase + '_release'):
                backend.release(phase)
        except Exception as exc:
            cleanup_errors.append(f'{phase}: {exc}')
            raise

    try:
        reserve(len(views) * PROPOSAL_BYTES_PER_VIEW)
        scratch.mkdir(parents=True, exist_ok=False)
        scratch_created = True
        for phase in ('fastsam', 'dino', 'sam2'):
            try:
                eligible = [r for r in rows if r['status'] == 'pending']
                if not eligible:
                    continue
                with timings.step(phase + '_load', temperature='cold'):
                    backend.load(phase)
                for ordinal, row in enumerate(eligible):
                    view = row['view']; identity = view['sequence_id']
                    proposals = masks = None
                    try:
                        reserve(0)
                        with timings.step(phase + '_infer', view=identity,
                                temperature='cold' if ordinal == 0 else 'warm'):
                            if phase == 'fastsam':
                                proposals = backend.proposals(view)
                                boxes, masks = proposals[:2]
                                if len(masks) == 0:
                                    if len(proposals) != 3 or proposals[2] not in (
                                            {'reason': 'cnos_no_proposals'}, {'reason': 'cnos_no_supported_proposals'}):
                                        raise ValueError('missing observed no-proposal diagnostics')
                                    row['detector'] = dict(proposals[2], scores=[], selected_proposal=None)
                                    raise ValueError('CNOS observed initial association unavailable')
                                if len(masks) > 200 or masks.shape[1:] != (480, 640):
                                    raise ValueError('proposal count/shape exceeds frozen reserve')
                                # No >.5 threshold here: DINO uses floating masks.
                                if masks.dtype.itemsize > 4 or not np.all(np.isfinite(masks)):
                                    raise ValueError('invalid floating proposal intermediate')
                                path = scratch / f'{len(list(scratch.glob("*.npy"))):03d}.npy'
                                with timings.step('proposal_write', view=identity):
                                    np.save(path, masks, allow_pickle=False)
                                row.update(proposals_path=path, boxes=boxes,
                                    proposals_sha256=sha256(path))
                                peak = max(peak, tree_bytes(scratch))
                                proposals = masks = None
                            elif phase == 'dino':
                                masks = np.load(row['proposals_path'], allow_pickle=False)
                                scores, bank_hash = backend.scores(view, row['boxes'], masks)
                                initial, detection = select_proposal(scores, masks, bank_hash)
                                row['detector'] = detection
                                if initial is None:
                                    raise ValueError('CNOS observed initial association unavailable')
                                yy, xx = np.nonzero(initial)
                                row['prompt_box_xyxy'] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                                del masks, initial
                            else:
                                selected = backend.segment(view, row['prompt_box_xyxy'])
                                selected = np.asarray(selected, bool)
                                if selected.shape != (480, 640) or not selected.any():
                                    raise ValueError('SAM2 observed mask unavailable/empty')
                                with timings.step('mask_write', view=identity):
                                    row['output'] = backend.write(view, selected, row)
                                row['status'] = 'available'
                    except Exception as exc:
                        fail(row, exc)
                    finally:
                        proposals = masks = None
                        # Consumed proposal files belong only to this invocation.
                        if phase in ('dino', 'sam2') or row['status'] == 'unavailable':
                            path = row.get('proposals_path')
                            if path is not None and path.exists():
                                path.unlink()
            except Exception as exc:
                for row in rows:
                    if row['status'] == 'pending':
                        fail(row, exc)
                if cleanup_errors:
                    break
            finally:
                release(phase)
    except Exception as exc:
        for row in rows:
            if row['status'] == 'pending':
                fail(row, exc)
    finally:
        for row in rows:
            if row['status'] == 'pending':
                fail(row, RuntimeError('unrun_after_termination'))
            path = row.pop('proposals_path', None)
            if path is not None and path.exists():
                path.unlink()
            row.pop('boxes', None)
        # Never recursive-delete: only consumed files created above are removed.
        if scratch_created and scratch.exists() and not any(scratch.iterdir()):
            scratch.rmdir()
        summary = dict(timings=timings.report(), scratch_peak_bytes=peak,
            prospective_proposal_reserve_bytes=len(views) * PROPOSAL_BYTES_PER_VIEW,
            cleanup_errors=cleanup_errors, scratch_created=scratch_created,
            scratch_cleaned=not scratch.exists(), test_payload_reads=0)
        persist(rows, summary)
    if cleanup_errors:
        raise RuntimeError('mask model release failed: ' + '; '.join(cleanup_errors))
    return rows, summary


class NeuralMaskBackend:
    """Same legacy proposal/crop/descriptor/SAM operations, phase-scoped models."""
    def __init__(self, root, doc, object_id, device, torch, model_cache, *,
                 pin_resolver=None, input_reader=None, mask_writer=None):
        from . import quality_adaptation_larger_prepare as prep
        self.root, self.doc, self.object_id = Path(root), doc, object_id
        self.device, self.torch, self.cache = device, torch, model_cache
        self.timings = Timings(device, torch)
        self.segmentor = self.model = self.predictor = self.references = None
        self.original_grid = None
        self.pin_resolver = pin_resolver or prep.larger.pinned_file
        self.input_reader, self.mask_writer = input_reader, mask_writer
        self.bank = self.pin_resolver(read_json(root / f'cnos-{int(object_id):06d}.json')['bank'])
        self.bank_hash = sha256(self.bank)

    def input(self, view):
        if self.input_reader is not None:
            return self.input_reader(view)
        from . import quality_adaptation_larger_prepare as prep
        from .quality_runner import read_input, read_rgb
        with self.timings.step('input_hash_read', view=view['sequence_id']):
            receipt = read_json(self.root / f"render-{int(self.object_id):06d}-{view['view_index']:02d}.json")
            for key in ('input', 'video', 'mesh'):
                prep.larger.pinned_file(receipt[key])
            bundle = prep._directory(self.root, view) / 'bundle'
            manifest = read_input(bundle)
            return receipt, bundle, manifest, read_rgb(bundle, manifest, 0)

    def load(self, phase):
        from .quality_cnos import load_descriptor
        from .quality_gotrack import upstream_path
        upstream_path()
        if phase == 'fastsam':
            from model.config import FastSAMOpts
            from utils import fastsam_util
            self.segmentor = fastsam_util.FastSAM(FastSAMOpts(
                model_path=str(self.cache / 'checkpoints/FastSAM-x.pt'), conf_threshold=.25,
                iou_threshold=.9, max_det=200, im_width_size=640, verbose=False))
            self.segmentor.set_device(self.torch.device(self.device))
        elif phase == 'dino':
            with np.load(self.bank, allow_pickle=False) as bank:
                expected = next(a['mesh']['sha256'] for a in self.doc['assets'] if a['object_id'] == self.object_id)
                if str(bank['asset_sha256']) != expected:
                    raise ValueError('Wrong CNOS object bank')
                self.references = self.torch.from_numpy(bank['descriptors'].copy()).to(self.device)
            self.model = load_descriptor(self.device)
        else:
            import sys
            sys.path.insert(0, str(self.cache / 'sources/sam2'))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from .quality_sam2 import deterministic_position_grid
            from sam2.modeling.position_encoding import PositionEmbeddingRandom
            self.original_grid = PositionEmbeddingRandom.forward
            PositionEmbeddingRandom.forward = deterministic_position_grid
            self.model = build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',
                str(self.cache / 'checkpoints/sam2.1_hiera_base_plus.pt'), device=self.device, apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(self.model)

    def release(self, phase):
        from . import quality_adaptation_larger as larger
        self.segmentor = self.references = self.predictor = self.model = None
        if self.original_grid is not None:
            from sam2.modeling.position_encoding import PositionEmbeddingRandom
            PositionEmbeddingRandom.forward = self.original_grid
            self.original_grid = None
        gc.collect()
        larger.cleanup_device(self.device, self.torch)

    def proposals(self, view):
        from utils import im_util
        _, _, _, rgb = self.input(view)
        proposals = self.segmentor.generate_detection_proposals(rgb[..., ::-1].copy())
        detection = {'reason': 'cnos_no_proposals'} if not len(proposals) else None
        if len(proposals):
            keep = im_util.filter_noisy_detections(proposals.boxes, proposals.masks,
                min_box_size=.05, min_mask_size=3e-4, width=rgb.shape[1], height=rgb.shape[0])
            proposals = proposals[keep]
            if not len(proposals):
                detection = {'reason': 'cnos_no_supported_proposals'}
        return proposals.boxes.cpu().numpy(), proposals.masks.cpu().numpy(), detection

    def scores(self, view, boxes, masks):
        from .quality_cnos import camera_for
        from utils import crop_generation, im_util
        _, _, manifest, rgb = self.input(view)
        # One view at a time, identical floating masks and original proposal order.
        crops = crop_generation.batch_cropping_from_bbox(source_images=[rgb] * len(masks),
            source_cameras=[camera_for(manifest)] * len(masks), source_xyxy_bboxes=boxes,
            source_masks=masks[..., None], crop_size=(280, 280), crop_rel_pad=.05)
        scores = []
        with self.torch.inference_mode():
            refs = self.torch.nn.functional.normalize(self.references, dim=1)
            for rgb_crop, mask in zip(crops.rgbs, crops.masks):
                q = self.model.forward_features(im_util.im_normalize((rgb_crop * mask)[None]).to(self.device))['x_norm_clstoken']
                similarities = self.torch.nn.functional.normalize(q, dim=1) @ refs.T
                scores.append(float(similarities.topk(5, dim=1).values.mean()))
        return scores, self.bank_hash

    def segment(self, view, box):
        _, _, _, rgb = self.input(view)
        with self.torch.inference_mode():
            # set_image resets predictor image state for every view, including A->B->A.
            self.predictor.set_image(rgb)
            predictions, scores, _ = self.predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=True)
        return predictions[int(np.argmax(scores))]

    def write(self, view, selected, row):
        if self.mask_writer is not None:
            return self.mask_writer(view, selected, row)
        from . import quality_adaptation_larger_prepare as prep
        from .vision import cv2
        receipt, _, _, _ = self.input(view)
        output = prep._directory(self.root, view) / 'observed-masks'
        output.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(output / '0.png'), selected.astype(np.uint8) * 255):
            raise OSError('SAM2 mask write failed')
        mask_pin = prep.file_pin(output / '0.png')
        producer = dict(schema='gotrack_larger_observed_sam_mask_v1', frame_id=0,
            input_sha256=receipt['input']['sha256'], mesh_sha256=receipt['mesh']['sha256'],
            video_sha256=receipt['video']['sha256'], source_contract_sha256=self.doc['source_contract']['sha256'],
            producer_sha256=sha256(Path(__file__)), device=self.device, preparation_variant=VARIANT,
            deterministic_algorithms=self.torch.are_deterministic_algorithms_enabled(),
            tf32=False if self.device == 'cuda' else None, cnos_bank=prep.file_pin(self.bank),
            automatic=True, mask_origin='observed_CNOS_box_then_SAM2_image_predictor',
            prompt_box_xyxy=row['prompt_box_xyxy'], detector=row['detector'], mask=mask_pin,
            proposal_intermediate_sha256=row['proposals_sha256'], test_payload_reads=0, annotation_inputs=[])
        write_json(output, 'results.json', producer)
        return dict(mask_index=prep.file_pin(output / 'results.json'), mask=mask_pin)


def masks_phased(root, object_id, view_index=None, *, cloud_execution=False, device='cpu'):
    cloud_guard(cloud_execution)
    from . import quality_adaptation_larger_prepare as prep
    larger = prep.larger; root = Path(root)
    doc = prep.frozen(root, device, VARIANT)
    views = prep._select(doc, object_id, view_index)
    for view in views:
        if (root / f"mask-{int(object_id):06d}-{view['view_index']:02d}.json").exists():
            raise FileExistsError('mask attempt receipt exists; preserve failure')
    cache = prep._existing_models()
    torch = larger.configure_torch(device)
    backend = NeuralMaskBackend(root, doc, object_id, device, torch, cache)
    def persist(rows, summary):
        for row in rows:
            result = dict(status=row['status'], device=device, preparation_variant=VARIANT,
                failure_reason=row['failure_reason'], diagnostics={k: row[k] for k in
                    ('detector', 'prompt_box_xyxy', 'proposals_sha256') if k in row})
            result.update(row.get('output', {}))
            write_json(root, f"mask-{int(object_id):06d}-{row['view']['view_index']:02d}.json", result)
        write_json(root, f'masks-amortized-{int(object_id):06d}-{views[0]["view_index"]:02d}.json', summary)
    with larger.stage_lock():
        rows, summary = run_mask_phases(views, root / f'.amortized-v1-{int(object_id):06d}-{views[0]["view_index"]:02d}',
            backend, reserve=lambda amount: larger.resources(root, reserve=amount), persist=persist)
    return [dict(status=r['status'], failure_reason=r['failure_reason'], **r.get('output', {})) for r in rows]


def dispose_search(search):
    """Explicitly delete isolated FAISS indices, suppressing no cleanup failures."""
    from . import quality_neighbors as neighbors
    failures = []
    for index in [getattr(search, 'visual_words', None), *getattr(search, 'indices', [])]:
        if index is None or getattr(index, 'index_id', None) is None:
            continue
        identity = index.index_id
        try:
            neighbors.client().request(('delete', identity))
            index.index_id = None  # Prevent destructor double-delete/reuse.
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError('explicit FAISS disposal failed: ' + '; '.join(failures))


class PreparedFoundPoseSearch:
    """One authenticated object bank and its immutable search representation."""
    def __init__(self, *, bank_pin, asset_sha256, object_id, variant, timings, loader,
                 recovery_class=None, dispose=dispose_search, pin_resolver=None):
        from . import quality_adaptation_larger as larger
        larger.preparation_variant(dict(preparation_variant=variant), VARIANT)
        with timings.step('bank_hash', temperature='cold'):
            path = (pin_resolver or larger.pinned_file)(bank_pin)
        with timings.step('bank_load', temperature='cold'):
            self.bank = loader(path)
        if self.bank.get('template_count') != 798 or self.bank.get('asset_sha256') != asset_sha256:
            raise ValueError('standard 798-template authenticated object bank required')
        self.identity = (object_id, bank_pin['sha256'], asset_sha256, variant)
        self.timings, self.owner, self.dispose = timings, None, dispose
        if recovery_class is None:
            from .quality_foundpose import FoundPoseRecovery
            recovery_class = FoundPoseRecovery
        self.recovery_class = recovery_class

    def bind(self, refiner, *, object_id, bank_pin, asset_sha256, variant, time_policy):
        if self.identity != (object_id, bank_pin['sha256'], asset_sha256, variant):
            raise ValueError('prepared search object/bank/asset/variant binding differs')
        if refiner.renderer.asset_sha256 != asset_sha256:
            raise ValueError('prepared search renderer asset differs')
        if self.owner is None:
            # Capture each constructed KNN so a partial constructor can be cleaned.
            from utils import knn_util
            from . import quality_neighbors as neighbors
            original, original_knn, created = neighbors.IsolatedKNN, knn_util.KNN, []
            def tracked(*args, **kwargs):
                index = original(*args, **kwargs); created.append(index); return index
            neighbors.IsolatedKNN = tracked
            knn_util.KNN = tracked
            try:
                with self.timings.step('search_indices', temperature='cold'):
                    self.owner = self.recovery_class(refiner, self.bank, time_policy=time_policy)
            except BaseException:
                from types import SimpleNamespace
                self.dispose(SimpleNamespace(indices=created))
                raise
            finally:
                neighbors.IsolatedKNN = original
                knn_util.KNN = original_knn
            # Detach view state immediately. The next object owns its own indices.
            self.owner.refiner = self.owner.backbone = None
        adapter = object.__new__(self.recovery_class)
        adapter.refiner, adapter.backbone = refiner, refiner.network.backbone
        adapter.time_policy = time_policy
        adapter.clock_modes_supported = frozenset((time_policy.mode,))
        adapter.memory = None
        for name in ('projector', 'repre', 'visual_words', 'indices'):
            setattr(adapter, name, getattr(self.owner, name))
        return adapter

    def close(self):
        if self.owner is not None:
            with self.timings.step('search_release'):
                self.dispose(self.owner)
            self.owner = None
        self.bank = None


class ObjectSearchCache:
    """Keeps at most one object alive; close precedes every object transition."""
    def __init__(self, factory=PreparedFoundPoseSearch):
        self.factory, self.current, self.identity = factory, None, None

    def get(self, **kwargs):
        identity = (kwargs['object_id'], kwargs['bank_pin']['sha256'], kwargs['asset_sha256'], kwargs['variant'])
        if identity != self.identity:
            self.close()
            self.current = self.factory(**kwargs)
            self.identity = identity
        return self.current

    def close(self):
        if self.current is not None:
            self.current.close()
        self.current = self.identity = None


def masks_aba_diagnostic(root, object_id, view_a, view_b, output, *, cloud_execution=False, device='cpu'):
    """Run actual A(image A)->B(image B)->A on each held phase model.

    Diagnostic case IDs and output directories are separate from the immutable
    training inventory. No existing mask, receipt or selected frame is replaced.
    First A is the cold predictor call; third A tests image-state reset.
    """
    cloud_guard(cloud_execution)
    from . import quality_adaptation_larger_prepare as prep
    from .quality_adaptation_larger_parity import _pixel_pin
    larger = prep.larger; root, output = Path(root), larger._safe_output(output)
    doc = prep.frozen(root, device, VARIANT)
    a, b = prep._select(doc, object_id, view_a)[0], prep._select(doc, object_id, view_b)[0]
    if view_a == view_b:
        raise ValueError('ABA diagnostic requires two distinct predeclared views')
    if output.exists():
        raise FileExistsError('ABA diagnostic output exists; retain previous trial')
    larger.resources(output, reserve=3 * PROPOSAL_BYTES_PER_VIEW)
    output.mkdir(parents=True)
    cases = [dict(view, sequence_id=f'diagnostic-mask-aba-{index}', original_sequence_id=view['sequence_id'])
        for index, view in enumerate((a, b, a))]
    torch = larger.configure_torch(device)
    class DiagnosticBackend(NeuralMaskBackend):
        def write(self, view, selected, row):
            receipt, _, _, _ = self.input(view)
            index = int(view['sequence_id'].rsplit('-', 1)[1])
            path = output / f'mask-{index}.npy'
            np.save(path, selected, allow_pickle=False)
            return dict(mask=prep.file_pin(path), pixels=_pixel_pin(selected),
                input_pins={key: {k: receipt[key][k] for k in ('bytes', 'sha256')}
                    for key in ('input', 'video', 'mesh')})
    backend = DiagnosticBackend(root, doc, object_id, device, torch, prep._existing_models())
    def persist(rows, summary):
        write_json(output, 'inventory.json', dict(rows=rows, summary=summary,
            diagnostic_only=True, training_inventory_unchanged=True, annotation_inputs=[], test_payload_reads=0))
    with larger.stage_lock():
        rows, summary = run_mask_phases(cases, output / '.proposals', backend,
            reserve=lambda amount: larger.resources(output, reserve=amount), persist=persist)
    if any(row['status'] != 'available' for row in rows):
        raise ValueError('ABA mask diagnostic unavailable; inventory retained')
    first, repeat = rows[0], rows[2]
    if (first['output']['pixels'] != repeat['output']['pixels'] or
            first['output']['input_pins'] != repeat['output']['input_pins'] or
            first['detector'] != repeat['detector'] or first['prompt_box_xyxy'] != repeat['prompt_box_xyxy']):
        raise ValueError('ABA mask/prompt/association state differs')
    result = dict(passed=True, diagnostic_only=True, cases=3, cold_a_vs_repeat_a=True,
        one_model_per_phase=True, test_payload_reads=0, annotation_inputs=[])
    write_json(output, 'parity.json', result)
    return result


def recovery_aba_diagnostic(plan_path, plan_sha256, object_id, view_a, view_b, output, *, cloud_execution=False, device='cpu'):
    """Actual same-search A->B->A with fresh renderer/refiner and per-view RNG."""
    cloud_guard(cloud_execution)
    import random
    from . import quality_adaptation_larger as larger
    from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
    from .quality_contract import Frame
    from .quality_runner import read_rgb
    from .quality_time import PHYSICAL
    from .vision import cv2
    _, plan = larger.read_plan(plan_path, plan_sha256)
    larger.preparation_variant(plan, VARIANT)
    inputs = larger.preflight(plan)  # Validates LAST row before any payload.
    selected = []
    for index in (view_a, view_b, view_a):
        rows = [r for r in plan['rows'] if r['object_id'] == object_id and r['view_index'] == index]
        if len(rows) != 1 or rows[0]['observation_status'] != 'available':
            raise ValueError('ABA diagnostic requires available predeclared views')
        selected.append(rows[0])
    if view_a == view_b:
        raise ValueError('ABA diagnostic requires distinct A/B views')
    output = larger._safe_output(output)
    if output.exists():
        raise FileExistsError('recovery ABA output exists')
    larger.resources(output); output.mkdir(parents=True)
    torch = larger.configure_torch(device, deterministic=True)
    timings = Timings(device, torch); cache = ObjectSearchCache()
    records = [dict(id=f'diagnostic-recovery-aba-{ordinal}', original_id=physical['id'],
        status='unavailable', failure_reason='not_processed', poses=None) for ordinal, physical in enumerate(selected)]
    network, failure = None, None
    with larger.stage_lock():
        try:
            with timings.step('network_load', temperature='cold'):
                network = load_network(device).eval()
            for ordinal, physical in enumerate(selected):
                renderer = backend = refiner = None
                record = records[ordinal]
                try:
                    larger.resources(output)
                    bundle, manifest, context = inputs[physical['id']]
                    rgb = read_rgb(bundle, manifest, physical['frame_id'])
                    mask = cv2.imread(str(larger.pinned_file(physical['mask'])), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                        raise ValueError('ABA observed mask unavailable')
                    frame = Frame(0, rgb, context.effective_intrinsics, timestamp_s=0., clock_mode=PHYSICAL)
                    random.seed(0); np.random.seed(0); torch.manual_seed(0)
                    with timings.step('renderer_setup', view=record['id']):
                        renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                            unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                    refiner = GoTrackRefiner(network, renderer, manifest['object_id'], device, inference_geometry=larger.inference.PROFILE)
                    binding = dict(object_id=object_id, bank_pin=physical['bank'], asset_sha256=physical['mesh']['sha256'], variant=VARIANT)
                    search = cache.get(**binding, timings=timings,
                        loader=lambda path: torch.load(path, map_location='cpu', weights_only=False))
                    backend = search.bind(refiner, **binding, time_policy=manifest['_timing']['time_policy'])
                    with timings.step('recovery', view=record['id']):
                        poses = backend.recover(frame, mask, top_k=5)
                    if not poses or not np.all(np.isfinite(poses)):
                        raise ValueError('ABA candidates unavailable/nonfinite')
                    record.update(status='available', failure_reason=None, poses=[np.asarray(pose).tolist() for pose in poses],
                        input_pins={key: {k: physical[key][k] for k in ('bytes', 'sha256')}
                            for key in ('input', 'video', 'mesh', 'mask', 'bank')})
                except Exception as exc:
                    record['failure_reason'] = f'{type(exc).__name__}: {exc}'[:1024]
                finally:
                    backend = refiner = None
                    if renderer is not None:
                        renderer.close()
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'[:1024]
        finally:
            try:
                cache.close()
            except Exception as exc:
                failure = f'search cleanup failed: {exc}'[:1024]
            try:
                if network is not None:
                    network.cpu()
            except Exception as exc:
                failure = f'network cleanup failed: {exc}'[:1024]
            finally:
                network = None
                try:
                    larger.cleanup_device(device, torch)
                except Exception as exc:
                    failure = f'device cleanup failed: {exc}'[:1024]
                write_json(output, 'inventory.json', dict(rows=records, fatal_error=failure,
                    timings=timings.report(), diagnostic_only=True, test_payload_reads=0, annotation_inputs=[]))
    if failure or len(records) != 3 or any(r['status'] != 'available' for r in records):
        raise ValueError('recovery ABA incomplete/unavailable; inventory retained')
    from .quality_adaptation_larger_parity import _equal_array
    _equal_array(records[0]['poses'], records[2]['poses'], 'ABA ordered recovery candidates')
    if records[0]['input_pins'] != records[2]['input_pins']:
        raise ValueError('ABA recovery input pins differ')
    result = dict(passed=True, cases=3, diagnostic_only=True, cold_a_vs_repeat_a=True,
        one_object_search=True, fresh_renderer_refiner=True, test_payload_reads=0, annotation_inputs=[])
    write_json(output, 'parity.json', result)
    return result
