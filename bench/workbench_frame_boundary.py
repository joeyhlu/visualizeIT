"""Additive, opt-in portable frame boundary: initialize -> process(timestamped frame) -> reset -> close.

One public lifecycle for a finite recorded source today and, later, a reviewed camera source. It
wraps the EXISTING validated records (``ClipSpec``/``RunSpec``/``FrameSpec``/``SurfacePackage``/
``FrameResult``) and the EXISTING real session classes (``NeuralSession`` and ``RelativeTracker``
plus ``render_marker``). It adds no metric math and no readiness signal.

The default CLI pipeline is NOT migrated. ``bench/workbench_cli.py`` and its admitted, cached
stages are unchanged and do not call this module. Nothing here is cached.

What is checked, and when:

* ``initialize`` re-derives a private canonical context from the validated clip, run and binding,
  and refuses any ``FrameContextV1`` whose fields (schema, K, dimensions, mode, order, hashes,
  prefix) differ. It fingerprints the caller's context and refuses to process or reset after
  that object is mutated.
* Neural roles need ``RunSpec.primary`` to equal the pinned primary identity. The complete metric
  bundle (validated SE(3) seed and its hash, object ID, GLB path and hash, seed label, origin,
  family, provenance) and seed target are bound. They are compared with the backend before open
  and before and after every process and reset, together with the session's run, clip, role and
  identity and the ENTIRE admission record. Any drift closes the session before estimation; there
  is no implicit reset or re-arm.
* Relative geometry comes only from a ``RelativePackageProof``: the surface record and component
  bytes. The loader's size, hash, dtype, finite, mesh, count and support checks are repeated,
  and private read-only arrays are decoded from the proven bytes. Packages must be object-unit,
  frozen within the reviewed onboarding prefix before scoring, and their generated artwork must
  match the package and marker artwork SHA.
* Every ``FrameInputV1`` is revalidated at ``process``: schema, full ``FrameSpec``, finite
  non-negative decoder time, identity formats, bounded strict metadata without reserved keys, and
  clock rules. All of this happens before the cursor moves or any backend call.
* Initialization failures of ANY kind close the transferred adapter and record cleanup errors.

Live neural camera sources are UNSUPPORTED. A synthetic camera-style source exercises the software
API for the relative role only and never inherits a neural family admission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import workbench_contract as wc

BOUNDARY_VERSION = "2"
SCHEMA_CONTEXT = "workbench_frame_context_v1"
SCHEMA_INPUT = "workbench_frame_input_v1"
SCHEMA_OUTPUT = "workbench_frame_output_v1"
SOURCE_MODES = ("recorded_finite", "camera_style_synthetic")
NEURAL_ROLES = ("primary_neural", "frozen_baseline")
RELATIVE_ROLE = "relative_geometry"
RESET_REASONS = ("replay_restart",)
MEDIA_TIME_TOL_S = 1e-3                   # same tolerance as FrameReader's decoder PTS check
MAX_METADATA_KEYS = 32
MAX_METADATA_BYTES = 16 << 10
MAX_REASON_CHARS = 500
MAX_SURFACE_JSON_BYTES = 1_000_000
RESERVED_METADATA_KEYS = frozenset({
    "K", "intrinsics", "width", "height", "frame_id", "source_frame_index", "timestamp_s", "media_time_s",
    "image_sha256", "mask_sha256", "mask_frame_id", "pose", "binding", "role", "run_id", "clip_id", "epoch",
    "seed_target_frame_id", "family", "admission"})
NEURAL_BINDING_KEYS = frozenset({"kind", "family", "pose_units", "metric_geometry", "bundle", "seed_frame_id",
                                 "seed_target_frame_id", "warmup_frame_ids", "automatic_reacquisition",
                                 "admission_inherited_by_synthetic_sources"})
BUNDLE_RECORD_KEYS = frozenset({"family", "glb_path", "glb_sha256", "obj_id", "principal_point_origin", "seed_label",
                                "seed_frame_id", "seed_pose_m", "seed_pose_sha256", "metric_provenance"})
RELATIVE_BINDING_KEYS = frozenset({
    "kind", "package_id", "package_record_sha256", "surface_json_sha256", "proof_source", "clip_id", "status",
    "frozen", "package_units", "metric_scale_available", "pose_units", "metric_geometry", "frozen_after_frame_id",
    "prefix_frame_ids", "camera_frame_ids", "artwork_kind", "package_artwork", "generated_artwork_sha256",
    "marker_artwork_sha256", "components", "tracker_version"})
METRIC_PROVENANCE = ("measured_metric_cad", "dataset_metric_cad")
BUILTIN_ARTWORK = ("checkerboard", "directional")


class BoundaryError(ValueError):
    """Base class for boundary refusals."""


class BoundaryRejected(BoundaryError):
    """Protocol/binding violation; nothing was processed and the cursor did not move."""


class LiveNeuralUnsupported(BoundaryError):
    """Neural roles accept only finite recorded manifests until a reviewed camera contract exists."""


class BoundaryStateError(RuntimeError):
    """Lifecycle misuse: process/reset before initialize or after close/failure."""


class BoundaryFailure(RuntimeError):
    """The adapter raised, drifted or produced an invalid row; its resources were closed."""


class ContextMutated(BoundaryRejected, BoundaryFailure):
    """The caller's context object changed after initialize; the adapter was closed."""


# --------------------------------------------------------------------------
# helpers


def _plain(value: Any) -> Any:
    """Convert numpy scalars/arrays, bytes and tuples into plain JSON values (never mutates the input)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _frozen_json(value: Any, what: str) -> bytes:
    try:
        return wc.canonical_json(wc.loads_strict(wc.canonical_json(_plain(value))))
    except (TypeError, ValueError) as exc:
        raise BoundaryRejected(f"{what} is not bounded strict JSON ({exc})") from exc


def _hex(value: Any) -> bool:
    return isinstance(value, str) and wc.SHA_RE.fullmatch(value) is not None


def _readonly_copy(array: Any, what: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise BoundaryRejected(f"{what} must be a numpy array")
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _safe_close(obj: Any) -> list[str]:
    close = getattr(obj, "close", None)
    if not callable(close):
        return []
    try:
        close()
        return []
    except Exception as exc:  # noqa: BLE001 - cleanup errors are reported, never hidden
        return [f"{type(obj).__name__}.close: {type(exc).__name__}: {exc}"[:500]]


def _check_primary(run: wc.RunSpec) -> None:
    """Runs that request a neural role (including a baseline comparator) must name the pinned primary."""
    if not set(run.roles) & set(NEURAL_ROLES):
        return
    from . import workbench_neural as wn
    pinned = {k: wn.PRIMARY[k] for k in ("artifact_id", "export_sha256", "export_bytes")}
    if dict(run.primary) != pinned:
        raise BoundaryRejected("RunSpec.primary differs from the pinned primary identity (no misleading model metadata)")


def neural_seed_plan(clip: wc.ClipSpec, run: wc.RunSpec, seed_frame_id: str) -> tuple[str, tuple[str, ...]]:
    """Seed target and causal warm-up exactly as the legacy pipeline derives them."""
    order = [f.frame_id for f in clip.frames]
    if seed_frame_id not in order:
        raise BoundaryRejected(f"seed frame {seed_frame_id} is not a clip frame")
    seed_pos = order.index(seed_frame_id)
    if seed_pos + 1 >= len(order):
        raise BoundaryRejected("no frame follows the seed frame")
    first = order.index(run.scheduled_frame_ids[0])
    if first <= seed_pos:
        raise BoundaryRejected("scheduled frames start at or before the seed frame")
    if seed_pos > max(order.index(k) for k in run.onboarding.keyframe_ids):
        raise BoundaryRejected("seed frame lies after the onboarding prefix")
    return order[seed_pos + 1], tuple(order[seed_pos + 1:first])


def _seed_label_ok(label: Any) -> bool:
    """The existing MetricBundle rule: a seed pose is labelled controlled_init:/onboarding: only."""
    from . import workbench_neural as wn
    return isinstance(label, str) and label.startswith(tuple(wn.SEED_PREFIXES)) and bool(label.split(":", 1)[1].strip())


def bundle_record(bundle: Any) -> dict:
    """Complete canonical record of a ``MetricBundle`` (validated SE(3) seed and its hash included)."""
    from . import workbench_neural as wn
    if not isinstance(bundle, wn.MetricBundle):
        raise BoundaryRejected("bundle must be a workbench_neural.MetricBundle")
    obj_id = bundle.obj_id
    if isinstance(obj_id, (bool, np.bool_)) or not isinstance(obj_id, (int, np.integer)):
        raise BoundaryRejected("bundle obj_id must be an integer")
    if bundle.metric_provenance not in METRIC_PROVENANCE:
        raise BoundaryRejected("relative/object-unit reconstructions never become neural metric geometry")
    if not _hex(bundle.glb_sha256) or bundle.principal_point_origin not in ("integer_centers", "corner"):
        raise BoundaryRejected("bundle GLB hash/principal point origin invalid")
    seed = None
    if bundle.seed_pose_m is not None:
        try:
            seed = [list(r) for r in wc._pose(np.asarray(bundle.seed_pose_m, dtype=np.float64).tolist(), "seed_pose_m")]
        except (wc.ContractError, TypeError, ValueError) as exc:
            raise BoundaryRejected(f"bundle seed pose is not a valid SE(3) matrix ({exc})") from exc
    for name in ("family", "seed_label", "seed_frame_id"):
        value = getattr(bundle, name)
        if value is not None and not isinstance(value, str):
            raise BoundaryRejected(f"bundle {name} must be a string")
    if seed is not None and not _seed_label_ok(bundle.seed_label):
        raise BoundaryRejected("a seed pose must be labelled controlled_init:/onboarding: (never evaluator or unlabelled data)")
    return {"family": bundle.family, "glb_path": Path(bundle.glb_path).as_posix(), "glb_sha256": bundle.glb_sha256,
            "obj_id": int(obj_id), "principal_point_origin": bundle.principal_point_origin,
            "seed_label": bundle.seed_label, "seed_frame_id": bundle.seed_frame_id, "seed_pose_m": seed,
            "seed_pose_sha256": None if seed is None else wc.canonical_sha256(seed),
            "metric_provenance": bundle.metric_provenance}


def neural_binding(clip: wc.ClipSpec, run: wc.RunSpec, *, family: str, bundle: Any = None) -> dict:
    """Declared neural binding. Verified against the real session/backend at open and every frame."""
    if not isinstance(family, str) or not family.strip():
        raise BoundaryRejected("neural binding requires an explicit family")
    if bundle is None:
        record, seed, target, warmup = None, None, None, ()
    else:
        record = bundle_record(bundle)
        if record["family"] != family:
            raise BoundaryRejected("bundle family differs from the declared family")
        seed = record["seed_frame_id"]
        target, warmup = neural_seed_plan(clip, run, seed)
    return {"kind": "neural", "family": family, "pose_units": "metric", "metric_geometry": True,
            "bundle": record, "seed_frame_id": seed, "seed_target_frame_id": target,
            "warmup_frame_ids": list(warmup), "automatic_reacquisition": False,
            "admission_inherited_by_synthetic_sources": False}


# --------------------------------------------------------------------------
# relative package proof (verifiable bytes, never caller arrays)


class RelativePackageProof:
    """Surface record bytes plus raw component bytes; the ONLY accepted relative geometry input.

    ``from_package_dir`` reads an explicit package directory with the existing loader's containment,
    size and aggregate rules. ``from_memory`` accepts manufactured bytes (labelled
    ``manufactured_memory_bytes``; not a real reconstruction). ``verify`` repeats the loader's
    hash/size/dtype/finite/mesh/count/support checks and returns private read-only arrays."""

    __slots__ = ("_surface_json", "_components", "_source")

    def __init__(self, surface_json: bytes, components: Mapping[str, bytes], source: str):
        from . import workbench_reconstruction as wr
        if not isinstance(surface_json, bytes) or len(surface_json) > MAX_SURFACE_JSON_BYTES:
            raise BoundaryRejected("surface record must be bytes of at most 1 MB")
        if (not isinstance(components, Mapping) or set(components) - set(wr.PACKAGE_FILES)
                or any(not isinstance(v, bytes) for v in components.values())):
            raise BoundaryRejected("components must map package component keys to raw bytes")
        if source not in ("package_dir", "manufactured_memory_bytes"):
            raise BoundaryRejected("unknown proof source")
        object.__setattr__(self, "_surface_json", bytes(surface_json))
        object.__setattr__(self, "_components", tuple(sorted((k, bytes(v)) for k, v in components.items())))
        object.__setattr__(self, "_source", source)

    def __setattr__(self, name, value):
        raise AttributeError("RelativePackageProof is immutable")

    @classmethod
    def from_memory(cls, surface_json: bytes, components: Mapping[str, bytes]) -> "RelativePackageProof":
        return cls(surface_json, components, "manufactured_memory_bytes")

    @classmethod
    def from_package_dir(cls, directory: str | Path) -> "RelativePackageProof":
        from . import workbench_reconstruction as wr
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise BoundaryRejected("package directory missing or a symlink")
        base = directory.resolve()

        def component(relative: str) -> Path:
            wc._relative_path(relative, "package component")
            if "/" in relative:
                raise BoundaryRejected("package components must be top-level files")
            p = directory / relative
            if p.is_symlink() or not p.is_file() or p.resolve().parent != base:
                raise BoundaryRejected(f"package file {relative} missing, a symlink or outside the package")
            return p
        surface = component("surface.json")
        if surface.stat().st_size > MAX_SURFACE_JSON_BYTES:
            raise BoundaryRejected("surface.json too large")
        raw = surface.read_bytes()
        pkg = wc.SurfacePackage.from_dict(wc.loads_strict(raw))
        refs = {key: getattr(pkg, key) for key in wr.PACKAGE_FILES if getattr(pkg, key) is not None}
        paths, total = {}, len(raw)
        for key, ref in refs.items():
            p = component(ref.path)
            if p.stat().st_size != ref.bytes:
                raise BoundaryRejected(f"package file {ref.path} size differs")
            total += ref.bytes
            paths[key] = p
        if total > wc.MAX_PACKAGE_BYTES:
            raise BoundaryRejected("package exceeds 20 MB before reading")
        return cls(raw, {key: p.read_bytes() for key, p in paths.items()}, "package_dir")

    @property
    def source(self) -> str:
        return self._source

    @property
    def surface_json_sha256(self) -> str:
        return hashlib.sha256(self._surface_json).hexdigest()

    def verify(self) -> tuple[wc.SurfacePackage, dict[str, np.ndarray], dict[str, dict]]:
        from . import workbench_geometry as wg
        from . import workbench_reconstruction as wr
        try:
            pkg = wc.SurfacePackage.from_dict(wc.loads_strict(self._surface_json))
        except (wc.ContractError, ValueError) as exc:
            raise BoundaryRejected(f"surface record contract: {exc}") from exc
        components = dict(self._components)
        refs = {key: getattr(pkg, key) for key in wr.PACKAGE_FILES if getattr(pkg, key) is not None}
        if set(refs) != set(components):
            raise BoundaryRejected("proof components differ from the package record's components")
        if len(self._surface_json) + sum(len(b) for b in components.values()) > wc.MAX_PACKAGE_BYTES:
            raise BoundaryRejected("package exceeds 20 MB")
        arrays, digests = {}, {}
        for key, ref in sorted(refs.items()):
            data = components[key]
            if len(data) != ref.bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
                raise BoundaryRejected(f"package component {key} size/hash differs from its FileRef")
            _name, dtype, cols = wr.PACKAGE_FILES[key]
            try:
                arr = np.load(io.BytesIO(data), allow_pickle=False)
            except (ValueError, OSError, EOFError) as exc:
                raise BoundaryRejected(f"package component {key} is not a plain .npy array") from exc
            if arr.dtype != dtype or (cols is not None and (arr.ndim != 2 or arr.shape[1] != cols)):
                raise BoundaryRejected(f"package array {key} dtype/shape invalid")
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                raise BoundaryRejected(f"package array {key} non-finite")
            arr = np.array(arr, copy=True)
            arr.setflags(write=False)
            arrays[key] = arr
            digests[key] = {"file_sha256": ref.sha256, "bytes": ref.bytes, "array_sha256": wg.array_sha256(arr),
                            "dtype": arr.dtype.str, "shape": list(arr.shape)}
        if pkg.status == "available":
            try:
                wg.validate_mesh(arrays["vertices"], arrays["faces"])
            except (wg.GeometryError, KeyError) as exc:
                raise BoundaryRejected(f"package mesh invalid ({exc})") from exc
            if len(arrays["vertices"]) != pkg.vertex_count or len(arrays["faces"]) != pkg.face_count:
                raise BoundaryRejected("package counts differ from arrays")
            if arrays["face_support_views"].shape not in ((pkg.face_count,), (pkg.face_count, 1)):
                raise BoundaryRejected("face support must have one count per face")
            if "descriptor_points" not in arrays or "descriptor_values" not in arrays:
                raise BoundaryRejected("available package lacks its descriptor bank")
        return pkg, arrays, digests


def _relative_binding_and_arrays(proof: Any, artwork_kind: str) -> tuple[dict, wc.SurfacePackage, dict]:
    from . import workbench_geometry as wg
    from .workbench_cli import TRACKER_VERSION
    if not isinstance(proof, RelativePackageProof):
        raise BoundaryRejected("relative geometry requires a RelativePackageProof (verified component bytes), "
                               "not caller arrays and a caller-declared hash")
    pkg, arrays, digests = proof.verify()
    generated = None
    if pkg.artwork is not None:
        if artwork_kind not in BUILTIN_ARTWORK:
            raise BoundaryRejected("imported_artwork_unavailable: only fixed built-in artwork can be regenerated")
        generated = wg.array_sha256(wg.builtin_artwork(artwork_kind, pkg.artwork["edge_px"]))
    binding = {"kind": "relative", "package_id": pkg.package_id, "package_record_sha256": wc.canonical_sha256(pkg.to_dict()),
               "surface_json_sha256": proof.surface_json_sha256, "proof_source": proof.source, "clip_id": pkg.clip_id,
               "status": pkg.status, "frozen": pkg.frozen, "package_units": pkg.units,
               "metric_scale_available": pkg.metric_scale_available, "pose_units": "object_unit", "metric_geometry": False,
               "frozen_after_frame_id": pkg.frozen_after_frame_id, "prefix_frame_ids": list(pkg.prefix_frame_ids),
               "camera_frame_ids": [c.frame_id for c in pkg.cameras], "artwork_kind": artwork_kind,
               "package_artwork": pkg.artwork, "generated_artwork_sha256": generated,
               "marker_artwork_sha256": None if pkg.marker is None else pkg.marker.artwork_sha256,
               "components": digests, "tracker_version": TRACKER_VERSION}
    return binding, pkg, arrays


def relative_binding(proof: RelativePackageProof, *, artwork_kind: str) -> dict:
    """Declared frozen-surface binding from VERIFIED package bytes (always object units)."""
    return _relative_binding_and_arrays(proof, artwork_kind)[0]


# --------------------------------------------------------------------------
# context


@dataclass(frozen=True)
class FrameContextV1:
    """Validated per-role context. Build with ``FrameContextV1.build``; ``initialize`` re-derives it."""
    clip: wc.ClipSpec
    run: wc.RunSpec
    role: str
    source_mode: str
    warmup_frame_ids: tuple[str, ...]
    scheduled_frame_ids: tuple[str, ...]
    permitted_order: tuple[str, ...]
    K: tuple[float, ...]
    width: int
    height: int
    binding_json: bytes
    binding_sha256: str
    clip_sha256: str
    run_sha256: str
    prefix_attestation_json: bytes
    schema: str = SCHEMA_CONTEXT

    @classmethod
    def build(cls, clip: wc.ClipSpec, run: wc.RunSpec, role: str, binding: Mapping[str, Any], *,
              source_mode: str = "recorded_finite", warmup_frame_ids: Sequence[str] | None = None) -> "FrameContextV1":
        if not isinstance(clip, wc.ClipSpec) or not isinstance(run, wc.RunSpec):
            raise BoundaryRejected("context requires a validated ClipSpec and RunSpec")
        try:   # private validated copies: later caller mutation of nested dicts cannot reach the context
            clip = wc.ClipSpec.from_dict(clip.to_dict())
            run = wc.RunSpec.from_dict(run.to_dict())
            attestation = run.validate_against(clip)
        except (wc.ContractError, KeyError, AttributeError, TypeError) as exc:
            raise BoundaryRejected(f"clip/run contract: {exc}") from exc
        if role not in wc.BACKEND_ROLES or role not in run.roles:
            raise BoundaryRejected(f"role {role!r} is not a scheduled role of this run")
        _check_primary(run)
        if source_mode not in SOURCE_MODES:
            raise BoundaryRejected(f"source_mode must be one of {SOURCE_MODES}")
        if role in NEURAL_ROLES and source_mode != "recorded_finite":
            raise LiveNeuralUnsupported("neural roles require a finite recorded manifest; live/camera-style neural "
                                        "sources are unsupported until a reviewed source/admission contract exists")
        if source_mode == "camera_style_synthetic" and clip.clock_kind != "timestamp_seconds":
            raise BoundaryRejected("a camera-style source needs a timestamp_seconds clock")
        if clip.intrinsics.status == "unavailable" or clip.intrinsics.K is None:
            raise BoundaryRejected("calibration_unavailable: K must be bound at initialization")
        if not isinstance(binding, Mapping):
            raise BoundaryRejected("binding must be a mapping")
        b = wc.loads_strict(_frozen_json(binding, "binding"))
        if role in NEURAL_ROLES:
            warmup = _check_neural_binding(b, clip, run)
        else:
            _check_relative_binding(b, clip, run)
            warmup = ()
        if warmup_frame_ids is not None and tuple(warmup_frame_ids) != warmup:
            raise BoundaryRejected("explicit warm-up differs from the frozen seed plan")
        permitted = tuple(warmup) + tuple(run.scheduled_frame_ids)
        position = clip.frame_position()
        if any(f not in position for f in permitted) or len(set(permitted)) != len(permitted):
            raise BoundaryRejected("permitted order contains unknown or duplicate frames")
        pos = [position[f] for f in permitted]
        if any(b2 <= a for a, b2 in zip(pos, pos[1:])):
            raise BoundaryRejected("permitted order must follow strictly increasing clip order")
        binding_json = _frozen_json(b, "binding")
        return cls(clip=clip, run=run, role=role, source_mode=source_mode, warmup_frame_ids=tuple(warmup),
                   scheduled_frame_ids=tuple(run.scheduled_frame_ids), permitted_order=permitted,
                   K=tuple(float(v) for v in clip.intrinsics.K), width=clip.width, height=clip.height,
                   binding_json=binding_json, binding_sha256=hashlib.sha256(binding_json).hexdigest(),
                   clip_sha256=wc.canonical_sha256(clip.to_dict()), run_sha256=wc.canonical_sha256(run.to_dict()),
                   prefix_attestation_json=_frozen_json(attestation, "prefix attestation"))

    @property
    def binding(self) -> dict:
        return wc.loads_strict(self.binding_json)

    def K_matrix(self) -> np.ndarray:
        K = np.array(self.K, dtype=np.float64).reshape(3, 3)
        K.setflags(write=False)
        return K

    def frame(self, frame_id: str) -> wc.FrameSpec:
        return self.clip.frame(frame_id)

    def phase(self, frame_id: str) -> str:
        return "warmup" if frame_id in self.warmup_frame_ids else "scored"

    def to_dict(self) -> dict:
        return {"schema": self.schema, "boundary_version": BOUNDARY_VERSION, "clip_id": self.clip.clip_id,
                "run_id": self.run.run_id, "role": self.role, "source_mode": self.source_mode,
                "clock_kind": self.clip.clock_kind, "warmup_frame_ids": list(self.warmup_frame_ids),
                "scheduled_frame_ids": list(self.scheduled_frame_ids), "K": list(self.K),
                "width": self.width, "height": self.height, "binding": self.binding,
                "binding_sha256": self.binding_sha256, "clip_sha256": self.clip_sha256, "run_sha256": self.run_sha256,
                "prefix_attestation": wc.loads_strict(self.prefix_attestation_json)}


def _context_fields(ctx: Any) -> dict:
    """Every field of a context, as plain JSON (clip/run via their contract serializations)."""
    out = {}
    for f in dataclasses.fields(FrameContextV1):
        value = getattr(ctx, f.name)
        if f.name in ("clip", "run"):
            value = value.to_dict()
        out[f.name] = _plain(value)
    return wc.loads_strict(wc.canonical_json(out))


def canonical_context(context: Any) -> tuple[FrameContextV1, str]:
    """Re-derive the canonical context from its validated clip, run and binding; refuse any difference."""
    if type(context) is not FrameContextV1:
        raise BoundaryRejected("context must be a FrameContextV1")
    if context.schema != SCHEMA_CONTEXT:
        raise BoundaryRejected(f"unsupported context schema {context.schema!r}")
    try:
        fields = _context_fields(context)
        binding = wc.loads_strict(bytes(context.binding_json))
    except Exception as exc:  # noqa: BLE001 - any malformed field is a refusal
        raise BoundaryRejected(f"context fields are malformed ({type(exc).__name__}: {exc})") from exc
    canonical = FrameContextV1.build(context.clip, context.run, context.role, binding, source_mode=context.source_mode)
    if _context_fields(canonical) != fields:
        diff = sorted(k for k in fields if _context_fields(canonical)[k] != fields[k])
        raise BoundaryRejected(f"context differs from its canonical rebuild: {diff}")
    return canonical, wc.canonical_sha256(fields)


def _check_neural_binding(b: dict, clip: wc.ClipSpec, run: wc.RunSpec) -> tuple[str, ...]:
    if set(b) != NEURAL_BINDING_KEYS or b["kind"] != "neural":
        raise BoundaryRejected("neural roles need an exact neural binding (see neural_binding)")
    if (b["pose_units"] != "metric" or b["metric_geometry"] is not True or b["automatic_reacquisition"] is not False
            or b["admission_inherited_by_synthetic_sources"] is not False
            or not isinstance(b["family"], str) or not b["family"].strip()):
        raise BoundaryRejected("neural binding must be metric, without reacquisition or inherited admission")
    if b["bundle"] is None:
        if b["seed_frame_id"] is not None or b["seed_target_frame_id"] is not None or b["warmup_frame_ids"]:
            raise BoundaryRejected("a seed/warm-up requires a bound metric bundle")
        return ()
    rec = b["bundle"]
    if (not isinstance(rec, dict) or set(rec) != BUNDLE_RECORD_KEYS or rec["seed_frame_id"] != b["seed_frame_id"]
            or rec["family"] != b["family"] or rec["metric_provenance"] not in METRIC_PROVENANCE
            or not _hex(rec["glb_sha256"]) or isinstance(rec["obj_id"], bool) or not isinstance(rec["obj_id"], int)):
        raise BoundaryRejected("neural bundle binding is incomplete, inconsistent or not metric")
    if rec["seed_pose_m"] is not None:
        try:
            seed = [list(r) for r in wc._pose(rec["seed_pose_m"], "seed_pose_m")]
        except wc.ContractError as exc:
            raise BoundaryRejected(f"bound seed pose is not SE(3) ({exc})") from exc
        if rec["seed_pose_sha256"] != wc.canonical_sha256(seed):
            raise BoundaryRejected("bound seed pose hash differs")
        if not _seed_label_ok(rec["seed_label"]):
            raise BoundaryRejected("bound seed pose must be labelled controlled_init:/onboarding:")
    elif rec["seed_pose_sha256"] is not None:
        raise BoundaryRejected("seed pose hash without a seed pose")
    target, warmup = neural_seed_plan(clip, run, b["seed_frame_id"])
    if b["seed_target_frame_id"] != target or tuple(b["warmup_frame_ids"]) != warmup:
        raise BoundaryRejected("seed target/warm-up differ from the frozen seed plan")
    return warmup


def _check_relative_binding(b: dict, clip: wc.ClipSpec, run: wc.RunSpec) -> None:
    if set(b) != RELATIVE_BINDING_KEYS or b["kind"] != "relative":
        raise BoundaryRejected("the relative role needs an exact proof-derived binding (see relative_binding)")
    if (b["pose_units"] != "object_unit" or b["metric_geometry"] is not False or b["package_units"] != "object_unit"
            or b["metric_scale_available"] is not False):
        raise BoundaryRejected("relative tracking requires an object_unit package; metric relabelling is refused")
    if b["status"] != "available" or b["frozen"] is not True:
        raise BoundaryRejected("relative tracking requires an available frozen package")
    if b["clip_id"] != clip.clip_id or b["artwork_kind"] != run.artwork_kind:
        raise BoundaryRejected("relative binding clip/artwork differs from the run")
    if run.artwork_kind not in BUILTIN_ARTWORK:
        raise BoundaryRejected("imported_artwork_unavailable: no imported artwork file is bound to this run")
    if run.surface_package_id is not None and run.surface_package_id != b["package_id"]:
        raise BoundaryRejected("run surface_package_id differs from the bound package")
    position = clip.frame_position()
    onboarding = set(run.onboarding.keyframe_ids)
    first_scored = position[run.scheduled_frame_ids[0]]
    prefix = b["prefix_frame_ids"]
    if (not prefix or any(f not in onboarding or f not in position or position[f] >= first_scored for f in prefix)
            or b["frozen_after_frame_id"] not in prefix or any(c not in prefix for c in b["camera_frame_ids"])):
        raise BoundaryRejected("package prefix/cameras/freeze must lie within the reviewed onboarding prefix, before scoring")
    art, generated, marker_sha = b["package_artwork"], b["generated_artwork_sha256"], b["marker_artwork_sha256"]
    if art is not None and generated != art.get("sha256"):
        raise BoundaryRejected("generated artwork differs from the package artwork SHA")
    if marker_sha is not None and (art is None or generated != marker_sha):
        raise BoundaryRejected("generated artwork differs from the marker's declared artwork SHA")


# --------------------------------------------------------------------------
# frame input


@dataclass(frozen=True)
class FrameInputV1:
    """One source frame: the manifest record, owned read-only pixel copies and their declared identities.
    Revalidated in full at ``FrameBoundary.process`` (construction checks are not trusted)."""
    frame: wc.FrameSpec
    rgb: np.ndarray | None
    image_sha256: str | None
    decoder_media_time_s: float | None
    mask: np.ndarray | None
    mask_frame_id: str | None
    mask_sha256: str | None
    mask_file_sha256: str | None
    rgb_unavailable_reason: str | None
    mask_unavailable_reason: str | None
    metadata_json: bytes
    schema: str = SCHEMA_INPUT

    @classmethod
    def create(cls, frame: wc.FrameSpec, rgb: Any, *, image_sha256: str | None = None,
               decoder_media_time_s: float | None = None, mask: Any = None, mask_frame_id: str | None = None,
               mask_sha256: str | None = None, mask_file_sha256: str | None = None,
               rgb_unavailable_reason: str | None = None, mask_unavailable_reason: str | None = None,
               metadata: Mapping[str, Any] | None = None) -> "FrameInputV1":
        t = decoder_media_time_s
        if t is not None and not isinstance(t, (bool, np.bool_)) and isinstance(t, (int, float, np.integer, np.floating)):
            t = float(t)
        inp = cls(frame=frame, rgb=None if rgb is None else _readonly_copy(rgb, "rgb"), image_sha256=image_sha256,
                  decoder_media_time_s=t,
                  mask=None if mask is None else _readonly_copy(mask, "mask"), mask_frame_id=mask_frame_id,
                  mask_sha256=mask_sha256, mask_file_sha256=mask_file_sha256,
                  rgb_unavailable_reason=None if rgb_unavailable_reason is None else str(rgb_unavailable_reason)[:MAX_REASON_CHARS],
                  mask_unavailable_reason=None if mask_unavailable_reason is None else str(mask_unavailable_reason)[:MAX_REASON_CHARS],
                  metadata_json=_metadata(metadata))
        validate_input(inp)
        return inp

    @classmethod
    def from_reader(cls, frame: wc.FrameSpec, rgb: Any, ident: Mapping[str, Any] | None, mask: Any,
                    mident: Mapping[str, Any] | None, **kw) -> "FrameInputV1":
        """From ``FrameReader.read`` and ``MaskStore.load`` results (either may be absent)."""
        ident = dict(ident or {})
        if rgb is not None and (ident.get("frame_id") != frame.frame_id
                                or ident.get("source_frame_index") != frame.source_frame_index):
            raise BoundaryRejected("reader identity belongs to another frame")
        mident = dict(mident or {})
        return cls.create(frame, rgb, image_sha256=ident.get("image_sha256") if rgb is not None else None,
                          decoder_media_time_s=ident.get("decoder_media_time_s") if rgb is not None else None,
                          mask=mask, mask_frame_id=mident.get("mask_frame_id"), mask_sha256=mident.get("mask_sha256"),
                          mask_file_sha256=mident.get("file_sha256"), **kw)

    @property
    def metadata(self) -> dict:
        return wc.loads_strict(self.metadata_json)


def _metadata(metadata: Mapping[str, Any] | None) -> bytes:
    if metadata is None:
        return b"{}"
    if not isinstance(metadata, Mapping):
        raise BoundaryRejected("metadata must be a mapping")
    return _check_metadata(_frozen_json(dict(metadata), "metadata"))


def _check_metadata(raw: Any) -> bytes:
    if not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES:
        raise BoundaryRejected("metadata must be canonical JSON bytes of at most 16 KiB")
    try:
        value = wc.loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BoundaryRejected("metadata is not strict JSON") from exc
    if not isinstance(value, dict) or len(value) > MAX_METADATA_KEYS:
        raise BoundaryRejected(f"metadata must be an object with at most {MAX_METADATA_KEYS} keys")
    if any(k in RESERVED_METADATA_KEYS for k in value):
        raise BoundaryRejected("metadata may not carry identity, calibration or binding fields")
    if wc.canonical_json(value) != raw:
        raise BoundaryRejected("metadata bytes are not canonical")
    return raw


def _opt_reason(value: Any, what: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > MAX_REASON_CHARS):
        raise BoundaryRejected(f"{what} must be None or a non-empty bounded string")


def validate_input(inp: Any) -> None:
    """Full structural revalidation of a FrameInputV1 (public constructor or dataclasses.replace)."""
    if type(inp) is not FrameInputV1:
        raise BoundaryRejected("process requires a FrameInputV1")
    if inp.schema != SCHEMA_INPUT:
        raise BoundaryRejected(f"unsupported frame input schema {inp.schema!r}")
    if not isinstance(inp.frame, wc.FrameSpec):
        raise BoundaryRejected("frame must be a FrameSpec")
    try:
        if wc.FrameSpec.from_dict(inp.frame.to_dict()) != inp.frame:
            raise BoundaryRejected("frame record does not survive contract revalidation")
    except wc.ContractError as exc:
        raise BoundaryRejected(f"frame record contract: {exc}") from exc
    for name in ("rgb", "mask"):
        value = getattr(inp, name)
        if value is not None and not isinstance(value, np.ndarray):
            raise BoundaryRejected(f"{name} must be a numpy array or None")
    if inp.rgb is None and inp.rgb_unavailable_reason is None:
        raise BoundaryRejected("a missing RGB needs an explicit unavailable reason")
    t = inp.decoder_media_time_s
    if t is not None and (type(t) is not float or not math.isfinite(t) or t < 0):
        raise BoundaryRejected("decoder media time must be None or a finite non-negative float")
    for name in ("image_sha256", "mask_sha256", "mask_file_sha256"):
        value = getattr(inp, name)
        if value is not None and not _hex(value):
            raise BoundaryRejected(f"{name} must be None or a lowercase SHA-256 digest")
    if inp.mask_frame_id is not None and (not isinstance(inp.mask_frame_id, str) or not wc.ID_RE.fullmatch(inp.mask_frame_id)):
        raise BoundaryRejected("mask_frame_id must be None or a frame identifier")
    _opt_reason(inp.rgb_unavailable_reason, "rgb_unavailable_reason")
    _opt_reason(inp.mask_unavailable_reason, "mask_unavailable_reason")
    _check_metadata(inp.metadata_json)


@dataclass(frozen=True)
class FrameOutputV1:
    """A re-validated FrameResult, an owned overlay copy (or None) and the replay epoch."""
    result: wc.FrameResult
    overlay_rgba: np.ndarray | None
    epoch: int
    phase: str
    position: int
    process_ms: float
    boundary_reason: str | None
    metadata_json: bytes
    schema: str = SCHEMA_OUTPUT


# --------------------------------------------------------------------------
# adapters around the real session classes


class FrameAdapter:
    """Adapter protocol; concrete adapters wrap the actual session classes only."""
    roles: tuple[str, ...] = ()

    def open(self, context: FrameContextV1) -> dict:          # pragma: no cover - protocol
        raise NotImplementedError

    def backend_identity(self) -> dict:                         # pragma: no cover - protocol
        raise NotImplementedError

    def process(self, context, frame, rgb, mask, identity, K):  # pragma: no cover - protocol
        raise NotImplementedError

    def restart(self) -> None:                                  # pragma: no cover - protocol
        raise NotImplementedError

    def close(self) -> None:                                    # pragma: no cover - protocol
        raise NotImplementedError


class NeuralSessionAdapter(FrameAdapter):
    """Wraps an actual ``workbench_neural.NeuralSession`` (never a subclass). Its admission, family,
    head/base, provenance and observed-mask gates stay authoritative, and this adapter snapshots the
    complete session/backend/bundle/admission binding and refuses any drift."""
    roles = NEURAL_ROLES

    def __init__(self, session: Any):
        self.session = session
        self._opened = False
        self._snapshot: bytes | None = None
        self._identity_json: bytes | None = None

    def _state(self) -> dict:
        s = self.session
        backend = s.backend
        bundle = getattr(backend, "bundle", None)
        adm = s.admission
        return {"session": {"class": f"{type(s).__module__}.{type(s).__qualname__}", "role": s.role, "run_id": s.run_id,
                            "clip_sha256": wc.canonical_sha256(s.clip.to_dict()), "renderer_bound": s.renderer is not None},
                "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
                "install_provenance_sha256": wc.canonical_sha256(_plain(dict(backend.install_provenance()))),
                "bundle": None if bundle is None else bundle_record(bundle),
                "seed_target_frame_id": getattr(backend, "seed_target_frame_id", None),
                "admission": {"admitted": adm.admitted, "reasons": list(adm.reasons), "body": adm.body},
                "identity": s.identity()}

    def open(self, context: FrameContextV1) -> dict:
        from . import workbench_neural as wn
        if self._opened:
            raise BoundaryRejected("an adapter can be opened only once")
        s = self.session
        if type(s) is not wn.NeuralSession:
            raise BoundaryRejected("NeuralSessionAdapter requires an actual workbench_neural.NeuralSession")
        if getattr(s, "_closed", False):
            raise BoundaryRejected("neural session is already closed")
        state_dict = getattr(s, "_state", None)
        if (not isinstance(state_dict, dict) or state_dict or getattr(s, "_last_index", None) is not None
                or getattr(s, "last_overlay", None) is not None
                or getattr(s.backend, "_seed_used", False) is not False):
            raise BoundaryRejected("neural session is not pristine (carried state, consumed frames, a retained overlay or "
                                   "a used seed); pass a new session - it is never reset or re-armed implicitly")
        if s.role != context.role or s.run_id != context.run.run_id:
            raise BoundaryRejected("session role/run differ from the context")
        if wc.canonical_sha256(s.clip.to_dict()) != context.clip_sha256:
            raise BoundaryRejected("session clip metadata differs from the initialized context clip")
        b = context.binding
        adm = s.admission
        if (type(adm) is not wn.ServingAdmission or adm.admitted is not True
                or b["family"] not in (adm.body.get("families") or [])):
            raise BoundaryRejected(f"serving admission does not cover family {b['family']!r}")
        state = self._state()
        if b["bundle"] is not None:
            if state["bundle"] != b["bundle"]:
                raise BoundaryRejected("backend metric bundle (seed pose, object, GLB, labels or provenance) differs "
                                       "from the bound bundle")
            if state["seed_target_frame_id"] != b["seed_target_frame_id"]:
                raise BoundaryRejected("backend seed target differs from the frozen seed plan")
        elif state["bundle"] is not None or state["seed_target_frame_id"] is not None:
            raise BoundaryRejected("a backend bundle/seed target requires a bound metric bundle")
        identity = state["identity"]
        if identity.get("admission_contract_sha256") != adm.body.get("contract_sha256"):
            raise BoundaryRejected("session identity does not carry the admission contract")
        self._snapshot = _frozen_json(state, "neural session state")
        self._identity_json = _frozen_json(identity, "session identity")
        self._opened = True
        return {"adapter": "NeuralSessionAdapter", "session_class": state["session"]["class"],
                "backend_class": state["backend_class"], "identity_sha256": wc.canonical_sha256(identity),
                "state_sha256": hashlib.sha256(self._snapshot).hexdigest(),
                "admission_sha256": wc.canonical_sha256(_plain(state["admission"])),
                "admission_contract_sha256": adm.body.get("contract_sha256"), "family": b["family"],
                "install_digest": identity.get("install_digest"), "seed_target_frame_id": state["seed_target_frame_id"],
                "bundle_record_sha256": None if state["bundle"] is None else wc.canonical_sha256(state["bundle"])}

    def backend_identity(self) -> dict:
        return wc.loads_strict(self._identity_json)

    def _check_drift(self, when: str) -> None:
        try:
            current = _frozen_json(self._state(), "neural session state")
        except BoundaryRejected as exc:
            raise BoundaryFailure(f"neural session state became invalid {when}: {exc}") from exc
        if current != self._snapshot:
            raise BoundaryFailure(f"neural session/backend/bundle/seed/admission drift detected {when}")

    def process(self, context, frame, rgb, mask, identity, K):
        self._check_drift("before process")
        row = self.session.process(frame, rgb, mask, identity, K)
        overlay = None
        if row.render_state == "rendered" and self.session.last_overlay is not None:
            overlay = np.array(self.session.last_overlay, copy=True)
        self.session.last_overlay = None
        self._check_drift("after process")
        return row, overlay

    def restart(self) -> None:
        self._check_drift("before restart")   # a drifted seed is never re-armed
        self.session.reset()                  # explicit only: re-arms the one-use seed for a full replay
        self._check_drift("after restart")

    def close(self) -> None:
        self.session.close()


class RelativeTrackerAdapter(FrameAdapter):
    """Wraps an actual ``workbench_geometry.RelativeTracker`` and ``render_marker`` exactly as the legacy
    relative stage does, over arrays decoded from a ``RelativePackageProof``. Poses are always ``object_unit``."""
    roles = (RELATIVE_ROLE,)

    def __init__(self, tracker: Any, proof: RelativePackageProof):
        self.tracker, self.proof = tracker, proof
        self._opened = False
        self._identity_json: bytes | None = None
        self._cfg: bytes | None = None

    def open(self, context: FrameContextV1) -> dict:
        from . import workbench_geometry as wg
        from .workbench_cli import TRACKER_VERSION
        if self._opened:
            raise BoundaryRejected("an adapter can be opened only once")
        if type(self.tracker) is not wg.RelativeTracker:
            raise BoundaryRejected("RelativeTrackerAdapter requires an actual workbench_geometry.RelativeTracker")
        binding, pkg, arrays = _relative_binding_and_arrays(self.proof, context.run.artwork_kind)
        if _frozen_json(binding, "relative binding") != context.binding_json:
            raise BoundaryRejected("package proof differs from the bound package (record, components or artwork)")
        self.tracker.initialize({"package_id": pkg.package_id, "status": pkg.status, "frozen": pkg.frozen,
                                 "points": arrays["descriptor_points"], "descriptors": arrays["descriptor_values"]})
        self._pkg, self._arrays = pkg, arrays
        self._art = wg.builtin_artwork(context.run.artwork_kind, pkg.artwork["edge_px"] if pkg.artwork else 256)
        self._patch = None
        if pkg.marker is not None:
            self._patch = wg.marker_patch(arrays["vertices"], arrays["faces"], arrays["face_support_views"],
                                          pkg.marker.to_dict())
        identity = {"role": RELATIVE_ROLE, "tracker": TRACKER_VERSION, "quality_claim": False,
                    "package_id": pkg.package_id, "package_sha256": self.proof.surface_json_sha256}
        self._identity_json = _frozen_json(identity, "relative identity")
        self._cfg = _frozen_json(self.tracker.cfg, "tracker cfg")
        self._opened = True
        return {"adapter": "RelativeTrackerAdapter", "tracker_class": "bench.workbench_geometry.RelativeTracker",
                "renderer": "bench.workbench_geometry.render_marker", "tracker_cfg": dict(self.tracker.cfg),
                "identity_sha256": wc.canonical_sha256(identity), "pose_units": "object_unit", "metric_geometry": False,
                "proof_source": self.proof.source, "surface_json_sha256": self.proof.surface_json_sha256}

    def backend_identity(self) -> dict:
        return wc.loads_strict(self._identity_json)

    def process(self, context, frame, rgb, mask, identity, K):
        from . import workbench_geometry as wg
        if _frozen_json(self.tracker.cfg, "tracker cfg") != self._cfg:
            raise BoundaryFailure("tracker configuration changed during the session")
        out = self.tracker.process(rgb, mask, identity, K)
        render_state, evidence, pose, overlay = "unavailable", {}, None, None
        if out.pose_state == "tracking" and out.pose is not None:
            pose = [list(map(float, r)) for r in out.pose]
            render_state = "suppressed"
            if self._pkg.marker is not None and self._patch is not None:
                rendered = wg.render_marker(self._arrays["vertices"], self._arrays["faces"], self._arrays["face_support_views"],
                                            self._pkg.marker.to_dict(), self._art, K, out.pose, context.width,
                                            context.height, mask, patch=self._patch)
                if rendered["stats"]["rendered_pixels"] > 0:
                    render_state = "rendered"
                    overlay = np.array(rendered["rgba"], copy=True)
                evidence["render_stats"] = _plain(rendered["stats"])
        elif mask is not None:
            render_state = "suppressed"
        if identity.get("mask_file_sha256") is not None:
            evidence["mask_file_sha256"] = identity["mask_file_sha256"]
        reason = out.failure_reason
        if out.pose_state == "tracking" and render_state != "rendered":
            reason = None
        observed = out.mask_state == "observed"
        row = wc.FrameResult(
            run_id=context.run.run_id, clip_id=context.clip.clip_id, frame_id=frame.frame_id,
            source_frame_index=frame.source_frame_index, timestamp_s=frame.timestamp_s,
            media_time_s=identity.get("decoder_media_time_s") if frame.media_time_s is None else frame.media_time_s,
            backend_role=RELATIVE_ROLE, backend_identity=self.backend_identity(), mask_state=out.mask_state,
            pose_state=out.pose_state, render_state=render_state, failure_reason=reason, pose=pose,
            pose_units="object_unit" if pose is not None else None, image_sha256=identity["image_sha256"],
            mask_frame_id=identity.get("mask_frame_id") if observed else None,
            mask_sha256=identity.get("mask_sha256") if observed else None,
            metrics=_plain(dict(out.metrics)), evidence=evidence)
        return row, overlay

    def restart(self) -> None:
        self.tracker.reset()

    def close(self) -> None:
        self.tracker.close()


# --------------------------------------------------------------------------
# the boundary


class FrameBoundary:
    """initialize(context, adapter) -> process(FrameInputV1)* -> reset('replay_restart') -> close().

    ``initialize`` takes ownership of the adapter (and so its session/backend or tracker): on ANY
    initialization failure the adapter is closed and ``cleanup_errors`` records close errors."""

    def __init__(self):
        self._state = "new"
        self._ctx: FrameContextV1 | None = None
        self._caller_ctx: Any = None
        self._caller_fingerprint: str | None = None
        self._adapter: FrameAdapter | None = None
        self._open_record: bytes | None = None
        self._identity_json: bytes | None = None
        self._epoch = 0
        self._cursor = 0
        self._invalid: str | None = None
        self._last_index: int | None = None
        self._last_ts: float | None = None
        self.cleanup_errors: list[str] = []

    # lifecycle -----------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def expected_next(self) -> str | None:
        if self._ctx is None or self._cursor >= len(self._ctx.permitted_order):
            return None
        return self._ctx.permitted_order[self._cursor]

    def identity(self) -> dict:
        if self._identity_json is None:
            raise BoundaryStateError("boundary is not initialized")
        return {"backend_identity": wc.loads_strict(self._identity_json), "adapter": wc.loads_strict(self._open_record),
                "boundary_version": BOUNDARY_VERSION, "context_binding_sha256": self._ctx.binding_sha256,
                "context_fingerprint": self._caller_fingerprint, "role": self._ctx.role,
                "source_mode": self._ctx.source_mode}

    def initialize(self, context: FrameContextV1, adapter: FrameAdapter) -> dict:
        if self._state != "new":
            raise BoundaryStateError(f"initialize is allowed once (state {self._state})")
        try:
            canonical, fingerprint = canonical_context(context)
            if not isinstance(adapter, (NeuralSessionAdapter, RelativeTrackerAdapter)) or canonical.role not in adapter.roles:
                raise BoundaryRejected("adapter kind does not serve this role")
            record = adapter.open(canonical)
            open_record = _frozen_json(record, "adapter record")
            identity_json = _frozen_json(adapter.backend_identity(), "backend identity")
        except BaseException:
            self.cleanup_errors += _safe_close(adapter)
            self._state = "closed"
            raise
        self._ctx, self._caller_ctx, self._caller_fingerprint = canonical, context, fingerprint
        self._adapter, self._open_record, self._identity_json = adapter, open_record, identity_json
        self._state = "open"
        return self.identity()

    def reset(self, reason: str = "replay_restart") -> None:
        if reason not in RESET_REASONS:
            raise BoundaryRejected(f"reset reason must be one of {RESET_REASONS}")
        self._require_open()
        self._check_caller_context()
        try:
            self._adapter.restart()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
        self._epoch += 1
        self._cursor = 0
        self._invalid = None
        self._last_index = self._last_ts = None

    def close(self) -> list[str]:
        """Idempotent. Returns cleanup errors instead of hiding or raising them."""
        if self._state == "open":
            errors = _safe_close(self._adapter)
            self.cleanup_errors += errors
            self._state = "closed"
            return errors
        if self._state == "new":
            self._state = "closed"
        return []

    def _require_open(self) -> None:
        if self._state == "new":
            raise BoundaryStateError("boundary is not initialized")
        if self._state != "open":
            raise BoundaryStateError(f"boundary is {self._state}")

    def _check_caller_context(self) -> None:
        try:
            current = wc.canonical_sha256(_context_fields(self._caller_ctx))
        except Exception:  # noqa: BLE001
            current = None
        if current != self._caller_fingerprint:
            self.cleanup_errors += _safe_close(self._adapter)
            self._state = "failed"
            raise ContextMutated("the caller's context object changed after initialize; adapter closed")

    def _fail(self, exc: BaseException) -> None:
        errors = _safe_close(self._adapter)
        self.cleanup_errors += errors
        self._state = "failed"
        note = f"; cleanup errors: {errors}" if errors else ""
        raise BoundaryFailure(f"adapter failed ({type(exc).__name__}: {exc}); resources closed{note}") from exc

    # processing ----------------------------------------------------------
    def process(self, frame_input: FrameInputV1) -> FrameOutputV1:
        self._require_open()
        self._check_caller_context()
        validate_input(frame_input)
        ctx = self._ctx
        expected = self.expected_next
        if expected is None:
            raise BoundaryRejected("permitted order is exhausted; an explicit replay restart is required")
        frame = frame_input.frame
        if frame.frame_id != expected:
            raise BoundaryRejected(f"order: expected {expected}, got {frame.frame_id} "
                                   "(duplicate, reversed or skipped frames are refused)")
        if frame != ctx.frame(expected):
            raise BoundaryRejected("frame record differs from the immutable manifest record")
        ts = frame.timestamp_s
        if ctx.clip.clock_kind == "timestamp_seconds":
            if ts is None or not math.isfinite(ts) or (self._last_ts is not None and ts <= self._last_ts):
                raise BoundaryRejected("timestamp_seconds must be finite and strictly increasing")
        elif ts is not None:
            raise BoundaryRejected("frame_index clocks keep timestamp_s null")
        if self._last_index is not None and frame.source_frame_index <= self._last_index:
            raise BoundaryRejected("source_frame_index must advance")
        if frame_input.rgb is not None:
            if ctx.clip.source_kind == "video_file" and frame.media_time_s is not None and frame_input.decoder_media_time_s is None:
                raise BoundaryRejected("the manifest declares media_time_s; a decoder PTS is required")
            if ctx.clip.source_kind == "image_sequence" and frame_input.decoder_media_time_s is not None:
                raise BoundaryRejected("image sequences have no decoder PTS")
        t0 = time.perf_counter()
        position, phase = self._cursor, ctx.phase(frame.frame_id)
        self._cursor += 1                      # the frame is consumed (and counted) from here on
        self._last_index, self._last_ts = frame.source_frame_index, ts
        row, overlay, boundary_reason = self._process_consumed(frame_input, frame)
        overlay = overlay if (overlay is not None and row.render_state == "rendered") else None
        return FrameOutputV1(result=row, overlay_rgba=overlay, epoch=self._epoch, phase=phase, position=position,
                             process_ms=(time.perf_counter() - t0) * 1000.0, boundary_reason=boundary_reason,
                             metadata_json=frame_input.metadata_json)

    def _no_inference_row(self, frame: wc.FrameSpec, reason: str, mask_state: str, pose_state: str) -> wc.FrameResult:
        ctx = self._ctx
        return wc.FrameResult(
            run_id=ctx.run.run_id, clip_id=ctx.clip.clip_id, frame_id=frame.frame_id,
            source_frame_index=frame.source_frame_index, timestamp_s=frame.timestamp_s, media_time_s=frame.media_time_s,
            backend_role=ctx.role, backend_identity=wc.loads_strict(self._identity_json), mask_state=mask_state,
            pose_state=pose_state, render_state="unavailable", failure_reason=reason, pose=None, pose_units=None,
            image_sha256=None, mask_frame_id=None, mask_sha256=None, metrics={},
            evidence={"boundary": "no inference for this frame", "epoch": self._epoch})

    def _process_consumed(self, inp: FrameInputV1, frame: wc.FrameSpec):
        from . import workbench_geometry as wg
        ctx = self._ctx
        if self._invalid is not None:
            reason = f"tracking_invalidated: explicit replay restart required ({self._invalid})"[:1000]
            return self._no_inference_row(frame, reason, "blocked", "blocked"), None, reason
        if inp.rgb is None:
            reason = f"rgb_unavailable: {inp.rgb_unavailable_reason}"[:1000]
            self._invalid = reason
            return self._no_inference_row(frame, reason, "blocked", "blocked"), None, reason
        rgb = inp.rgb
        reason = None
        if rgb.dtype != np.uint8 or rgb.shape != (ctx.height, ctx.width, 3):
            reason = f"rgb_dimension_mismatch: expected {ctx.height}x{ctx.width}x3 uint8"
        elif inp.image_sha256 is None or wg.array_sha256(rgb) != inp.image_sha256:
            reason = "image_identity_mismatch: pixels differ from the declared frame identity"
        elif (frame.media_time_s is not None and inp.decoder_media_time_s is not None
              and abs(inp.decoder_media_time_s - frame.media_time_s) > MEDIA_TIME_TOL_S):
            reason = "decoder_media_time_mismatch: decoder PTS differs from the manifest"
        if reason is not None:
            self._invalid = reason
            return self._no_inference_row(frame, reason, "rejected", "unavailable"), None, reason
        mask, mask_reason = inp.mask, None
        if mask is not None:
            if mask.shape != (ctx.height, ctx.width):
                mask_reason = "mask_dimension_mismatch"
            elif inp.mask_frame_id != frame.frame_id:
                mask_reason = "stale_mask: mask belongs to another frame"
            elif inp.mask_sha256 is None or wg.mask_sha256(mask) != inp.mask_sha256:
                mask_reason = "stale_mask: mask identity does not match its pixels"
            if mask_reason is not None:
                mask = None              # withheld -> the session's own safe mask-unavailable guard
        identity = {"frame_id": frame.frame_id, "source_frame_index": frame.source_frame_index,
                    "image_sha256": inp.image_sha256, "decoder_media_time_s": inp.decoder_media_time_s}
        if mask is not None:
            identity.update(mask_frame_id=inp.mask_frame_id, mask_sha256=inp.mask_sha256,
                            mask_file_sha256=inp.mask_file_sha256)
        try:
            row, overlay = self._adapter.process(ctx, frame, np.array(rgb, copy=True),
                                                 None if mask is None else np.array(mask, copy=True),
                                                 identity, ctx.K_matrix())
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
        try:
            if mask_reason is not None:
                if row.pose is not None or row.mask_state != "unavailable":
                    raise BoundaryFailure("withheld mask still produced a mask/pose state")
                row = dataclasses.replace(row, mask_state="rejected", render_state="unavailable",
                                          failure_reason=mask_reason, mask_frame_id=None, mask_sha256=None)
                overlay = None
            elif inp.mask is None and inp.mask_unavailable_reason:
                row = dataclasses.replace(row, evidence={**dict(row.evidence or {}),
                                                         "mask_unavailable_reason": inp.mask_unavailable_reason})
            row = self._validate(row, frame, inp, mask)
        except Exception as exc:  # noqa: BLE001 - an invalid adapter row closes the session
            self._fail(exc)
        return row, overlay, mask_reason

    def _validate(self, row: Any, frame: wc.FrameSpec, inp: FrameInputV1, mask: np.ndarray | None) -> wc.FrameResult:
        ctx = self._ctx
        if not isinstance(row, wc.FrameResult):
            raise BoundaryFailure("adapter did not return a FrameResult")
        row = wc.FrameResult.from_dict(_plain(row.to_dict()))
        if (row.run_id != ctx.run.run_id or row.clip_id != ctx.clip.clip_id or row.frame_id != frame.frame_id
                or row.source_frame_index != frame.source_frame_index or row.timestamp_s != frame.timestamp_s
                or row.backend_role != ctx.role):
            raise BoundaryFailure("adapter row identity differs from the manifest frame/run/role")
        if frame.media_time_s is not None and row.media_time_s != frame.media_time_s:
            raise BoundaryFailure("adapter row media time differs from the manifest")
        if frame.media_time_s is None and row.media_time_s not in (None, inp.decoder_media_time_s):
            raise BoundaryFailure("adapter row media time is not the decoder media time")
        units = "metric" if ctx.role in NEURAL_ROLES else "object_unit"
        if row.pose_units not in (None, units):
            raise BoundaryFailure(f"{ctx.role} rows must use {units} pose units")
        if _frozen_json(row.backend_identity, "row identity") != self._identity_json:
            raise BoundaryFailure("row backend identity differs from the initialized identity")
        if row.image_sha256 is not None and row.image_sha256 != inp.image_sha256:
            raise BoundaryFailure("row image identity differs from the verified input")
        if row.mask_state == "observed" and (mask is None or row.mask_sha256 != inp.mask_sha256):
            raise BoundaryFailure("observed row does not carry this frame's verified mask")
        return row
