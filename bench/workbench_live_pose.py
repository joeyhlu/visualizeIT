"""Explicit live-camera lifecycle for the pinned trained GoTrack backend.

Separate from the finite recorded FrameBoundary and NeuralSession contracts.
Importing this module never imports Torch, decodes an image, or creates a model.
The operator must supply reviewed live admission and current observed-mask and
onboarding providers. There is no fallback tracker or client-supplied pose/mask.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import hashlib
import importlib
import importlib.util
import importlib.metadata
import io
import gc
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping
import uuid
import zipfile

import numpy as np

from . import workbench_contract as wc
from . import workbench_geometry as wg
from . import workbench_neural as wn

HEAD = wn.PRIMARY["export_sha256"]
SCHEMA_VERSION = 1
LIMITS = {"max_width": 1920, "max_height": 1920, "max_pixels": 2_073_600,
          "max_jpeg_bytes": 4_194_304, "max_body_bytes": 6_291_456,
          "max_sessions": 1, "max_inflight": 1}
LIVE_SOURCES = ("bench/workbench_live_pose.py", "tools/native_pose_bridge.py", "tools/serve_native_pose.py")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")


class NativeError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.status = code, status


def keys(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise NativeError("invalid_contract", f"{name} requires exactly {sorted(required)}")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise NativeError("invalid_contract", f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise NativeError("invalid_contract", f"{name} must be finite")
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise NativeError("invalid_contract", f"{name} is not a valid identifier")
    return value


def digest(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise NativeError("invalid_contract", f"{name} must be lowercase SHA256")
    return value


@dataclass(frozen=True)
class LiveCalibration:
    """K describes the actual encoded pixels, after the declared camera crop.

    Distortion must already be removed by the camera pipeline. No hidden image
    rotation, mirroring or principal-point conversion is performed by the server.
    """
    body_json: bytes

    @classmethod
    def parse(cls, value):
        d = keys(value, ("calibration_revision", "width", "height", "fx", "fy", "cx", "cy",
                         "units", "orientation", "mirrored", "crop"), "camera")
        identifier(d["calibration_revision"], "calibration_revision")
        w, h = integer(d["width"], "width", 1), integer(d["height"], "height", 1)
        if w > LIMITS["max_width"] or h > LIMITS["max_height"] or w*h > LIMITS["max_pixels"]:
            raise NativeError("camera_limits", "camera dimensions exceed admission limits")
        fx, fy, cx, cy = [finite(d[k], k) for k in ("fx", "fy", "cx", "cy")]
        if min(fx, fy) <= 0 or not (0 <= cx < w and 0 <= cy < h):
            raise NativeError("invalid_calibration", "invalid focal lengths or principal point")
        if d["units"] != "pixels" or d["mirrored"] is not False:
            raise NativeError("invalid_calibration", "pixel intrinsics and unmirrored RGB are required")
        if d["orientation"] not in ("portrait", "portrait_upside_down", "landscape_left", "landscape_right"):
            raise NativeError("invalid_calibration", "unknown encoded-pixel orientation")
        c = keys(d["crop"], ("x", "y", "width", "height"), "crop")
        for k in c:
            integer(c[k], "crop."+k, 1 if k in ("width", "height") else 0)
        if c["width"] != w or c["height"] != h:
            raise NativeError("invalid_calibration", "crop must describe the encoded, unscaled raster")
        return cls(wc.canonical_json(d))

    @property
    def body(self):
        return wc.loads_strict(self.body_json)

    @property
    def sha256(self):
        return hashlib.sha256(self.body_json).hexdigest()

    @property
    def K(self):
        d = self.body
        return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class ObservedMask:
    frame_id: str
    rgb_sha256: str
    mask: np.ndarray
    source: str                 # must be "observed_foreground"


@dataclass(frozen=True)
class LiveInitialization:
    frame_id: str
    rgb_sha256: str
    glb_sha256: str
    pose_m: np.ndarray
    provenance: str             # onboarding:..., produced by trusted provider


def _json(path: Path, maximum=2_097_152):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise NativeError("admission_unavailable", "missing, linked or oversized operator metadata", 503)
    return wc.loads_strict(path.read_bytes())


class ConfiguredGoTrackFactory:
    """Concrete operator composition; real backend construction is fixed here.

    A trusted module supplies only current observations/onboarding and optional
    geometry-bound rendering. It cannot select a substitute backend via HTTP.
    Config, providers, dependencies, assets, evidence and live source hashes are
    authenticated before importing that module or decoding uploaded images.
    """
    def __init__(self, config_path: Path, *, root: Path = wn.ROOT):
        self.root, self.config_path = Path(root).resolve(), Path(config_path).resolve()
        self.config = _json(self.config_path)
        keys(self.config, ("schema", "head_path", "device", "serving_admission_dir", "live_admission_path",
                           "live_admission_sha256", "provider_factory", "provider_sources", "provider_resources", "provider_config",
                           "assets", "frame_timeout_s", "session_idle_timeout_s"), "operator config")
        if self.config["schema"] != "native_gotrack_operator_v1":
            raise NativeError("invalid_config", "unknown operator config", 503)
        for k in ("frame_timeout_s", "session_idle_timeout_s"):
            if not 0 < finite(self.config[k], k) <= (120 if k == "frame_timeout_s" else 3600):
                raise NativeError("invalid_config", "invalid timeout", 503)
        if self.config["device"] not in ("cpu", "cuda:0"):
            raise NativeError("invalid_config", "device must be cpu or cuda:0", 503)
        if not isinstance(self.config["assets"], dict) or not self.config["assets"]:
            raise NativeError("invalid_config", "server asset registry is empty", 503)
        self._last_admission = None

    def path(self, value):
        if not isinstance(value, str) or not value:
            raise NativeError("invalid_config", "operator paths must be strings", 503)
        raw = Path(value)
        p = raw if raw.is_absolute() else self.root/raw
        if p.is_symlink():
            raise NativeError("invalid_config", "operator paths cannot be symlinks", 503)
        resolved = p.resolve()
        # All executable/provider/evidence/model assets belong to this checkout.
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise NativeError("invalid_config", "operator path escapes checkout", 503) from exc
        return resolved

    def _pins(self):
        c = self.config
        if _json(self.config_path) != c:
            raise NativeError("admission_drift", "operator config changed", 503)
        files = set(LIVE_SOURCES)
        if not isinstance(c["provider_sources"], list) or not c["provider_sources"]:
            raise NativeError("invalid_config", "provider source closure is required", 503)
        files.update(c["provider_sources"])
        resources = list(c["provider_resources"])
        if c["provider_factory"] == "bench.workbench_live_pose:sam2_metric_reference_provider":
            files.add("bench/quality_assets.py")
            sam_root = self.path(wn.SAM2_SOURCES_RELATIVE)
            if not (sam_root/"sam2/build_sam.py").is_file():
                raise NativeError("admission_unavailable", "pinned local SAM2 source checkout missing", 503)
            for path in sam_root.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
                    files.add(path.relative_to(self.root).as_posix())
            resources.extend((c["provider_config"]["sam2_checkpoint"], c["provider_config"]["reference_features"]))
        assets = {}
        for name, entry in c["assets"].items():
            identifier(name, "asset_id")
            keys(entry, ("metric_bundle", "references", "artwork_revision", "artwork_files"), "asset")
            identifier(entry["artwork_revision"], "artwork_revision")
            if not isinstance(entry["references"], list) or not entry["references"]:
                raise NativeError("invalid_config", "metric references required", 503)
            if not isinstance(entry["artwork_files"], list):
                raise NativeError("invalid_config", "artwork_files must be a list", 503)
            bundle_path = self.path(entry["metric_bundle"])
            bundle = wn.MetricBundle.from_file(bundle_path, root=self.root)
            self.path(str(bundle.glb_path))
            if bundle.seed_pose_m is not None or bundle.seed_frame_id is not None or bundle.seed_label is not None:
                raise NativeError("invalid_config", "live bundles cannot contain a recorded seed", 503)
            if bundle.principal_point_origin != "integer_centers":
                raise NativeError("invalid_config", "live calibration uses integer pixel centres", 503)
            paths = [entry["metric_bundle"], str(bundle.glb_path), *entry["references"], *entry["artwork_files"]]
            assets[name] = {"family": bundle.family, "glb_sha256": bundle.glb_sha256,
                            "artwork_revision": entry["artwork_revision"],
                            "files": {str(self.path(p).relative_to(self.root).as_posix()): wn.sha256_file(self.path(p))
                                      for p in paths}}
        if not isinstance(c["provider_resources"], list) or not c["provider_resources"]:
            raise NativeError("invalid_config", "provider checkpoint/reference/config resources must be pinned", 503)
        packages = wn.runtime_packages()
        for package in ("Pillow", "hydra-core", "omegaconf", "torchvision", "iopath"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        return {"source_files": {str(self.path(p).relative_to(self.root).as_posix()): wn.sha256_file(self.path(p))
                                  for p in sorted(files)},
                "provider_resources": {str(self.path(p).relative_to(self.root).as_posix()): wn.sha256_file(self.path(p))
                                       for p in resources},
                "provider_factory": c["provider_factory"], "provider_config_sha256": wc.canonical_sha256(c["provider_config"]),
                "assets": assets, "head_sha256": wn.sha256_file(self.path(c["head_path"])),
                "base_sha256": wn.sha256_file(self.root/wn.BASE_CHECKPOINT_RELATIVE),
                "runtime_packages": packages, "device": c["device"],
                "frame_timeout_s": c["frame_timeout_s"], "session_idle_timeout_s": c["session_idle_timeout_s"]}

    def admission(self):
        """Recompute admission; unavailable by default, before decode/import/model.

        The live receipt must be independently reviewed/pinned by the operator;
        this API does not issue receipts or turn assertions into measurements.
        Existing measured full-image/metric serving evidence is still reverified.
        """
        c = self.config
        path = self.path(c["live_admission_path"])
        if wn.sha256_file(path) != digest(c["live_admission_sha256"], "live_admission_sha256"):
            raise NativeError("admission_drift", "reviewed live admission digest differs", 503)
        receipt = _json(path)
        keys(receipt, ("schema", "contract_id", "bindings", "serving_contract_sha256", "calibrations", "evidence"), "live admission")
        identifier(receipt["contract_id"], "live contract_id")
        if receipt["schema"] != "native_live_admission_v1":
            raise NativeError("admission_unavailable", "independent live admission required", 503)
        bindings = self._pins()
        if bindings["head_sha256"] != HEAD or bindings["base_sha256"] != wn.PRIMARY["base_checkpoint_sha256"]:
            raise NativeError("model_identity", "primary head/base differs from trained pins", 503)
        if bindings != receipt["bindings"]:
            raise NativeError("admission_drift", "live source/config/runtime/asset bindings differ", 503)
        serving = wn.load_admission(self.path(c["serving_admission_dir"]), root=self.root)
        if not serving.admitted or serving.body.get("contract_sha256") != receipt["serving_contract_sha256"]:
            raise NativeError("admission_unavailable", "recorded serving admission invalid: "+"; ".join(serving.reasons[:4]), 503)
        families = {v["family"] for v in bindings["assets"].values()}
        if not families.issubset(set(serving.body.get("families", []))):
            raise NativeError("admission_unavailable", "serving admission does not cover registered metric families", 503)
        if not isinstance(receipt["calibrations"], list) or not receipt["calibrations"]:
            raise NativeError("admission_unavailable", "measured live calibration profiles missing", 503)
        calibrations = [LiveCalibration.parse(v).sha256 for v in receipt["calibrations"]]
        evidence = receipt["evidence"]
        keys(evidence, ("path", "sha256"), "live evidence")
        ep = self.path(evidence["path"])
        if wn.sha256_file(ep) != digest(evidence["sha256"], "evidence.sha256"):
            raise NativeError("admission_drift", "live measurement evidence changed", 503)
        measured = _json(ep)
        keys(measured, ("schema", "contract_id", "method", "bindings_sha256", "calibration_sha256", "checks", "trials"), "live evidence")
        required = {"geometry_units", "pixel_centres", "orientation", "mirror", "crop", "observed_mask_binding",
                    "seed_frame_binding", "source_drift_rejected", "stale_frame_rejected", "loss_suppresses_pose"}
        if (measured["schema"] != "native_live_measurements_v1" or
                measured["contract_id"] != receipt["contract_id"] or
                measured["method"] != "measured_live_camera_geometry_and_observed_mask_v1" or
                measured["bindings_sha256"] != wc.canonical_sha256(bindings) or
                measured["calibration_sha256"] != calibrations or
                not isinstance(measured["checks"], dict) or set(measured["checks"]) != required or
                any(v is not True for v in measured["checks"].values())):
            raise NativeError("admission_unavailable", "live measurement coverage differs", 503)
        if not isinstance(measured["trials"], list) or len(measured["trials"]) < 2:
            raise NativeError("admission_unavailable", "saved positive/negative live trials required", 503)
        kinds = set()
        for trial in measured["trials"]:
            keys(trial, ("kind", "path", "sha256"), "trial")
            if trial["kind"] not in ("positive", "negative"):
                raise NativeError("admission_unavailable", "unknown live trial kind", 503)
            kinds.add(trial["kind"])
            if wn.sha256_file(self.path(trial["path"])) != digest(trial["sha256"], "trial.sha256"):
                raise NativeError("admission_drift", "live trial bytes changed", 503)
        if kinds != {"positive", "negative"}:
            raise NativeError("admission_unavailable", "both positive and negative live trials required", 503)
        if importlib.util.find_spec("torch") is None:
            raise NativeError("runtime_unavailable", "Torch runtime missing", 503)
        token = wc.canonical_sha256({"receipt": receipt, "evidence": measured, "serving": serving.body})
        self._last_admission = token
        return receipt, token

    def capabilities(self):
        blockers = []
        try:
            self.admission()
        except Exception as exc:
            blockers.append(f"{type(exc).__name__}: {str(exc)[:300]}")
        if importlib.util.find_spec("torch") is None:
            blockers.append("Torch runtime missing")
        return {"schema_version": 1, "model_head": HEAD, "installed_head_sha256": None,
                "admission_ready": not blockers, "live_camera_supported": not blockers,
                "recorded_finite_supported": True, "blockers": blockers, "limits": dict(LIMITS)}

    def create(self, request):
        d = keys(request, ("schema_version", "expected_model_head", "camera", "asset_id", "onboarding"), "session")
        if type(d["schema_version"]) is not int or d["schema_version"] != 1 or d["expected_model_head"] != HEAD:
            raise NativeError("model_identity", "schema version or expected trained head differs")
        calibration = LiveCalibration.parse(d["camera"])
        asset_id = identifier(d["asset_id"], "asset_id")
        asset = self.config["assets"].get(asset_id)
        if asset is None:
            raise NativeError("asset_not_found", "metric asset is not registered", 404)
        onboarding = keys(d["onboarding"], ("selection_tap", "artwork_revision"), "onboarding")
        tap = keys(onboarding["selection_tap"], ("x", "y"), "selection_tap")
        if not (0 <= finite(tap["x"], "tap.x") < calibration.body["width"] and
                0 <= finite(tap["y"], "tap.y") < calibration.body["height"]):
            raise NativeError("invalid_selection", "tap must be in encoded RGB pixel coordinates")
        # A generated client artwork ID is a display identity, not permission
        # to load any file. Pose-only sessions accept it; rendering separately
        # requires an exact server-registered texture/revision binding.
        identifier(onboarding["artwork_revision"], "artwork_revision")
        receipt, token = self.admission()
        if calibration.sha256 not in [LiveCalibration.parse(v).sha256 for v in receipt["calibrations"]]:
            raise NativeError("camera_not_admitted", "camera calibration has no measured live admission", 503)
        module_name, sep, name = self.config["provider_factory"].partition(":")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module_name) or not name.isidentifier():
            raise NativeError("invalid_config", "provider_factory must be trusted.module:function", 503)
        spec = importlib.util.find_spec(module_name)
        sources = {self.path(p) for p in self.config["provider_sources"]}
        if spec is None or spec.origin is None or Path(spec.origin).resolve() not in sources:
            raise NativeError("invalid_config", "provider module is not in the pinned source closure", 503)
        provider = getattr(importlib.import_module(module_name), name)(
            config=wc.loads_strict(wc.canonical_json(self.config["provider_config"])),
            calibration=calibration, asset_id=asset_id, asset=dict(asset), onboarding=dict(onboarding),
            root=self.root)
        try:
            return LivePoseSession(self, token, calibration, asset_id, onboarding, provider)
        except BaseException:
            provider.close()
            raise


class LivePoseSession:
    """One ordered live stream, one single-use seed, no stale pose publication."""
    def __init__(self, factory, admission_token, calibration, asset_id, onboarding, provider):
        if not all(callable(getattr(provider, n, None)) for n in ("observe", "initialize", "close")):
            raise NativeError("invalid_provider", "provider must implement observe, initialize and close", 503)
        self.factory, self.admission_token = factory, admission_token
        self.calibration, self.asset_id, self.onboarding, self.provider = calibration, asset_id, onboarding, provider
        self.session_id, self.generation = uuid.uuid4().hex, 1
        self.backend = None
        self._install_digest = None
        self._bundle_digest = None
        self._state = {}
        self._closed = False
        self._sequence = self._timestamp = -1
        self._seed_target_sequence = None
        self.last_activity = time.monotonic()

    @property
    def installed_head(self):
        return None if self.backend is None else self.backend.install_provenance().get("head", {}).get("installed_sha256")

    def describe(self):
        return {"schema_version": 1, "session_id": self.session_id, "generation": self.generation,
                "state": "onboarding", "calibration_revision": self.calibration.body["calibration_revision"],
                "artwork_revision": self.onboarding["artwork_revision"], "model_head": HEAD,
                "installed_head_sha256": self.installed_head}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._state.clear()
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            self.provider.close()

    def _check(self):
        if self._closed:
            raise NativeError("session_closed", "native session is closed", 410)
        if time.monotonic()-self.last_activity > self.factory.config["session_idle_timeout_s"]:
            self.close()
            raise NativeError("session_expired", "native session expired", 410)
        try:
            _, token = self.factory.admission()
        except BaseException:
            self.close()
            raise
        if token != self.admission_token:
            self.close()
            raise NativeError("admission_drift", "live admission changed during session", 503)
        if self.backend is not None:
            if (wc.canonical_sha256(self.backend.install_provenance()) != self._install_digest or
                    self.installed_head != HEAD or
                    self._bundle_identity() != self._bundle_digest):
                self.close()
                raise NativeError("model_drift", "installed backend or metric initialization changed", 503)

    def _bundle_identity(self):
        b = self.backend.bundle
        return wc.canonical_sha256({"family": b.family, "glb": str(b.glb_path), "sha": b.glb_sha256,
                                   "obj_id": b.obj_id, "origin": b.principal_point_origin,
                                   "seed": np.asarray(b.seed_pose_m).tolist(), "label": b.seed_label,
                                   "seed_frame_id": b.seed_frame_id, "metric_provenance": b.metric_provenance,
                                   "seed_target_frame_id": self.backend.seed_target_frame_id})

    def process(self, request):
        self._check()               # admission BEFORE base64 decode, Pillow, providers or models
        d = keys(request, ("schema_version", "generation", "sequence", "timestamp_ns", "jpeg_base64",
                           "jpeg_sha256", "calibration_revision", "artwork_revision"), "frame")
        if (type(d["schema_version"]) is not int or d["schema_version"] != 1 or
                type(d["generation"]) is not int or d["generation"] != self.generation):
            raise NativeError("generation_conflict", "schema or session generation differs", 409)
        seq, stamp = integer(d["sequence"], "sequence"), integer(d["timestamp_ns"], "timestamp_ns", 1)
        if seq <= self._sequence or stamp <= self._timestamp:
            raise NativeError("stale_frame", "sequence and timestamp must both advance", 409)
        if (d["calibration_revision"] != self.calibration.body["calibration_revision"] or
                d["artwork_revision"] != self.onboarding["artwork_revision"]):
            raise NativeError("revision_conflict", "calibration or artwork changed; create a new session", 409)
        encoded = d["jpeg_base64"]
        if not isinstance(encoded, str) or len(encoded) > ((LIMITS["max_jpeg_bytes"]+2)//3)*4:
            raise NativeError("frame_too_large", "JPEG exceeds byte limit", 413)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise NativeError("invalid_jpeg", "JPEG is not valid base64") from exc
        if not raw or len(raw) > LIMITS["max_jpeg_bytes"] or hashlib.sha256(raw).hexdigest() != digest(d["jpeg_sha256"], "jpeg_sha256"):
            raise NativeError("image_identity", "JPEG byte count/hash differs")
        from PIL import Image
        try:
            with Image.open(io.BytesIO(raw)) as image:
                camera = self.calibration.body
                if image.format != "JPEG" or image.size != (camera["width"], camera["height"]):
                    raise NativeError("invalid_jpeg", "JPEG format/dimensions differ from calibration")
                if image.getexif().get(274, 1) != 1:
                    raise NativeError("invalid_jpeg", "EXIF rotation is unsupported; encode in calibrated orientation")
                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                rgb.flags.writeable = False
        except NativeError:
            raise
        except Exception as exc:
            raise NativeError("invalid_jpeg", "JPEG decode failed") from exc
        frame_id = f"{self.session_id}:{self.generation}:{seq}"
        rgb_sha = wg.array_sha256(rgb)
        # Consume accepted image identities even if observation/estimation fails.
        self._sequence, self._timestamp = seq, stamp
        self.last_activity = time.monotonic()
        started = time.monotonic()
        try:
            observed = self.provider.observe(frame_id=frame_id, rgb=rgb, rgb_sha256=rgb_sha,
                                             timestamp_ns=stamp)
            if (not isinstance(observed, ObservedMask) or observed.source != "observed_foreground" or
                    observed.frame_id != frame_id or observed.rgb_sha256 != rgb_sha):
                raise NativeError("mask_identity", "provider returned no current observed foreground mask", 503)
            mask = np.array(observed.mask, copy=True)
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise NativeError("mask_contract", "observed mask must be boolean HxW", 503)
            if wg.array_sha256(rgb) != rgb_sha:
                raise NativeError("image_identity", "observation provider mutated current RGB", 503)
            if not mask.any():
                self._state.clear()
                out = {"status": "lost", "pose_m": None, "reason": "observed_mask_empty"}
            elif self.backend is None:
                self._initialize(frame_id, seq, rgb, rgb_sha, mask)
                out = {"status": "recovering", "pose_m": None,
                       "reason": "onboarding initialized; next frame requires temporal pose validation"}
            else:
                if self._seed_target_sequence is not None and seq != self._seed_target_sequence:
                    self._state.clear()
                    # A dropped seed successor cannot silently apply that seed to a later image.
                    self.backend.seed_target_frame_id = None
                    self._bundle_digest = self._bundle_identity()
                self._seed_target_sequence = None
                self._state["_frame_id"] = frame_id
                out = self.backend.estimate(rgb, mask, self.calibration.K, self._state)
            if wg.array_sha256(rgb) != rgb_sha:
                raise NativeError("image_identity", "provider/backend mutated current RGB", 503)
            self._check()
            if time.monotonic()-started > self.factory.config["frame_timeout_s"]:
                raise NativeError("frame_timeout", "frame processing exceeded its time budget", 504)
            return self._response(seq, stamp, frame_id, rgb_sha, mask, out)
        except BaseException:
            # A provider/backend fault may leave causal state partially advanced; never retry it.
            self.close()
            raise

    def _initialize(self, frame_id, seq, rgb, rgb_sha, mask):
        asset = self.factory.config["assets"][self.asset_id]
        bundle = wn.MetricBundle.from_file(self.factory.path(asset["metric_bundle"]), root=self.factory.root)
        init = self.provider.initialize(frame_id=frame_id, rgb=rgb, rgb_sha256=rgb_sha,
                                        observed_mask=mask.copy(), K=self.calibration.K,
                                        selection_tap=dict(self.onboarding["selection_tap"]), bundle=bundle)
        if (not isinstance(init, LiveInitialization) or init.frame_id != frame_id or init.rgb_sha256 != rgb_sha or
                init.glb_sha256 != bundle.glb_sha256 or not isinstance(init.provenance, str) or
                not init.provenance.startswith("onboarding:")):
            raise NativeError("initialization_binding", "initializer did not bind current RGB and metric geometry", 503)
        pose = np.array(wc._pose(np.asarray(init.pose_m).tolist(), "onboarding pose"), dtype=float)
        if pose[2, 3] <= 0:
            raise NativeError("initialization_binding", "metric object origin must be in front of camera", 503)
        live_bundle = replace(bundle, seed_pose_m=pose, seed_label=init.provenance, seed_frame_id=frame_id)
        self.backend = wn.GoTrackMetricBackend("primary_neural", live_bundle,
                                               head_path=self.factory.path(self.factory.config["head_path"]),
                                               device=self.factory.config["device"])
        self.backend.seed_target_frame_id = f"{self.session_id}:{self.generation}:{seq+1}"
        self._seed_target_sequence = seq+1
        provenance = self.backend.install_provenance()
        if (provenance.get("head", {}).get("installed_sha256") != HEAD or
                provenance.get("base_checkpoint_sha256") != wn.PRIMARY["base_checkpoint_sha256"]):
            raise NativeError("model_identity", "actual installed model differs from trained head/base", 503)
        self._install_digest = wc.canonical_sha256(provenance)
        self._bundle_digest = self._bundle_identity()

    def _response(self, seq, stamp, frame_id, rgb_sha, mask, out):
        state = out.get("status")
        if state not in ("recovering", "tracking", "lost"):
            state = "unavailable"
        pose, render = None, None
        reason = out.get("reason")
        if state == "tracking" and out.get("pose_m") is not None:
            matrix = wc._pose(np.asarray(out["pose_m"]).tolist(), "current live pose")
            if matrix[2][3] <= 0:
                raise NativeError("pose_contract", "tracked pose is behind camera", 503)
            pose = {"object_to_camera": [list(r) for r in matrix], "units": "metres",
                    "camera_axes": "opencv_x_right_y_down_z_forward"}
            draw = getattr(self.provider, "render", None)
            registered = self.factory.config["assets"][self.asset_id]
            if callable(draw) and registered["artwork_revision"] == self.onboarding["artwork_revision"] and registered["artwork_files"]:
                try:
                    drawn = draw(frame_id=frame_id, rgb_sha256=rgb_sha, pose_m=np.asarray(matrix),
                                 observed_mask=mask.copy(), K=self.calibration.K)
                    if (drawn["frame_id"] != frame_id or drawn["rgb_sha256"] != rgb_sha or
                            drawn["glb_sha256"] != self.backend.bundle.glb_sha256 or
                            drawn["artwork_revision"] != self.onboarding["artwork_revision"]):
                        raise ValueError("render identity differs")
                    rgba = np.asarray(drawn["rgba"])
                    if rgba.dtype != np.uint8 or rgba.shape != (*mask.shape, 4) or not (rgba[..., 3] > 0).any():
                        raise ValueError("render produced no current RGBA pixels")
                    from PIL import Image
                    stream = io.BytesIO()
                    Image.fromarray(rgba, "RGBA").save(stream, format="PNG")
                    payload = stream.getvalue()
                    if len(payload) > LIMITS["max_jpeg_bytes"]:
                        raise ValueError("render exceeds response byte limit")
                    render = {"frame_id": frame_id, "rgb_sha256": rgb_sha,
                              "artwork_revision": self.onboarding["artwork_revision"],
                              "rgba_png_base64": base64.b64encode(payload).decode("ascii")}
                except Exception:
                    reason = "current pose available; artwork renderer unavailable"
        elif state == "tracking":
            state, reason = "unavailable", "backend returned tracking without a current validated pose"
        return {"schema_version": 1, "session_id": self.session_id, "generation": self.generation,
                "frame_id": frame_id, "sequence": seq, "timestamp_ns": stamp,
                "calibration_revision": self.calibration.body["calibration_revision"],
                "artwork_revision": self.onboarding["artwork_revision"], "model_head": HEAD,
                "installed_head_sha256": self.installed_head, "rgb_sha256": rgb_sha,
                "mask_sha256": wg.mask_sha256(mask), "state": state, "mask_state": "observed",
                "pose_state": state, "render_state": ("rendered" if render is not None else
                                                       "unavailable" if pose is not None else "suppressed"),
                "pose": pose, "render": render,
                "outline_segments": observed_outline(mask), "reason": reason}


def observed_outline(mask):
    """Bounded normalized contours of the actual current observed foreground.

    No boxes, projected geometry, or previous observations are substituted.
    Outline availability is independent of pose state.
    """
    if not np.asarray(mask).any():
        return []
    try:
        import cv2
        contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        segments, remaining = [], 2048
        height, width = mask.shape
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:32]:
            points = cv2.approxPolyDP(contour, 0.75, True).reshape(-1, 2)
            count = min(len(points), 511, remaining-1)
            if count < 2:
                continue
            indices = np.linspace(0, len(points)-1, count, dtype=int)
            points = points[indices].astype(float)
            points[:, 0] /= max(width-1, 1)
            points[:, 1] /= max(height-1, 1)
            segment = points.tolist()
            segment.append(segment[0])
            segments.append(segment)
            remaining -= len(segment)
            if remaining < 3:
                break
        return segments
    except Exception:
        return []


class SAM2MetricReferenceProvider:
    """Concrete experimental current-RGB SAM2 + registered-reference onboarding.

    SAM2 stays on CPU. Optical flow moves observed-foreground point prompts;
    it never produces a pose or a mask. SAM2 recomputes the observed mask from
    each actual current RGB. Lost prompt correspondences return an empty mask;
    there is no projected silhouette, old-mask reuse or automatic reacquisition.
    SIFT/PnP is used ONCE to initialize metric geometry from server references.
    All subsequent public poses are exclusively trained GoTrack estimates.
    """
    def __init__(self, *, config, calibration, asset_id, asset, onboarding, root):
        keys(config, ("sam2_checkpoint", "sam2_checkpoint_sha256", "sam2_config", "reference_features",
                      "minimum_sam_score"), "SAM2 provider config")
        self.root, self.asset, self.config = Path(root), asset, config
        self.calibration, self.asset_id = calibration, asset_id
        self.tap = np.array([[onboarding["selection_tap"]["x"], onboarding["selection_tap"]["y"]]], np.float32)
        self.previous_gray = self.points = None
        self.initialized = False
        self.predictor = None
        score = finite(config["minimum_sam_score"], "minimum_sam_score")
        if not 0 < score <= 1:
            raise NativeError("invalid_config", "SAM2 confidence threshold must be in (0,1]", 503)
        checkpoint = self._path(config["sam2_checkpoint"])
        if (str(checkpoint.relative_to(self.root).as_posix()) != wn.SAM2_CHECKPOINT_RELATIVE or
                wn.sha256_file(checkpoint) != digest(config["sam2_checkpoint_sha256"], "SAM2 checkpoint SHA")):
            raise NativeError("model_identity", "SAM2 checkpoint differs from registered local bytes", 503)
        from .quality_assets import MODELS
        if config["sam2_checkpoint_sha256"] != MODELS["sam2.1_hiera_base_plus.pt"]["sha256"].lower():
            raise NativeError("model_identity", "SAM2 must use the existing pinned base-plus checkpoint", 503)
        if config["sam2_config"] != wn.SAM2_CONFIG:
            raise NativeError("invalid_config", "SAM2 architecture differs from pinned checkpoint", 503)
        self.reference_path = self._path(config["reference_features"])
        if self.reference_path not in {self._path(p) for p in asset["references"]}:
            raise NativeError("invalid_config", "onboarding features are not server-registered metric references", 503)
        # No pickle/object arrays. The reference descriptor-to-XYZ association
        # must be measured independently and bound to the registered metric GLB.
        if self.reference_path.stat().st_size > 33_554_432:
            raise NativeError("invalid_config", "reference features exceed 32 MiB", 503)
        with zipfile.ZipFile(self.reference_path) as archive:
            entries = archive.infolist()
            sizes = {"descriptors.npy": 25_616_384, "xyz_m.npy": 616_384, "glb_sha256.npy": 16_384}
            if (len(entries) != 3 or {entry.filename for entry in entries} != set(sizes) or
                    any(entry.file_size > sizes[entry.filename] for entry in entries)):
                raise NativeError("invalid_config", "reference NPZ decompressed payload exceeds limits", 503)
        with np.load(self.reference_path, allow_pickle=False) as refs:
            if set(refs.files) != {"descriptors", "xyz_m", "glb_sha256"}:
                raise NativeError("invalid_config", "reference features require descriptors, xyz_m, glb_sha256", 503)
            self.descriptors = np.asarray(refs["descriptors"]).copy()
            self.xyz = np.asarray(refs["xyz_m"]).copy()
            self.reference_glb = str(refs["glb_sha256"].item())
        n = len(self.xyz)
        if (not 24 <= n <= 50_000 or self.xyz.shape != (n, 3) or self.descriptors.shape != (n, 128) or
                self.xyz.dtype != np.float32 or self.descriptors.dtype != np.float32 or
                not np.isfinite(self.xyz).all() or not np.isfinite(self.descriptors).all()):
            raise NativeError("invalid_config", "invalid registered SIFT-to-metric-XYZ arrays", 503)
        import torch
        import cv2
        sources = self.root/wn.SAM2_SOURCES_RELATIVE
        if str(sources) not in sys.path:
            sys.path.insert(0, str(sources))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        for module in ("sam2.build_sam", "sam2.sam2_image_predictor"):
            origin = Path(sys.modules[module].__file__).resolve()
            try:
                origin.relative_to(sources.resolve())
            except ValueError as exc:
                raise NativeError("invalid_config", "SAM2 import resolved outside pinned local source", 503) from exc
        self.torch, self.cv2 = torch, cv2
        self.predictor = SAM2ImagePredictor(build_sam2(config["sam2_config"], str(checkpoint), device="cpu",
                                                       apply_postprocessing=False))

    def _path(self, value):
        raw = Path(value)
        p = raw if raw.is_absolute() else self.root/raw
        if p.is_symlink():
            raise NativeError("invalid_config", "provider resource cannot be linked", 503)
        resolved = p.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise NativeError("invalid_config", "provider resource escapes checkout", 503) from exc
        return resolved

    def observe(self, *, frame_id, rgb, rgb_sha256, timestamp_ns):
        cv2 = self.cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        points = self.tap.copy()
        if self.previous_gray is not None:
            if self.points is None or len(self.points) < 3:
                return ObservedMask(frame_id, rgb_sha256, np.zeros(gray.shape, bool), "observed_foreground")
            moved, forward, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, self.points, None)
            if moved is None:
                return ObservedMask(frame_id, rgb_sha256, np.zeros(gray.shape, bool), "observed_foreground")
            back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None)
            if back is None:
                return ObservedMask(frame_id, rgb_sha256, np.zeros(gray.shape, bool), "observed_foreground")
            delta = np.linalg.norm((back-self.points).reshape(-1, 2), axis=1)
            p = moved.reshape(-1, 2)
            valid = ((forward.ravel() == 1) & (backward.ravel() == 1) & np.isfinite(delta) & (delta < 1.0) &
                     (p[:, 0] >= 0) & (p[:, 0] < gray.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < gray.shape[0]))
            points = p[valid]
            if len(points) < 3:
                self.points = None
                return ObservedMask(frame_id, rgb_sha256, np.zeros(gray.shape, bool), "observed_foreground")
        # SAM uses continuous corner-origin coordinates; tap/flow pixels use
        # integer centres. The +0.5 convention matches existing segmentation.
        with self.torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(point_coords=points+0.5,
                                                      point_labels=np.ones(len(points), np.int32),
                                                      multimask_output=False)
        mask = np.asarray(masks[0], dtype=bool)
        if (mask.shape != gray.shape or not np.isfinite(scores).all() or
                float(scores[0]) < self.config["minimum_sam_score"]):
            mask = np.zeros(gray.shape, bool)
        self.previous_gray = gray
        self.points = cv2.goodFeaturesToTrack(gray, maxCorners=24, qualityLevel=0.02, minDistance=8,
                                              mask=mask.astype(np.uint8)*255)
        return ObservedMask(frame_id, rgb_sha256, mask, "observed_foreground")

    def initialize(self, *, frame_id, rgb, rgb_sha256, observed_mask, K, selection_tap, bundle):
        if self.initialized:
            raise NativeError("initialization_unavailable", "metric onboarding is single-use", 503)
        self.initialized = True
        if self.reference_glb != bundle.glb_sha256:
            raise NativeError("initialization_binding", "reference XYZ uses another metric object", 503)
        cv2 = self.cv2
        keypoints, descriptors = cv2.SIFT_create(nfeatures=2048).detectAndCompute(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), observed_mask.astype(np.uint8)*255)
        if descriptors is None or len(keypoints) < 24:
            raise NativeError("initialization_unavailable", "not enough observed reference correspondences", 503)
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, self.descriptors, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7*p[1].distance]
        # Multiple image points must not vote for the same registered XYZ.
        unique = {}
        for match in sorted(matches, key=lambda v: v.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 24:
            raise NativeError("initialization_unavailable", "metric reference matching insufficient", 503)
        xyz = self.xyz[[m.trainIdx for m in matches]].astype(np.float64)
        pixels = np.array([keypoints[m.queryIdx].pt for m in matches], np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, pixels, K, None, iterationsCount=1000,
                                                    reprojectionError=3.0, confidence=0.999,
                                                    flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < 24:
            raise NativeError("initialization_unavailable", "onboarding PnP lacks 24 current inliers", 503)
        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, tvec.ravel()
        camera_points = xyz[inliers.ravel()]@rotation.T+tvec.ravel()
        if not np.isfinite(pose).all() or not (camera_points[:, 2] > 0).all():
            raise NativeError("initialization_unavailable", "onboarding points fail cheirality", 503)
        return LiveInitialization(frame_id, rgb_sha256, bundle.glb_sha256, pose,
                                  "onboarding:current_observed_RGB_registered_metric_SIFT_PnP")

    def close(self):
        self.predictor = None
        self.previous_gray = self.points = None
        gc.collect()


def sam2_metric_reference_provider(**kwargs):
    """Operator entrypoint: bench.workbench_live_pose:sam2_metric_reference_provider."""
    return SAM2MetricReferenceProvider(**kwargs)
