"""RAM-only CPU artwork replay from a SAVED pose (no new pose, no neural inference, no files written).

For a finalized run whose neural artwork PNG is genuinely absent (for example an atomic publication
denied by the OS), this module can rebuild the frozen ``NeuralArtwork`` and re-render the saved
metric pose with the frame's current observed mask, entirely in memory. The result is labelled
``cpu_replay_of_saved_pose``. It is NOT the original artwork file: the historical PNG bytes were
never stored and cannot be verified. It is not new tracking and makes no quality claim.

Authenticated before any render (fail closed; nothing is ever silently substituted):

* a finalized run: report present with a terminal status; bounded ``run.json``, ``results.jsonl``,
  ``report.json`` and ``neural-artwork.json``; exactly one row per (role, scheduled frame);
* the requested row: a neural role with ``tracking`` + ``rendered`` + ``metric`` units, a valid
  saved pose, an observed mask of the same frame and recorded ``render_stats``. Its original PNG must
  be genuinely absent (``artifact_write_error`` or ``overlay_artifact`` and no file on disk). A
  recorded PNG always takes precedence, and a recorded-but-missing PNG is refused, never replaced;
* the producer's computational source closure: report ``dependencies.sources`` (cli, neural,
  geometry, contract, image_io), the role stage's boundary hash, version and path, and the parsed
  admission ``contract.json`` metadata. That covers the contract self-hash, the row's admission
  contract and ``source_closure.current_files``, including render and boundary. All are compared with
  the current files. ``load_admission`` is never called;
* the pinned identities: primary installed/export head 27e06…, base f7d1…, baseline with no
  installed/export head;
* the server-registered metric bundle (never a browser path): bundle file SHA, then
  ``MetricBundle.from_file`` (GLB re-hash). The recomputed ``neural_binding`` / ``FrameContextV1``
  binding hash, context digest and boundary provenance must equal the role stage's ``execution``;
* the masks: the recorded mask-stage identity equals the current manifest, and the requested and
  selection masks equal the row and the artwork record;
* the source frame: registered media verified, and the exact frame decoded through ``FrameReader``
  with the row's image SHA;
* the artwork: a fresh ``NeuralArtwork`` whose complete canonical record and hash (effective K,
  anchor, marker) equal ``neural-artwork.json``;
* the render: every recorded ``render_stats`` key and value, uint8 RGBA dimensions, binary alpha,
  the alpha count and observed-mask containment.

All input and source closures are re-verified after the render and before responding. One request
renders at a time (non-blocking ``busy``). There is no memoization or cache, and nothing is written.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from . import workbench_contract as wc

SCHEMA = "workbench_artwork_ram_replay_v1"
SOURCES_SCHEMA = "workbench_artwork_sources_v1"
ROOT = Path(__file__).resolve().parents[1]
NEURAL_ROLES = ("primary_neural", "frozen_baseline")
TERMINAL_STATUSES = ("completed", "completed_with_cleanup_errors", "failed", "blocked", "cancelled")
PRIMARY_HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
BASE_SHA256 = "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
LABEL = "CPU replay from saved pose; original artwork file unavailable. Historical bytes cannot be verified."
MAX_EDGE = 1280
MAX_PIXELS = 2_000_000
MAX_FACES = 200_000
MAX_GLB_BYTES = 32 << 20
MAX_RESULTS_BYTES = 16 << 20
MAX_SMALL_JSON_BYTES = 4 << 20
MAX_CONTRACT_BYTES = 16 << 20
MAX_PNG_BYTES = 2 << 20
MAX_RESPONSE_BYTES = 4 << 20
RAM_ALLOWANCE_ESTIMATE = 512 << 20
PRODUCER_MODULES = ("workbench_cli", "workbench_neural", "workbench_geometry", "workbench_contract", "workbench_image_io")
CONTRACT_CLOSURE_FILES = ("bench/workbench_cli.py", "bench/workbench_neural.py", "bench/workbench_geometry.py",
                          "bench/workbench_contract.py", "bench/workbench_render.py", "bench/workbench_frame_boundary.py")
REPLAY_SOURCES = ("bench/workbench_artwork_replay.py", "bench/workbench_neural.py", "bench/workbench_geometry.py",
                  "bench/workbench_image_io.py", "bench/workbench_contract.py", "bench/workbench_cli.py",
                  "bench/workbench_frame_boundary.py")
HTTP = {"ok": 200, "busy": 429, "not_ready": 409, "original_available": 409, "refused": 422, "not_found": 404}
FLAGS = {"source": "cpu_replay_of_saved_pose", "historical_png_bytes_verified": False, "new_pose_estimation": False,
         "new_neural_inference": False, "disk_files_written": False, "quality_claim": False}


class Refused(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _sha_file(path: Path, limit: int | None = None) -> str:
    """Streaming SHA-256. With ``limit``: early stat check, then a CUMULATIVE read bound. Each read
    asks for at most the remaining allowance plus one detection byte, and growth past the cap is
    refused immediately (no digest is returned). ``limit=None`` is an ordinary streaming file hash."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Refused("refused", f"{path.name} is missing or not a regular file")
    if limit is not None and path.stat().st_size > limit:
        raise Refused("refused", f"{path.name} exceeds its {limit}-byte bound")
    h, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            want = (1 << 20) if limit is None else min(1 << 20, limit - total + 1)
            block = stream.read(want)
            if not block:
                break
            total += len(block)
            if limit is not None and total > limit:
                raise Refused("refused", f"{path.name} grew beyond its {limit}-byte bound while hashing")
            h.update(block)
    return h.hexdigest()


def _read_bounded(path: Path, limit: int) -> bytes:
    """Early stat check, then read AT MOST ``limit + 1`` bytes (never an unbounded allocation)."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Refused("not_ready", f"{path.name} is not present")
    if path.stat().st_size > limit:
        raise Refused("refused", f"{path.name} exceeds its {limit}-byte bound")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise Refused("refused", f"{path.name} grew beyond its bound")
    return data


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


COUNT_STATS = ("rendered_pixels", "mask_rejected_pixels", "overlap_rejected_pixels", "backfacing_rejected_pixels",
               "patch_faces", "folded_faces")


def _exact_count(value: Any) -> bool:
    """A recorded count is an exact non-negative ``int`` (``bool`` is never a count)."""
    return type(value) is int and value >= 0


def recorded_stats_problem(stats: Any) -> str | None:
    """Early type validation of recorded render_stats; ``None`` when acceptable."""
    if not isinstance(stats, Mapping):
        return "no recorded render_stats"
    if type(stats.get("rendered_pixels")) is not int or stats["rendered_pixels"] <= 0:
        return "recorded rendered_pixels is not a positive integer"
    bad = [k for k in COUNT_STATS if k in stats and not _exact_count(stats[k])]
    if bad:
        return f"recorded render_stats counts are not exact non-negative integers: {bad}"
    if "reason" in stats and stats["reason"] is not None and not isinstance(stats["reason"], str):
        return "recorded render_stats reason is not a string or null"
    return None


def eligibility(row: wc.FrameResult, run_dir: Path) -> tuple[str, str]:
    """Static eligibility of one row (no authentication): (state, reason). States:
    ``eligible`` | ``original_recorded`` | ``original_missing`` | ``ineligible``."""
    if row.backend_role not in NEURAL_ROLES:
        return "ineligible", "only primary_neural / frozen_baseline rows are replayable"
    if (row.pose_state != "tracking" or row.render_state != "rendered" or row.pose_units != "metric" or row.pose is None
            or row.mask_state != "observed" or row.mask_frame_id != row.frame_id or row.mask_sha256 is None):
        return "ineligible", f"row is {row.pose_state}/{row.render_state}; only tracking+rendered metric rows render"
    evidence = row.evidence or {}
    problem = recorded_stats_problem(evidence.get("render_stats"))
    if problem is not None:
        return "ineligible", problem
    expected = f"evidence/{row.backend_role}/{row.frame_id}.png"
    if evidence.get("overlay") is not None:
        if evidence["overlay"] != expected:
            return "ineligible", "recorded overlay path is not this role/frame's"
        return ("original_recorded", "the original PNG is recorded; it takes precedence") if (Path(run_dir) / expected).is_file() \
            else ("original_missing", "the recorded original PNG is missing; it is never silently replaced")
    if "artifact_write_error" not in evidence and "overlay_artifact" not in evidence:
        return "ineligible", "no recorded reason for the absent original PNG"
    if (Path(run_dir) / expected).exists():
        return "ineligible", "an unrecorded file exists at the original path; refusing an ambiguous artifact"
    return "eligible", str(evidence.get("artifact_write_error") or evidence.get("overlay_artifact"))[:300]


def artwork_sources(run_id: str, rows: list[wc.FrameResult], run_dir: Path) -> dict:
    """Small descriptor adjacent to run_detail (FrameResult schema unchanged)."""
    out = []
    for row in rows:
        if row.backend_role in NEURAL_ROLES and row.render_state == "rendered":
            state, reason = eligibility(row, run_dir)
            out.append({"role": row.backend_role, "frame_id": row.frame_id, "original_png": state if state.startswith("original")
                        else "not_recorded", "ram_replay": "eligible" if state == "eligible" else "ineligible",
                        "reason": reason})
    return {"schema": SOURCES_SCHEMA, "label": LABEL,
            "replay_route": f"/api/runs/{run_id}/frames/{{frame_id}}/artwork-replay?role={{role}}",
            "note": "eligibility is static; the replay endpoint re-authenticates everything before rendering",
            "rows": out}


class ArtworkRamReplay:
    """One bounded CPU replay at a time; never writes, caches or loads a model."""

    def __init__(self, workspace: Any, *, bundle_resolver: Callable[[str], tuple[Path, str] | None],
                 admission_dir: Path | None, repo_root: Path = ROOT):
        self.ws, self.bundle_resolver, self.repo_root = workspace, bundle_resolver, Path(repo_root)
        self.admission_dir = None if admission_dir is None else Path(admission_dir)
        self._lock = threading.Lock()

    # --------------------------------------------------------------- public
    def replay(self, run_id: str, frame_id: str, role: str) -> tuple[int, dict]:
        base = {"schema": SCHEMA, "run_id": run_id, "frame_id": frame_id, "role": role, **FLAGS}
        if not self._lock.acquire(blocking=False):
            return HTTP["busy"], {**base, "status": "busy", "reason": "another artwork replay is rendering; retry shortly"}
        try:
            body = self._replay(run_id, frame_id, role)
            return HTTP["ok"], {**base, **body, "status": "ok"}
        except Refused as exc:
            return HTTP.get(exc.status, 422), {**base, "status": exc.status, "reason": exc.reason[:1000]}
        except Exception as exc:  # noqa: BLE001 - every other failure is a typed refusal, never a substitute
            return HTTP["refused"], {**base, "status": "refused", "reason": f"{type(exc).__name__}: {exc}"[:1000]}
        finally:
            self._lock.release()

    # --------------------------------------------------------------- internals
    def _source_hashes(self, rels) -> dict:
        return {rel: _sha_file(self.repo_root / rel) for rel in rels}

    def _load_run(self, run_id: str) -> dict:
        wc._ident(run_id, "run_id")
        run_dir = self.ws.runs_dir / run_id
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise Refused("not_found", "run is not registered")
        report_raw = _read_bounded(run_dir / "report.json", MAX_SMALL_JSON_BYTES)
        report = wc.loads_strict(report_raw)
        if report.get("status") not in TERMINAL_STATUSES:
            raise Refused("not_ready", "run is not finalized")
        run_raw = _read_bounded(run_dir / "run.json", MAX_SMALL_JSON_BYTES)
        results_raw = _read_bounded(run_dir / "results.jsonl", MAX_RESULTS_BYTES)
        art_raw = _read_bounded(run_dir / "neural-artwork.json", MAX_SMALL_JSON_BYTES)
        run_doc = wc.loads_strict(run_raw)
        run = wc.RunSpec.from_dict(run_doc["run"])
        rows = [wc.FrameResult.from_dict(wc.loads_strict(line)) for line in results_raw.splitlines() if line.strip()]
        if run.run_id != run_id or report.get("run_id") != run_id:
            raise Refused("refused", "run identity differs between run.json and report.json")
        if report.get("clip_id") != run.clip_id or run_doc.get("clip_id") != run.clip_id:
            raise Refused("refused", "clip identity differs between run.json, its RunSpec and report.json")
        table: dict[tuple[str, str], wc.FrameResult] = {}
        for row in rows:
            k = (row.backend_role, row.frame_id)
            if (row.run_id != run_id or row.clip_id != run.clip_id or row.backend_role not in run.roles
                    or row.frame_id not in run.scheduled_frame_ids or k in table):
                raise Refused("refused", "results are not exactly one row of this run/clip per scheduled role/frame")
            table[k] = row
        if len(table) != len(run.roles) * len(run.scheduled_frame_ids):
            raise Refused("refused", "results do not account for every scheduled role/frame")
        return {"dir": run_dir, "report": report, "run": run, "table": table,
                "files": {"report.json": hashlib.sha256(report_raw).hexdigest(), "run.json": hashlib.sha256(run_raw).hexdigest(),
                          "results.jsonl": hashlib.sha256(results_raw).hexdigest(),
                          "neural-artwork.json": hashlib.sha256(art_raw).hexdigest()},
                "artwork_record": wc.loads_strict(art_raw)}

    def _admission_contract(self, expected_sha: Any) -> dict:
        if self.admission_dir is None:
            raise Refused("not_ready", "no admission directory is configured; the producer render source cannot be authenticated")
        raw = _read_bounded(self.admission_dir / "contract.json", MAX_CONTRACT_BYTES)
        body = wc.loads_strict(raw)
        stored = body.pop("contract_sha256", None) if isinstance(body, dict) else None
        if not isinstance(body, dict) or stored is None or wc.canonical_sha256(body) != stored:
            raise Refused("refused", "admission contract metadata self-hash differs")
        if stored != expected_sha:
            raise Refused("refused", "the row's admission contract is not the configured admission contract")
        current = ((body.get("source_closure") or {}).get("current_files") or {})
        return {"contract_sha256": stored, "contract_file_sha256": hashlib.sha256(raw).hexdigest(),
                "closure": {rel: current.get(rel) for rel in CONTRACT_CLOSURE_FILES}}

    def _producer_sources(self, report: dict, stage: dict, contract: dict) -> dict:
        from . import workbench_cli as cli
        from . import workbench_frame_boundary as fb
        deps = ((report.get("dependencies") or {}).get("sources") or {})
        current = {m: _sha_file(self.repo_root / "bench" / f"{m}.py") for m in PRODUCER_MODULES}
        if any(deps.get(m) != current[m] for m in PRODUCER_MODULES):
            raise Refused("refused", "producer source closure (report dependencies) differs from the current files")
        execution = stage.get("execution") or {}
        boundary_now = _sha_file(self.repo_root / "bench/workbench_frame_boundary.py")
        if (execution.get("boundary_source_sha256") != boundary_now or execution.get("boundary_version") != fb.BOUNDARY_VERSION
                or execution.get("execution_path") != cli.ADOPTED_EXECUTION_PATH):
            raise Refused("refused", "producer boundary source/version/path differs from the current files")
        closure_now = self._source_hashes(CONTRACT_CLOSURE_FILES)
        if contract["closure"] != closure_now:
            raise Refused("refused", "admission contract source closure (incl. render/boundary) differs from the current files")
        return {"report_dependencies": {m: deps[m] for m in PRODUCER_MODULES}, "boundary_source_sha256": boundary_now,
                "admission_closure": closure_now}

    @staticmethod
    def _identity_checks(role: str, ident: dict, bundle: Any, binding: dict, effective_K: list) -> None:
        if ident.get("base_checkpoint_sha256") != BASE_SHA256 or ident.get("quality_claim") is not False:
            raise Refused("refused", "row base/claim identity differs from the pins")
        if role == "primary_neural" and (ident.get("installed_head_sha256") != PRIMARY_HEAD_SHA256
                                         or ident.get("export_sha256") != PRIMARY_HEAD_SHA256):
            raise Refused("refused", "primary row is not the pinned installed/export head")
        if role == "frozen_baseline" and (ident.get("installed_head_sha256") is not None or ident.get("export_sha256") is not None
                                          or ident.get("head") != "frozen_base_pose_head"):
            raise Refused("refused", "baseline row carries an installed/export head")
        expected = {"principal_point_origin": bundle.principal_point_origin, "initialization": bundle.seed_label,
                    "seed_frame_id": bundle.seed_frame_id, "seed_target_frame_id": binding["seed_target_frame_id"],
                    "glb_sha256": bundle.glb_sha256, "metric_units": "metres"}
        if any(k not in ident or ident[k] != v for k, v in expected.items()):
            raise Refused("refused", "row bundle/seed/origin/units identity differs from the registered bundle")
        K = ident.get("effective_K")
        if (not isinstance(K, list) or len(K) != 9 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                           or not np.isfinite(float(x)) for x in K)
                or [float(x) for x in K] != [float(x) for x in effective_K]):
            raise Refused("refused", "row effective K differs from the current calibration/origin")

    def _bundle(self, clip_id: str):
        from . import workbench_neural as wn
        resolved = self.bundle_resolver(clip_id)
        if resolved is None:
            raise Refused("not_ready", "no server-registered metric bundle for this clip")
        path, sha = resolved
        if _sha_file(path, MAX_SMALL_JSON_BYTES) != sha:
            raise Refused("refused", "registered metric bundle file changed")
        doc = wc.loads_strict(_read_bounded(path, MAX_SMALL_JSON_BYTES))
        glb = Path(doc["glb_path"])
        glb = glb if glb.is_absolute() else self.repo_root / glb
        if glb.is_symlink() or not glb.is_file() or glb.stat().st_size > MAX_GLB_BYTES:
            raise Refused("refused", "bundle GLB is missing or exceeds the 32 MiB replay bound")
        bundle = wn.MetricBundle.from_file(path, root=self.repo_root)   # re-hashes the GLB
        return bundle, {"bundle_sha256": sha, "glb_path": glb, "glb_sha256": bundle.glb_sha256}

    def _replay(self, run_id: str, frame_id: str, role: str) -> dict:
        from . import workbench_cli as cli
        from . import workbench_frame_boundary as fb
        from . import workbench_image_io as wio
        from . import workbench_neural as wn
        if role not in NEURAL_ROLES:
            raise Refused("refused", "only primary_neural / frozen_baseline artwork can be replayed")
        wc._ident(frame_id, "frame_id")
        state = self._load_run(run_id)
        run, report, run_dir = state["run"], state["report"], state["dir"]
        row = state["table"].get((role, frame_id))
        if row is None:
            raise Refused("not_found", "no result row for this role/frame")
        verdict, why = eligibility(row, run_dir)
        if verdict == "original_recorded":
            raise Refused("original_available", why)
        if verdict != "eligible":
            raise Refused("refused", why)
        stage = (report.get("stages") or {}).get(role)
        if not isinstance(stage, dict) or stage.get("status") not in ("completed", "cache_hit"):
            raise Refused("refused", "role stage did not complete")
        clip = self.ws.load_clip(run.clip_id)
        if clip.clip_id != run.clip_id or row.clip_id != clip.clip_id:
            raise Refused("refused", "row/run clip identity differs from the registered clip")
        try:
            manifest_frame = clip.frame(frame_id)
        except KeyError as exc:
            raise Refused("refused", "requested frame is not in the current clip manifest") from exc
        # Exact FrameSpec metadata, before any artwork construction or render (a decoded image hash is
        # not a substitute for the row's frame identity).
        for name in ("frame_id", "source_frame_index", "timestamp_s", "media_time_s"):
            if getattr(row, name) != getattr(manifest_frame, name) or type(getattr(row, name)) is not type(getattr(manifest_frame, name)):
                raise Refused("refused", f"row {name} differs from the current clip manifest")
        if clip.width > MAX_EDGE or clip.height > MAX_EDGE or clip.width * clip.height > MAX_PIXELS:
            raise Refused("refused", "frame dimensions exceed the replay bounds (1280 edge, 2 MP)")
        if clip.intrinsics.K is None:
            raise Refused("refused", "clip calibration is unavailable")
        clip_sha = _sha_file(self.ws.clip_path(clip.clip_id), MAX_SMALL_JSON_BYTES)
        contract = self._admission_contract(row.backend_identity.get("admission_contract_sha256"))
        producer = self._producer_sources(report, stage, contract)
        replay_before = self._source_hashes(REPLAY_SOURCES)
        bundle, bundle_files = self._bundle(clip.clip_id)
        binding = fb.neural_binding(clip, run, family=bundle.family, bundle=bundle)
        context = fb.FrameContextV1.build(clip, run, role, binding)
        provenance = cli.boundary_provenance(context)
        execution = stage.get("execution") or {}
        if any(execution.get(k) != provenance[k] for k in provenance):
            raise Refused("refused", "recomputed binding/context/boundary provenance differs from the role stage execution")
        effective_K = wn.native_K(np.asarray(clip.intrinsics.K, float).ravel(), bundle.principal_point_origin).ravel().tolist()
        self._identity_checks(role, row.backend_identity, bundle, binding, effective_K)
        mask_set = (run.config or {}).get("mask_set")
        store = cli.MaskStore(self.ws, clip.clip_id, mask_set)
        recorded_masks = ((report.get("stages") or {}).get("masks") or {}).get("identity")
        if not store.available or store.identity() != recorded_masks:
            raise Refused("refused", "current observed-mask manifest differs from the recorded mask stage")
        mask, mident = store.load(frame_id, (clip.height, clip.width))
        if mask is None or mident["mask_frame_id"] != frame_id or mident["mask_sha256"] != row.mask_sha256:
            raise Refused("refused", "current observed mask differs from the row's mask identity")
        selection = run.onboarding.selection
        sel_mask, _ = store.load(selection.frame_id, (clip.height, clip.width))
        mask_files = store.verify([frame_id, selection.frame_id])
        reader = cli.FrameReader(self.ws, clip)
        try:
            media = reader.verify_media()
            frame_files = reader.verify_frames([frame_id])
            _rgb, ident = reader.read(clip.frame(frame_id))
        finally:
            reader.close()
        if ident.get("image_sha256") != row.image_sha256:
            raise Refused("refused", "decoded source frame differs from the row's image identity")
        artwork = wn.NeuralArtwork(bundle, selection, run.artwork_kind, clip_K=clip.intrinsics.K, width=clip.width,
                                   height=clip.height, selection_mask=sel_mask, max_faces=MAX_FACES)
        saved = dict(state["artwork_record"])
        saved_sha = saved.pop("record_sha256", None)
        if (artwork.sha256 != saved_sha or wc.canonical_sha256(_plain(artwork.record)) != saved_sha
                or wc.canonical_json(_plain(artwork.record)) != wc.canonical_json(saved)
                or [float(x) for x in artwork.record["effective_K"]] != [float(x) for x in effective_K]):
            raise Refused("refused", "rebuilt artwork record (anchor/marker/effective K) differs from neural-artwork.json")
        drawn = artwork.render(np.asarray(row.pose, dtype=np.float64), np.asarray(mask) > 0)
        rgba, stats = drawn["rgba"], _plain(dict(drawn["stats"]))
        if stats.get("reason") is None and stats.get("rendered_pixels") == 0:
            stats["reason"] = "no visible supported artwork pixels"
        recorded_stats = row.evidence["render_stats"]
        # Type-exact: canonical JSON distinguishes False from 0 and 1.0 from 1; every computed count is an int.
        if (recorded_stats_problem(stats) is not None
                or wc.canonical_json(stats) != wc.canonical_json(_plain(recorded_stats))):
            raise Refused("refused", "re-rendered statistics differ (value or type) from the recorded render_stats")
        if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (clip.height, clip.width, 4):
            raise Refused("refused", "re-rendered RGBA has the wrong dtype or dimensions")
        alpha = rgba[:, :, 3]
        if not np.isin(alpha, (0, 255)).all() or int((alpha > 0).sum()) != stats["rendered_pixels"] \
                or bool(((alpha > 0) & ~(np.asarray(mask) > 0)).any()):
            raise Refused("refused", "re-rendered alpha is not binary, does not match the count or leaves the observed mask")
        png = wio.encode_png_rgba(rgba)
        if len(png) > MAX_PNG_BYTES:
            raise Refused("refused", "encoded PNG exceeds 2 MiB")
        # Re-verify every input and source closure after the render and before responding.
        after = self._load_run(run_id)
        fresh_store = cli.MaskStore(self.ws, clip.clip_id, mask_set)   # never the earlier cached manifest
        if (after["files"] != state["files"] or _sha_file(self.ws.clip_path(clip.clip_id), MAX_SMALL_JSON_BYTES) != clip_sha
                or self._source_hashes(REPLAY_SOURCES) != replay_before
                or self._producer_sources(after["report"], (after["report"].get("stages") or {}).get(role) or {}, contract) != producer
                or self._admission_contract(row.backend_identity.get("admission_contract_sha256")) != contract
                or _sha_file(self.bundle_resolver(clip.clip_id)[0], MAX_SMALL_JSON_BYTES) != bundle_files["bundle_sha256"]
                or _sha_file(bundle_files["glb_path"], MAX_GLB_BYTES) != bundle_files["glb_sha256"]
                or not fresh_store.available or fresh_store.identity() != recorded_masks
                or fresh_store.verify([frame_id, selection.frame_id]) != mask_files):
            raise Refused("refused", "inputs or sources changed during the replay; nothing is returned")
        reader = cli.FrameReader(self.ws, clip)
        try:
            if reader.verify_media() != media or reader.verify_frames([frame_id]) != frame_files:
                raise Refused("refused", "source media changed during the replay")
        finally:
            reader.close()
        # Immediately before responding: a newly appearing original or ambiguous artifact always wins.
        final_row = after["table"].get((role, frame_id))
        final_verdict, final_why = eligibility(final_row, run_dir) if final_row is not None else ("ineligible", "row vanished")
        if final_verdict == "original_recorded":
            raise Refused("original_available", final_why)
        if final_verdict != "eligible" or final_why != why:
            raise Refused("refused", f"original-artifact eligibility changed during the replay: {final_why}")
        body = {"label": LABEL, "original_run_status": report.get("status"),
                "original_artifact": {"recorded": False, "reason": why},
                "png": {"sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png), "width": clip.width,
                        "height": clip.height, "base64": base64.b64encode(png).decode("ascii")},
                "render_stats": stats,
                "provenance": {"results_sha256": state["files"]["results.jsonl"], "report_sha256": state["files"]["report.json"],
                               "run_json_sha256": state["files"]["run.json"],
                               "artwork_record_sha256": saved_sha, "artwork_file_sha256": state["files"]["neural-artwork.json"],
                               "binding_sha256": context.binding_sha256, "context_digest": provenance["context_digest"],
                               "mask_manifest_sha256": recorded_masks.get("manifest_sha256"),
                               "mask_sha256": row.mask_sha256, "image_sha256": row.image_sha256,
                               "bundle_sha256": bundle_files["bundle_sha256"], "glb_sha256": bundle_files["glb_sha256"],
                               "admission_contract_sha256": contract["contract_sha256"],
                               "producer_sources": producer, "replay_sources": replay_before},
                "limits": {"ram_working_allowance_estimate_bytes": RAM_ALLOWANCE_ESTIMATE,
                           "note": "estimate, not a measured peak; one replay at a time; no cache"}}
        if len(wc.canonical_json(body)) > MAX_RESPONSE_BYTES:
            raise Refused("refused", "response exceeds 4 MiB")
        return body
