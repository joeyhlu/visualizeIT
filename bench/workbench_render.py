"""Workbench zero-sample-storage renderer adapter (``workbench_zero_sample_storage_v1``).

Numerical repair for the serving profile: pyrender's main framebuffer requests
4x multisample colour/depth renderbuffers and resolves depth by blit, which
shifts template depth by an MSAA sample offset even with ``GL_MULTISAMPLE``
disabled. This adapter wraps the *unchanged* legacy ``TexturedRenderer``
(``legacy_v1`` + ``integer_centers_v1``, unlit, no multisampling) and, inside a
scoped hook, routes exactly those two 4x allocations to ordinary single-sample
storage. It then verifies the actual GL state:

* exactly one redirected colour (GL_RGBA) and depth (GL_DEPTH_COMPONENT24)
  allocation plus the ordinary resolve pair, at the requested dimensions;
* draw/read FBO attachment identities, completeness, viewport, internal
  formats and ``GL_RENDERBUFFER_SAMPLES == 0`` / ``GL_SAMPLES == 0``;
* on reuse, the same offscreen, FBOs and attachments; any unknown
  reallocation or attachment mutation is rejected.

Bounds are its own (each edge <= 1280, <= 2,000,000 pixels, so native
1024x1280 and 280x280 templates fit); the historical capture guard (1120) and
all historical pins are untouched. Rendering requires the repository GPU stage
and ``quality_render_stability._render_lock``; hooks, GL bindings and the
current context are restored on every path, and a failed renderer is disposed
and poisoned with no fallback to the legacy multisample depth.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import operator
from pathlib import Path
from typing import Any, Iterator, Mapping

POLICY = "workbench_zero_sample_storage_v1"
MAX_EDGE = 1280
MAX_PIXELS = 2_000_000
ROOT = Path(__file__).resolve().parents[1]

# Pinned sources this adapter relies on (no unpinned source may be used).
PINNED_SOURCES = {
    "bench/quality_gotrack.py": "0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29",
    "bench/quality_render_stability.py": "66f907f0c6c3f9ed32c6edbedb996a9c45e5ab1aea1d1914f99026a7f725c6bd",
    "pyrender/renderer.py": "364c16c98487e808b3124bdaa652ca8592bbf200cc63b90049057912c1c8f1b6",
    "pyrender/offscreen.py": "dab44974b08eeee46774653714524a500ff49bdc59cb0657f9720e00d21d3099",
}


class RenderPolicyError(RuntimeError):
    """The zero-storage policy could not be established or verified."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gl_int(value: Any) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple, bool)):
        raise RenderPolicyError("OpenGL integer query did not return one scalar")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderPolicyError("OpenGL integer query is not representable") from exc


def _gl_ints(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [_gl_int(v) for v in value]


def check_dimensions(width: Any, height: Any) -> tuple[int, int]:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise RenderPolicyError("viewport dimensions must be positive integers")
    if width > MAX_EDGE or height > MAX_EDGE or width * height > MAX_PIXELS:
        raise RenderPolicyError(f"viewport {width}x{height} exceeds the {POLICY} bounds")
    return width, height


def pinned_source_paths() -> dict[str, Path | None]:
    """Locate pinned files without importing pyrender/OpenGL."""
    spec = importlib.util.find_spec("pyrender")
    pyrender_dir = Path(spec.origin).parent if spec is not None and spec.origin else None
    return {
        "bench/quality_gotrack.py": ROOT / "bench/quality_gotrack.py",
        "bench/quality_render_stability.py": ROOT / "bench/quality_render_stability.py",
        "pyrender/renderer.py": None if pyrender_dir is None else pyrender_dir / "renderer.py",
        "pyrender/offscreen.py": None if pyrender_dir is None else pyrender_dir / "offscreen.py",
    }


def factory_identity(*, require_pins: bool = False) -> dict:
    """Static policy/source identity bound into contracts, parity, geometry and caches."""
    actual = {}
    for key, path in pinned_source_paths().items():
        actual[key] = _sha(path) if path is not None and path.is_file() else None
    mismatched = sorted(k for k, v in actual.items() if v != PINNED_SOURCES[k])
    if require_pins and mismatched:
        raise RenderPolicyError(f"render sources differ from their pins: {mismatched}")
    return {"policy": POLICY, "bounds": {"max_edge": MAX_EDGE, "max_pixels": MAX_PIXELS},
            "adapter_sha256": _sha(Path(__file__)), "pinned_sources": dict(PINNED_SOURCES),
            "actual_sources": actual, "pins_match": not mismatched, "legacy_inner_policy": "legacy_v1",
            "coordinate_mode": "integer_centers_v1",
            "storage": "ordinary single-sample GL_RGBA + GL_DEPTH_COMPONENT24 (4x requests redirected)"}


# --------------------------------------------------------------------------
# scoped allocation hook

@contextmanager
def zero_storage_scope(renderer_module: Any, gl: Any, *, dimensions: tuple[int, int],
                       log: list) -> Iterator[None]:
    """Route pyrender's two 4x requests to ordinary storage; log every allocation.
    Both module functions are restored on every path."""
    width, height = check_dimensions(*dimensions)
    original_ms = getattr(renderer_module, "glRenderbufferStorageMultisample", None)
    original_ordinary = getattr(renderer_module, "glRenderbufferStorage", None)
    if not callable(original_ms) or not callable(original_ordinary):
        raise RenderPolicyError("pinned renderbuffer storage functions are unavailable")

    def describe(kind, target, samples, internalformat, w, h):
        row = {"kind": kind, "target": None, "samples": None, "format": None, "width": None, "height": None,
               "renderbuffer_id": None, "success": False}
        log.append(row)
        row.update(target=_gl_int(target), samples=samples if samples is None else _gl_int(samples),
                   format=_gl_int(internalformat), width=_gl_int(w), height=_gl_int(h))
        if row["target"] != int(gl.GL_RENDERBUFFER):
            raise RenderPolicyError("allocation used an unexpected target")
        if row["format"] not in (int(gl.GL_RGBA), int(gl.GL_DEPTH_COMPONENT24)):
            raise RenderPolicyError("allocation used an unexpected internal format")
        if (row["width"], row["height"]) != (width, height):
            raise RenderPolicyError("allocation dimensions differ from the requested viewport")
        row["renderbuffer_id"] = _gl_int(gl.glGetIntegerv(gl.GL_RENDERBUFFER_BINDING))
        if row["renderbuffer_id"] <= 0:
            raise RenderPolicyError("allocation has no bound renderbuffer identity")
        return row

    def redirected(target, samples, internalformat, w, h):
        row = describe("redirected_multisample", target, samples, internalformat, w, h)
        if row["samples"] != 4:
            raise RenderPolicyError("multisample request did not use the pinned 4 samples")
        original_ordinary(target, internalformat, w, h)
        row["success"] = True

    def ordinary(target, internalformat, w, h):
        row = describe("ordinary_resolve", target, None, internalformat, w, h)
        original_ordinary(target, internalformat, w, h)
        row["success"] = True

    renderer_module.glRenderbufferStorageMultisample = redirected
    renderer_module.glRenderbufferStorage = ordinary
    try:
        yield
    finally:
        renderer_module.glRenderbufferStorageMultisample = original_ms
        renderer_module.glRenderbufferStorage = original_ordinary


def validate_allocations(log: list, gl: Any, dimensions: tuple[int, int]) -> dict:
    """Exactly one redirected colour/depth pair and one ordinary resolve pair."""
    width, height = check_dimensions(*dimensions)
    if any(not isinstance(r, dict) or r.get("success") is not True for r in log):
        raise RenderPolicyError("an allocation failed or was malformed")
    out = {}
    for kind in ("redirected_multisample", "ordinary_resolve"):
        rows = [r for r in log if r["kind"] == kind]
        formats = sorted(r["format"] for r in rows)
        if len(rows) != 2 or formats != sorted([int(gl.GL_RGBA), int(gl.GL_DEPTH_COMPONENT24)]):
            raise RenderPolicyError(f"expected one colour and one depth {kind} allocation, observed {len(rows)}")
        if rows[0]["renderbuffer_id"] == rows[1]["renderbuffer_id"]:
            raise RenderPolicyError(f"{kind} colour and depth share a renderbuffer")
        if any((r["width"], r["height"]) != (width, height) for r in rows):
            raise RenderPolicyError(f"{kind} allocation dimensions differ")
        out[kind] = {"color": next(r for r in rows if r["format"] == int(gl.GL_RGBA))["renderbuffer_id"],
                     "depth": next(r for r in rows if r["format"] == int(gl.GL_DEPTH_COMPONENT24))["renderbuffer_id"]}
    if len(log) != 4:
        raise RenderPolicyError(f"unexpected allocation count {len(log)}")
    ids = set(out["redirected_multisample"].values()) | set(out["ordinary_resolve"].values())
    if len(ids) != 4:
        raise RenderPolicyError("draw and read attachments must be four distinct renderbuffers")
    return out


def verify_framebuffers(offscreen: Any, gl: Any, *, dimensions: tuple[int, int], allocations: Mapping) -> dict:
    """Query actual GL state for both FBOs; restore bindings and release the context."""
    width, height = check_dimensions(*dimensions)
    platform = getattr(offscreen, "_platform", None)
    renderer = getattr(offscreen, "_renderer", None)
    if platform is None or renderer is None:
        raise RenderPolicyError("offscreen renderer does not expose pinned framebuffer state")
    if tuple(getattr(renderer, "_main_fb_dims", (None, None))) != (width, height):
        raise RenderPolicyError("framebuffer dimensions differ from the requested viewport")
    fbos = {"draw": _gl_int(getattr(renderer, "_main_fb_ms", None)), "read": _gl_int(getattr(renderer, "_main_fb", None))}
    if min(fbos.values()) <= 0 or fbos["draw"] == fbos["read"]:
        raise RenderPolicyError("framebuffer identities are invalid")
    expected = {"draw": allocations["redirected_multisample"], "read": allocations["ordinary_resolve"]}
    saved = {}
    proof: dict = {"fbos": fbos, "attachments": {}, "renderbuffers": {}}
    primary = None
    try:
        platform.make_current()
        saved["draw"] = _gl_int(gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING))
        saved["read"] = _gl_int(gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING))
        saved["rb"] = _gl_int(gl.glGetIntegerv(gl.GL_RENDERBUFFER_BINDING))
        viewport = _gl_ints(gl.glGetIntegerv(gl.GL_VIEWPORT))
        if viewport != [0, 0, width, height]:
            raise RenderPolicyError(f"viewport {viewport} differs from {[0, 0, width, height]}")
        proof["viewport"] = viewport
        for name, fbo in fbos.items():
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo)
            status = _gl_int(gl.glCheckFramebufferStatus(gl.GL_DRAW_FRAMEBUFFER))
            if status != int(gl.GL_FRAMEBUFFER_COMPLETE):
                raise RenderPolicyError(f"{name} framebuffer is incomplete ({status})")
            samples = _gl_int(gl.glGetIntegerv(gl.GL_SAMPLES))
            if samples != 0:
                raise RenderPolicyError(f"{name} framebuffer reports GL_SAMPLES={samples}")
            attached = {}
            for slot, attachment in (("color", gl.GL_COLOR_ATTACHMENT0), ("depth", gl.GL_DEPTH_ATTACHMENT)):
                kind = _gl_int(gl.glGetFramebufferAttachmentParameteriv(
                    gl.GL_DRAW_FRAMEBUFFER, attachment, gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE))
                if kind != int(gl.GL_RENDERBUFFER):
                    # Texture and renderbuffer names share numeric IDs; never compare names across namespaces.
                    raise RenderPolicyError(f"{name} {slot} attachment is not a renderbuffer (type {kind})")
                attached[slot] = _gl_int(gl.glGetFramebufferAttachmentParameteriv(
                    gl.GL_DRAW_FRAMEBUFFER, attachment, gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME))
            if attached != dict(expected[name]):
                raise RenderPolicyError(f"{name} framebuffer attachments {attached} differ from allocations "
                                        f"{dict(expected[name])}")
            proof["attachments"][name] = {"status": status, "gl_samples": samples, "object_type": "GL_RENDERBUFFER",
                                          **attached}
            for slot, rb in attached.items():
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rb)
                row = {key: _gl_int(gl.glGetRenderbufferParameteriv(gl.GL_RENDERBUFFER, pname)) for key, pname in (
                    ("samples", gl.GL_RENDERBUFFER_SAMPLES), ("width", gl.GL_RENDERBUFFER_WIDTH),
                    ("height", gl.GL_RENDERBUFFER_HEIGHT), ("internal_format", gl.GL_RENDERBUFFER_INTERNAL_FORMAT))}
                allowed = ({int(gl.GL_RGBA), int(gl.GL_RGBA8)} if slot == "color"
                           else {int(gl.GL_DEPTH_COMPONENT24)})
                if row["samples"] != 0 or (row["width"], row["height"]) != (width, height) \
                        or row["internal_format"] not in allowed:
                    raise RenderPolicyError(f"{name} {slot} renderbuffer {rb} storage differs: {row}")
                proof["renderbuffers"][f"{name}_{slot}"] = {"id": rb, **row}
    except BaseException as exc:
        primary = exc
    finally:
        cleanup = []
        for key, target in (("draw", "GL_DRAW_FRAMEBUFFER"), ("read", "GL_READ_FRAMEBUFFER")):
            if key in saved:
                try:
                    gl.glBindFramebuffer(getattr(gl, target), saved[key])
                except BaseException as exc:  # noqa: BLE001
                    cleanup.append(exc)
        if "rb" in saved:
            try:
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, saved["rb"])
            except BaseException as exc:  # noqa: BLE001
                cleanup.append(exc)
        try:
            platform.make_uncurrent()
        except BaseException as exc:  # noqa: BLE001
            cleanup.append(exc)
    if primary is not None:
        raise primary
    if cleanup:
        raise RenderPolicyError(f"GL binding/context restoration failed: {cleanup[0]}") from cleanup[0]
    proof["bindings_restored"] = True
    return proof


# --------------------------------------------------------------------------
# adapter

class WorkbenchZeroStorageRenderer:
    """Wraps the unchanged legacy TexturedRenderer with the zero-storage policy."""

    def __init__(self, path: str, obj_id: int, *, inner: Any = None, gl: Any = None, stability: Any = None,
                 require_gpu_stage: bool = True, require_pins: bool = True):
        self.identity = factory_identity(require_pins=require_pins)
        if inner is None:
            from . import quality_gotrack as qg
            inner = qg.TexturedRenderer(path, obj_id, unlit=True, disable_multisampling=True,
                                        coordinate_mode="integer_centers_v1", render_policy="legacy_v1")
        if (getattr(inner, "render_policy", None) != "legacy_v1"
                or getattr(inner, "coordinate_mode", None) != "integer_centers_v1"
                or getattr(inner, "unlit", None) is not True or getattr(inner, "disable_multisampling", None) is not True):
            raise RenderPolicyError("inner renderer must be legacy_v1, integer_centers_v1, unlit, no multisampling")
        if gl is None:
            from OpenGL import GL as gl  # noqa: N811
        if stability is None:
            from . import quality_render_stability as stability
        self._inner, self._gl, self._stability = inner, gl, stability
        self._require_gpu_stage = require_gpu_stage
        self._poisoned = False
        self._verified: dict | None = None
        self.metadata: dict | None = None
        self.workbench_render_policy = POLICY

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def render_object_model(self, obj_id, camera_model_c2w, render_types=None, return_tensors=False,
                            background=None, **kwargs):
        if self._poisoned:
            raise RenderPolicyError("renderer was poisoned by an earlier failure; create a new one")
        size = check_dimensions(camera_model_c2w.width, camera_model_c2w.height)
        if self._require_gpu_stage:
            from . import workbench_telemetry as wt
            if not wt.gpu_stage_active():
                raise RenderPolicyError("zero-storage rendering requires the repository GPU stage")
        renderer_module = getattr(self._inner.pyrender, "renderer", None)
        if renderer_module is None:
            raise RenderPolicyError("pyrender renderer module is unavailable")
        with self._stability._render_lock:
            log: list = []
            try:
                with zero_storage_scope(renderer_module, self._gl, dimensions=size, log=log):
                    result = self._inner.render_object_model(obj_id, camera_model_c2w, render_types=render_types,
                                                             return_tensors=return_tensors, background=background,
                                                             **kwargs)
                offscreen = self._inner.offscreen
                internal = getattr(offscreen, "_renderer", None)
                fresh = (self._verified is None or self._verified["offscreen"] is not offscreen
                         or self._verified["internal"] is not internal or self._verified["size"] != size)
                if log:
                    if not fresh:
                        raise RenderPolicyError("unknown reallocation of a verified framebuffer")
                    allocations = validate_allocations(log, self._gl, size)
                else:
                    if fresh:
                        raise RenderPolicyError("fresh framebuffer has no logged zero-storage allocation")
                    allocations = self._verified["allocations"]
                proof = verify_framebuffers(offscreen, self._gl, dimensions=size, allocations=allocations)
                if not fresh and proof["fbos"] != self._verified["fbos"]:
                    raise RenderPolicyError("framebuffer identities changed on reuse")
                self._verified = {"offscreen": offscreen, "internal": internal, "size": size,
                                  "allocations": allocations, "fbos": proof["fbos"]}
                self.metadata = {"policy": POLICY, "dimensions": list(size), "fresh_allocation": bool(log),
                                 "allocation_log": [dict(r) for r in log], "allocations": allocations,
                                 "framebuffer_proof": proof, "factory": self.identity}
                return result
            except BaseException as exc:
                self._poisoned = True
                self._verified = None
                self.metadata = None
                try:
                    self._inner.close()
                except BaseException as close_exc:  # noqa: BLE001
                    raise RenderPolicyError(f"zero-storage render failed ({exc}); cleanup also failed "
                                            f"({close_exc})") from exc
                if isinstance(exc, RenderPolicyError):
                    raise
                raise RenderPolicyError(f"zero-storage render failed: {type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        self._verified = None
        self._inner.close()
