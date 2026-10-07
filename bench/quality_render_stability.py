"""Scoped renderer experiment; never modify installed pyrender source."""
from contextlib import contextmanager
import operator
from threading import RLock

_render_lock = RLock()


@contextmanager
def without_multisampling(module):
    # The benchmark renders synchronously on one thread. Lock the experimental
    # hook and restore it even when GL raises; other capabilities are unchanged.
    with _render_lock:
        original = module.glEnable
        def enable(capability):
            if capability == module.GL_MULTISAMPLE: module.glDisable(capability)
            else: original(capability)
        module.glEnable = enable
        try: yield
        finally: module.glEnable = original


def _capture_dimensions(value):
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError('Capture render dimensions must be a width/height pair') from exc
    if len(pair) != 2 or any(type(item) is not int or item <= 0 or item > 1120 for item in pair):
        raise ValueError('Capture render dimensions must be positive integer values no larger than 1120')
    return pair


def _gl_int(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple, bool)):
        raise ValueError('OpenGL integer query did not return one scalar')
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('OpenGL integer query is not representable') from exc


@contextmanager
def capture_storage_scope(renderer_module, gl, *, dimensions, allocation_calls):
    """Route the pinned 4x color/depth allocations through ordinary storage."""

    width, height = _capture_dimensions(dimensions)
    if not isinstance(allocation_calls, list):
        raise TypeError('Capture allocation log must be a list')
    _render_lock.acquire()
    patched = False
    original_multisample = None
    try:
        original_multisample = getattr(renderer_module, 'glRenderbufferStorageMultisample')
        ordinary_storage = getattr(renderer_module, 'glRenderbufferStorage')
        if not callable(original_multisample) or not callable(ordinary_storage):
            raise ValueError('Pinned renderbuffer storage functions are unavailable')

        def intercept(target, samples, internalformat, actual_width, actual_height):
            row = {
                'target_name': None,
                'samples': None,
                'format_name': None,
                'width': None,
                'height': None,
                'renderbuffer_id': None,
                'ordinary_storage_delegated': False,
                'success': False,
            }
            allocation_calls.append(row)
            try:
                target_value = _gl_int(target)
                sample_value = _gl_int(samples)
                format_value = _gl_int(internalformat)
                width_value = _gl_int(actual_width)
                height_value = _gl_int(actual_height)
                row.update({
                    'target_name': 'GL_RENDERBUFFER' if target_value == int(gl.GL_RENDERBUFFER) else str(target_value),
                    'samples': sample_value,
                    'format_name': (
                        'GL_RGBA' if format_value == int(gl.GL_RGBA) else
                        'GL_DEPTH_COMPONENT24' if format_value == int(gl.GL_DEPTH_COMPONENT24) else
                        str(format_value)
                    ),
                    'width': width_value,
                    'height': height_value,
                })
                if target_value != int(gl.GL_RENDERBUFFER):
                    raise ValueError('Capture allocation used an unexpected target')
                if sample_value != 4:
                    raise ValueError('Capture allocation used an unexpected sample count')
                if format_value not in (int(gl.GL_RGBA), int(gl.GL_DEPTH_COMPONENT24)):
                    raise ValueError('Capture allocation used an unexpected attachment format')
                if (width_value, height_value) != (width, height):
                    raise ValueError('Capture allocation dimensions differ from the requested viewport')
                renderbuffer_id = _gl_int(gl.glGetIntegerv(gl.GL_RENDERBUFFER_BINDING))
                row['renderbuffer_id'] = renderbuffer_id
                if renderbuffer_id <= 0:
                    raise ValueError('Capture allocation has no bound renderbuffer identity')
                ordinary_storage(target, internalformat, actual_width, actual_height)
                row['ordinary_storage_delegated'] = True
                row['success'] = True
            except BaseException as exc:
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
                raise

        renderer_module.glRenderbufferStorageMultisample = intercept
        patched = True
        yield
    finally:
        try:
            if patched:
                renderer_module.glRenderbufferStorageMultisample = original_multisample
        finally:
            _render_lock.release()


def validate_capture_allocation_pairs(allocation_calls, *, dimensions):
    width, height = _capture_dimensions(dimensions)
    if not isinstance(allocation_calls, list) or len(allocation_calls) != 2:
        raise ValueError('Capture allocation must contain exactly one color/depth pair')
    by_format = {row.get('format_name'): row for row in allocation_calls if isinstance(row, dict)}
    if set(by_format) != {'GL_RGBA', 'GL_DEPTH_COMPONENT24'}:
        raise ValueError('Capture allocation must contain one color and one depth renderbuffer')
    ids = []
    for row in allocation_calls:
        if not isinstance(row, dict):
            raise ValueError('Capture allocation record is malformed')
        if (row.get('target_name') != 'GL_RENDERBUFFER' or
                type(row.get('samples')) is not int or row.get('samples') != 4 or
                type(row.get('width')) is not int or row.get('width') != width or
                type(row.get('height')) is not int or row.get('height') != height or
                row.get('ordinary_storage_delegated') is not True or row.get('success') is not True):
            raise ValueError('Capture attachment pair does not match the exact authorized allocation')
        renderbuffer_id = row.get('renderbuffer_id')
        if type(renderbuffer_id) is not int or renderbuffer_id <= 0:
            raise ValueError('Capture allocation has an invalid renderbuffer identity')
        ids.append(renderbuffer_id)
    if ids[0] == ids[1]:
        raise ValueError('Capture color and depth renderbuffers must have distinct identities')
    return {
        'passed': True,
        'dimensions': [width, height],
        'color': dict(by_format['GL_RGBA']),
        'depth': dict(by_format['GL_DEPTH_COMPONENT24']),
    }


def verify_capture_framebuffer(offscreen, gl, *, expected_dimensions):
    """Verify the actual draw FBO and restore both framebuffer bindings/context."""

    expected = _capture_dimensions(expected_dimensions)
    platform = getattr(offscreen, '_platform', None)
    renderer = getattr(offscreen, '_renderer', None)
    if platform is None or renderer is None:
        raise ValueError('Capture OffscreenRenderer did not expose the pinned framebuffer state')
    if not callable(getattr(platform, 'supports_framebuffers', None)) or not platform.supports_framebuffers():
        raise ValueError('Capture platform does not support framebuffers')
    draw_fbo = getattr(renderer, '_main_fb_ms', None)
    read_fbo = getattr(renderer, '_main_fb', None)
    actual = _capture_dimensions(getattr(renderer, '_main_fb_dims', None))
    try:
        draw_id = _gl_int(draw_fbo)
        read_id = _gl_int(read_fbo)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Capture framebuffer IDs are malformed') from exc
    if draw_id <= 0 or read_id <= 0:
        raise ValueError('Capture framebuffer IDs must be positive')
    if actual != expected:
        raise ValueError(f'Capture framebuffer dimensions {actual} do not match requested {expected}')

    saved_draw = None
    saved_read = None
    draw_known = False
    read_known = False
    primary_error = None
    cleanup_errors = []
    result = None
    try:
        try:
            platform.make_current()
            saved_draw = _gl_int(gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING))
            draw_known = True
            saved_read = _gl_int(gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING))
            read_known = True
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_id)
            status = _gl_int(gl.glCheckFramebufferStatus(gl.GL_DRAW_FRAMEBUFFER))
            complete = status == int(gl.GL_FRAMEBUFFER_COMPLETE)
            samples = _gl_int(gl.glGetIntegerv(gl.GL_SAMPLES))
            sample_buffers = _gl_int(gl.glGetIntegerv(gl.GL_SAMPLE_BUFFERS))
            result = {
                'framebuffer_complete': complete,
                'framebuffer_status': status,
                'gl_samples': samples,
                'gl_sample_buffers': sample_buffers,
                'framebuffer_fields': {
                    'multisample_draw_fbo': draw_id,
                    'single_sample_read_fbo': read_id,
                    'multisample_dimensions': [actual[0], actual[1]],
                },
            }
        except BaseException as exc:
            primary_error = exc
    finally:
        if draw_known:
            try:
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, saved_draw)
            except BaseException as exc:
                cleanup_errors.append(('restore_draw_binding', exc))
        if read_known:
            try:
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, saved_read)
            except BaseException as exc:
                cleanup_errors.append(('restore_read_binding', exc))
        try:
            platform.make_uncurrent()
        except BaseException as exc:
            cleanup_errors.append(('release_current_context', exc))
    if cleanup_errors:
        details = '; '.join(f'{name}: {type(exc).__name__}: {exc}' for name, exc in cleanup_errors)
        if primary_error is not None:
            raise RuntimeError(f'Capture framebuffer query failed ({primary_error}); cleanup also failed ({details})') from primary_error
        raise RuntimeError(f'Capture framebuffer cleanup failed: {details}') from cleanup_errors[0][1]
    if primary_error is not None:
        raise primary_error
    if not (draw_known and read_known):
        raise ValueError('Capture framebuffer query did not observe both bindings')
    if result is None:
        raise ValueError('Capture framebuffer query produced no result')
    result.update({
        'framebuffer_bindings_restored': True,
        'current_context_released': True,
        'dimension_match': True,
    })
    if not result['framebuffer_complete']:
        raise ValueError('Capture draw framebuffer is incomplete')
    if result['gl_samples'] != 0 or result['gl_sample_buffers'] != 0:
        raise ValueError('Capture draw framebuffer is not zero-sample')
    return result
