"""Adversarial checks for the new renderer's proof boundary; no GL/GPU needed."""
from types import SimpleNamespace
import unittest
from bench import workbench_render as wr
from bench import workbench_neural as wn


class FakeGL:
    GL_DRAW_FRAMEBUFFER = 1
    GL_READ_FRAMEBUFFER = 2
    GL_DRAW_FRAMEBUFFER_BINDING = 3
    GL_READ_FRAMEBUFFER_BINDING = 4
    GL_RENDERBUFFER = 5
    GL_RENDERBUFFER_BINDING = 6
    GL_VIEWPORT = 7
    GL_FRAMEBUFFER_COMPLETE = 8
    GL_SAMPLES = 9
    GL_COLOR_ATTACHMENT0 = 10
    GL_DEPTH_ATTACHMENT = 11
    GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE = 12
    GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME = 13
    GL_RENDERBUFFER_SAMPLES = 14
    GL_RENDERBUFFER_WIDTH = 15
    GL_RENDERBUFFER_HEIGHT = 16
    GL_RENDERBUFFER_INTERNAL_FORMAT = 17
    GL_RGBA = 18
    GL_RGBA8 = 19
    GL_DEPTH_COMPONENT24 = 20
    GL_TEXTURE = 21

    def __init__(self, texture_attack=False):
        self.draw, self.read, self.rb = 100, 200, 300
        self.texture_attack = texture_attack
        self.storage_queries = 0

    def glGetIntegerv(self, key):
        return {3: self.draw, 4: self.read, 6: self.rb, 7: [0, 0, 280, 280], 9: 0}[key]

    def glBindFramebuffer(self, target, value):
        if target == self.GL_DRAW_FRAMEBUFFER:
            self.draw = value
        else:
            self.read = value

    def glBindRenderbuffer(self, target, value):
        self.rb = value

    def glCheckFramebufferStatus(self, target):
        return self.GL_FRAMEBUFFER_COMPLETE

    def glGetFramebufferAttachmentParameteriv(self, target, slot, query):
        if query == self.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
            return self.GL_TEXTURE if self.texture_attack else self.GL_RENDERBUFFER
        return {1: {10: 11, 11: 12}, 2: {10: 21, 11: 22}}[self.draw][slot]

    def glGetRenderbufferParameteriv(self, target, query):
        self.storage_queries += 1
        return {14: 0, 15: 280, 16: 280, 17: self.GL_RGBA8 if self.rb in (11, 21) else self.GL_DEPTH_COMPONENT24}[query]


class RenderProofTests(unittest.TestCase):
    def verify(self, gl):
        self.context = []
        platform = SimpleNamespace(make_current=lambda: self.context.append("current"),
                                   make_uncurrent=lambda: self.context.append("released"))
        offscreen = SimpleNamespace(_platform=platform,
            _renderer=SimpleNamespace(_main_fb_ms=1, _main_fb=2, _main_fb_dims=(280, 280)))
        return wr.verify_framebuffers(offscreen, gl, dimensions=(280, 280), allocations={
            "redirected_multisample": {"color": 11, "depth": 12},
            "ordinary_resolve": {"color": 21, "depth": 22}})

    def test_success_restores_bindings_and_context(self):
        gl = FakeGL()
        proof = self.verify(gl)
        self.assertTrue(proof["bindings_restored"])
        self.assertEqual((gl.draw, gl.read, gl.rb), (100, 200, 300))
        self.assertEqual(self.context, ["current", "released"])

    def test_same_id_texture_is_rejected_before_old_renderbuffer_is_queried(self):
        gl = FakeGL(texture_attack=True)
        with self.assertRaises(wr.RenderPolicyError):
            self.verify(gl)
        self.assertEqual(gl.storage_queries, 0)
        self.assertEqual((gl.draw, gl.read, gl.rb), (100, 200, 300))
        self.assertEqual(self.context[-1], "released")

    def test_native_portrait_dimensions_are_supported_with_a_finite_bound(self):
        self.assertEqual(wr.check_dimensions(1024, 1280), (1024, 1280))
        for dims in ((1281, 1024), (1024, 1281), (True, 280), (280, 0)):
            with self.subTest(dims=dims), self.assertRaises(wr.RenderPolicyError):
                wr.check_dimensions(*dims)

    def test_invalid_depth_metrics_cannot_pass_gate(self):
        for error in (-1., float("nan"), float("inf"), True, 1.6e-4):
            with self.subTest(error=error):
                self.assertFalse(wn.depth_row_passes({"depth_max_relative_error": error, "coverage_iou": 1.}))
        self.assertTrue(wn.depth_row_passes({"depth_max_relative_error": 8e-6, "coverage_iou": 1.}))


if __name__ == "__main__":
    unittest.main()
