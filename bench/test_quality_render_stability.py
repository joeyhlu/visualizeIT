import unittest
from types import SimpleNamespace
from .quality_render_stability import without_multisampling


class RenderStabilityTests(unittest.TestCase):
    def test_hook_preserves_other_gl_state_and_restores_after_failure(self):
        calls=[]; original=lambda flag:calls.append(('enable', flag))
        module=SimpleNamespace(GL_MULTISAMPLE=8, glEnable=original, glDisable=lambda flag:calls.append(('disable',flag)))
        with self.assertRaisesRegex(RuntimeError,'render failed'):
            with without_multisampling(module):
                module.glEnable(8); module.glEnable(9)
                raise RuntimeError('render failed')
        self.assertEqual(calls,[('disable',8),('enable',9)])
        self.assertIs(module.glEnable,original)


if __name__ == '__main__': unittest.main()
