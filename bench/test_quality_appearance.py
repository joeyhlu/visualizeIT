import unittest
import numpy as np
from types import SimpleNamespace
from .quality_appearance import AppearanceSettings, AppearanceCheckedBackend, appearance_metrics, appearance_decision


class AppearanceTests(unittest.TestCase):
    def test_validation_cannot_reuse_a_previous_image(self):
        backend = SimpleNamespace(refiner=SimpleNamespace(renderer=object()),
            validate_motion=lambda candidate, frame_id: (True, None, {}))
        checker = AppearanceCheckedBackend(backend, object())
        with self.assertRaisesRegex(ValueError, 'Current appearance frame required'):
            checker.validate_motion(None, 2)

    def test_appearance_cannot_mutate_the_tracking_render_context(self):
        renderer = object(); backend = SimpleNamespace(refiner=SimpleNamespace(renderer=renderer))
        with self.assertRaisesRegex(ValueError, 'separate renderer'):
            AppearanceCheckedBackend(backend, renderer)

    def test_brightness_change_preserves_texture_evidence(self):
        rng = np.random.default_rng(8)
        rgb = rng.integers(20, 190, (80, 80, 3), dtype=np.uint8)
        color = rgb.astype(np.float32)/255*.6+.1
        stats = appearance_metrics(rgb, color, np.ones((80, 80), bool))
        self.assertGreater(stats['grayscale_correlation'], .99)
        self.assertTrue(appearance_decision(stats, AppearanceSettings())[0])

    def test_bad_color_and_gradient_must_both_conflict(self):
        settings = AppearanceSettings()
        stats = dict(state='measured', pixels=1000, grayscale_correlation=.15, gradient_magnitude_correlation=.19)
        self.assertEqual(appearance_decision(stats, settings)[1], 'contradicts_rendered_texture')
        stats['grayscale_correlation'] = .7
        self.assertTrue(appearance_decision(stats, settings)[0])

    def test_hand_background_and_flat_texture_cannot_supply_evidence(self):
        rgb = np.full((40, 40, 3), 120, np.uint8)
        stats = appearance_metrics(rgb, rgb.astype(np.float32)/255, np.ones((40, 40), bool))
        self.assertIsNone(stats['grayscale_correlation'])
        self.assertTrue(appearance_decision(stats, AppearanceSettings())[0])
        self.assertEqual(appearance_metrics(rgb, rgb.astype(np.float32)/255, np.zeros((40, 40), bool))['state'], 'insufficient_overlap')


if __name__ == '__main__': unittest.main()
