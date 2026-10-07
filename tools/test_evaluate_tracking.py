"""Synthetic scorer unit tests, never device-validation evidence."""
import unittest
from evaluate_tracking import evaluate


def row(frame, error=0, valid=1, point='a'):
    return dict(frame=frame, point=point, valid=valid, expected_x=10,
                expected_y=20, observed_x=10+error, observed_y=20)


class ScorerTests(unittest.TestCase):
    def test_known_errors(self):
        report = evaluate([row(i, i) for i in range(5)])
        self.assertEqual(report['medianErrorPixels'], 2)
        self.assertAlmostEqual(report['p95ErrorPixels'], 3.8)
        self.assertTrue(report['attachmentGatePassed'])

    def test_loss_counts_by_frame_not_point_count(self):
        report = evaluate([row(0), row(0, point='b'), row(1, valid=0)])
        self.assertEqual(report['trackingAvailability'], .5)
        self.assertFalse(report['attachmentGatePassed'])

    def test_all_lost(self):
        report = evaluate([row(0, valid=0)])
        self.assertIsNone(report['medianErrorPixels'])
        self.assertFalse(report['attachmentGatePassed'])

    def test_exact_threshold_fails(self):
        self.assertFalse(evaluate([row(0, 5)])['attachmentGatePassed'])

    def test_duplicate_or_inconsistent_frame_rejected(self):
        with self.assertRaises(ValueError): evaluate([row(0), row(0)])
        with self.assertRaises(ValueError): evaluate([row(0), row(0, valid=0, point='b')])

    def test_malformed_data_rejected(self):
        for rows in ([], [row(0, float('nan'))], [row(0, valid=2)]):
            with self.assertRaises(ValueError): evaluate(rows)


if __name__ == '__main__':
    unittest.main()
