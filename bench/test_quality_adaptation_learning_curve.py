"""Manufactured scalar diagnostics only; no optimization or model imports."""
import copy
import unittest

from bench import quality_adaptation_learning_curve as curve


def history():
    return [{"epoch": 0, "steps": 16, "train_loss": .5, "validation": {
        "adapted": {"object_macro_epe_px": 3.}, "frozen": {"object_macro_epe_px": 4.},
        "availability": {"available": 2, "total": 3}}}]


def metrics():
    base = {"steps": 16, "metric_definition": "mean_endpoint_error_v1", "units": "crop_pixels",
            "aggregation": "object_macro", "coverage_policy": "all_requested_no_missing",
            "dataset_manifest_sha256": "a" * 64, "available": 2, "total": 2}
    return [{**base, "split": "train", "value": 1.},
            {**base, "split": "validation", "value": 3., "available": 3, "total": 3}]


def frame(frame_id, object_id, epe, visible, known, tp, fp, fn):
    return {"id": frame_id, "object_id": object_id, "epe_px": epe,
            "visible_pixels": visible,
            "visibility": {"threshold": .3, "visible_pixels": visible,
                           "known_pixels": known, "true_positive": tp,
                           "false_positive": fp, "false_negative": fn}}


def paired_history(*, reverse=False):
    frozen = [
        frame("a-low", "objA", 5., 1, 20, 0, 1, 1),
        frame("a-mid", "objA", 3., 2, 10, 2, 1, 0),
        frame("b-high", "objB", 10., 9, 10, 9, 0, 0),
        frame("b-full", "objB", 8., 10, 10, 9, 0, 1),
    ]
    adapted = [
        frame("a-low", "objA", 4., 1, 20, 0, 2, 1),
        frame("a-mid", "objA", 3., 2, 10, 1, 2, 1),
        frame("b-high", "objB", 12., 9, 10, 8, 1, 1),
        frame("b-full", "objB", 6., 10, 10, 10, 0, 0),
    ]
    if reverse:
        frozen.reverse()
        adapted.reverse()
    def row(epoch, adapted_epe, frozen_epe, adapted_frames, frozen_frames):
        return {"epoch": epoch, "steps": (epoch + 1) * 10, "train_loss": 1.,
                "validation": {"adapted": {"object_macro_epe_px": adapted_epe},
                               "frozen": {"object_macro_epe_px": frozen_epe},
                               "availability": {"available": 4, "total": 4},
                               "frames": copy.deepcopy(adapted_frames),
                               "frozen_frames": copy.deepcopy(frozen_frames)}}
    older_adapted = [{**copy.deepcopy(item), "epe_px": item["epe_px"] + 1.}
                     for item in adapted]
    older_frozen = [{**copy.deepcopy(item), "epe_px": item["epe_px"] + 1.}
                    for item in frozen]
    return [row(0, 7.25, 7.5, older_adapted, older_frozen),
            row(1, 6.25, 6.5, adapted, frozen)]


class LearningCurveTests(unittest.TestCase):
    def test_composite_loss_and_epe_never_produce_gap(self):
        report = curve.summarize(history(), {"status": "incomplete", "steps": 16})
        self.assertEqual(report["matched_metric_gaps"], [])
        self.assertEqual(report["diagnostic_evidence"], "insufficient_diagnostic_evidence")
        self.assertIsNone(report["fit_diagnosis"])
        self.assertEqual(report["epochs"][0]["validation_availability"]["failed"], 1)
        self.assertEqual(report["epochs"][0]["validation_frozen_object_macro_epe_px"], 4.)

    def test_matched_units_can_have_different_disjoint_denominators(self):
        result = curve.summarize(history(), {"steps": 16}, matched_metrics=metrics())
        gap = result["matched_metric_gaps"][0]
        self.assertEqual(gap["validation_minus_train"], 2.)
        self.assertEqual(gap["train_available"], 2)
        self.assertEqual(gap["validation_available"], 3)
        self.assertIsNone(result["fit_diagnosis"])

    def test_mismatched_metrics_partial_coverage_and_nonfinite_rejected(self):
        for key in ("metric_definition", "units", "aggregation", "coverage_policy", "dataset_manifest_sha256"):
            with self.subTest(key=key):
                rows = metrics()
                rows[1][key] = "b" * 64 if key == "dataset_manifest_sha256" else "different"
                with self.assertRaisesRegex(ValueError, "incomparable"):
                    curve.compare_matched_metrics(rows)
        for value in (float("nan"), float("inf"), -1, True):
            rows = metrics()
            rows[1]["value"] = value
            with self.assertRaises(ValueError):
                curve.compare_matched_metrics(rows)
        rows = metrics()
        rows[1]["available"] = 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            curve.compare_matched_metrics(rows)

    def test_bad_history_status_and_missing_matched_side_rejected(self):
        rows = history()
        rows[0]["validation"]["availability"]["available"] = 4
        with self.assertRaises(ValueError):
            curve.summarize(rows, {"steps": 16})
        with self.assertRaises(ValueError):
            curve.summarize(history(), {"steps": 1})
        with self.assertRaises(ValueError):
            curve.compare_matched_metrics(metrics()[:1])
        self.assertEqual(curve.summarize([], {"steps": 0})["completed_epochs"], 0)

    def test_paired_frame_diagnostics_are_hand_computed_and_order_independent(self):
        left = curve.summarize(paired_history(), {"steps": 20})["paired_frame_diagnostics"]
        right = curve.summarize(paired_history(reverse=True), {"steps": 20})["paired_frame_diagnostics"]
        self.assertEqual(left, right)
        self.assertEqual(left["status"], "available")
        self.assertEqual(left["selected_epoch"], 1)
        self.assertEqual(left["pairing"], {"by": "id", "matched_frames": 4, "independent_objects": 2})
        self.assertEqual(left["all_frame_epe_px"], {
            "baseline_frozen_mean": 6.5, "adapted_mean": 6.25,
            "adapted_minus_baseline_mean": -.25,
            "improved": 2, "tied": 1, "regressed": 1})
        self.assertEqual(left["object_macro_paired_gains"]["mean_gain_px"], .25)
        self.assertEqual(left["object_macro_paired_gains"]["per_object"], [
            {"object_id": "objA", "frames": 2, "baseline_frozen_mean_epe_px": 4.,
             "adapted_mean_epe_px": 3.5, "gain_px": .5},
            {"object_id": "objB", "frames": 2, "baseline_frozen_mean_epe_px": 9.,
             "adapted_mean_epe_px": 9., "gain_px": 0.}])
        frozen = left["pooled_visibility"]["baseline_frozen"]
        self.assertEqual({key: frozen[key] for key in (
            "true_positive", "false_positive", "false_negative", "true_negative")},
            {"true_positive": 20, "false_positive": 2, "false_negative": 2, "true_negative": 26})
        self.assertAlmostEqual(frozen["precision"], 20 / 22)
        self.assertAlmostEqual(frozen["recall"], 20 / 22)
        adapted = left["pooled_visibility"]["adapted"]
        self.assertEqual({key: adapted[key] for key in (
            "true_positive", "false_positive", "false_negative", "true_negative")},
            {"true_positive": 19, "false_positive": 5, "false_negative": 3, "true_negative": 23})
        self.assertEqual({key: value["frames"] for key, value in
                          left["visible_fraction_bins"]["bins"].items()},
                         {"<0.1": 1, "0.1<=x<0.9": 1, ">=0.9": 2})
        self.assertEqual(left["per_frame"][0]["id"], "a-low")
        self.assertEqual(left["per_frame"][0]["visibility"]["baseline_frozen"]["true_negative"], 18)
        self.assertIn("no cutoff tuning", left["pooled_visibility"]["threshold_policy"])
        self.assertIn("one independent object is insufficient",
                      " ".join(left["limitations"]))

    def test_no_frame_evidence_keeps_scalar_history_api_and_reports_unavailable(self):
        report = curve.summarize(history(), {"status": "incomplete", "steps": 16})
        self.assertEqual(report["epochs"][0]["validation_adapted_object_macro_epe_px"], 3.)
        self.assertEqual(report["paired_frame_diagnostics"]["status"], "unavailable")

    def test_zero_denominator_visibility_rates_are_none(self):
        rows = paired_history()
        no_positive_predictions = frame("no-positive-predictions", "objZ", 1., 1, 10, 0, 0, 1)
        rows[1]["validation"]["frames"] = [no_positive_predictions]
        rows[1]["validation"]["frozen_frames"] = [no_positive_predictions.copy()]
        rows[1]["validation"]["availability"] = {"available": 1, "total": 4}
        rows[1]["validation"]["adapted"]["object_macro_epe_px"] = 1.
        rows[1]["validation"]["frozen"]["object_macro_epe_px"] = 1.
        diagnostics = curve.summarize(rows, {"steps": 20})["paired_frame_diagnostics"]
        for run in ("baseline_frozen", "adapted"):
            self.assertIsNone(diagnostics["pooled_visibility"][run]["precision"])
            self.assertEqual(diagnostics["pooled_visibility"][run]["recall"], 0.)
        self.assertEqual(diagnostics["visible_fraction_bins"]["zero_known_support_frames"], 0)

    def test_paired_frame_count_must_match_availability(self):
        rows = paired_history()
        rows[1]["validation"]["frames"].pop()
        rows[1]["validation"]["frozen_frames"].pop()
        with self.assertRaisesRegex(ValueError, "match validation availability.available"):
            curve.summarize(rows, {"steps": 20})

        rows = paired_history()
        rows[1]["validation"]["availability"] = {"available": 3, "total": 4}
        with self.assertRaisesRegex(ValueError, "match validation availability.available"):
            curve.summarize(rows, {"steps": 20})

    def test_frame_epe_rejects_zero_visible_or_zero_known_support(self):
        cases = (
            ("zero visible", frame("no-visible", "objZ", 0., 0, 10, 0, 0, 0),
             "require positive visible support"),
            ("all zero", frame("all-zero", "objZ", 0., 0, 0, 0, 0, 0),
             "known support must be positive"),
        )
        for label, bad_frame, message in cases:
            with self.subTest(label=label):
                rows = paired_history()
                rows[1]["validation"]["frames"] = [bad_frame]
                rows[1]["validation"]["frozen_frames"] = [bad_frame.copy()]
                rows[1]["validation"]["availability"] = {"available": 1, "total": 4}
                rows[1]["validation"]["adapted"]["object_macro_epe_px"] = 0.
                rows[1]["validation"]["frozen"]["object_macro_epe_px"] = 0.
                with self.assertRaisesRegex(ValueError, message):
                    curve.summarize(rows, {"steps": 20})

    def test_history_object_macro_must_match_every_frame_evidence_epoch(self):
        for index, side in ((1, "adapted"), (1, "frozen"), (0, "adapted")):
            with self.subTest(index=index, side=side):
                rows = paired_history()
                rows[index]["validation"][side]["object_macro_epe_px"] += .01
                message_side = "adapted" if side == "adapted" else "frozen"
                with self.assertRaisesRegex(ValueError, f"{message_side} history object-macro EPE"):
                    curve.summarize(rows, {"steps": 20})

    def test_unequal_object_frame_counts_use_object_macro_weighting(self):
        rows = paired_history()[1:]
        validation = rows[0]["validation"]
        validation["frames"].append(frame("b-extra", "objB", 21., 1, 10, 1, 0, 0))
        validation["frozen_frames"].append(frame("b-extra", "objB", 27., 1, 10, 1, 0, 0))
        validation["availability"] = {"available": 5, "total": 5}
        validation["adapted"]["object_macro_epe_px"] = 8.25
        validation["frozen"]["object_macro_epe_px"] = 9.5
        report = curve.summarize(rows, {"steps": 20})["paired_frame_diagnostics"]
        self.assertEqual(report["pairing"]["matched_frames"], 5)
        self.assertEqual(report["object_macro_paired_gains"]["mean_gain_px"], 1.25)
        self.assertEqual([item["frames"] for item in
                          report["object_macro_paired_gains"]["per_object"]], [2, 3])
        self.assertEqual(report["all_frame_epe_px"]["adapted_minus_baseline_mean"], -1.4)

    def test_paired_frame_evidence_rejects_malformed_inputs(self):
        mutations = (
            ("one list absent", lambda h: h[1]["validation"].pop("frozen_frames"), "both validation.frames"),
            ("duplicate id", lambda h: h[1]["validation"]["frames"].append(
                h[1]["validation"]["frames"][0].copy()), "duplicate adapted frame id"),
            ("missing counterpart", lambda h: h[1]["validation"]["frames"][0].update(
                id="missing-counterpart"), "missing counterparts"),
            ("object mismatch", lambda h: h[1]["validation"]["frames"][0].update(object_id="other"),
             "object mismatch"),
            ("visible/known support mismatch", lambda h: h[1]["validation"]["frames"][0]["visibility"].update(
                known_pixels=11), "visible/known support mismatch"),
            ("incoherent confusion counts", lambda h: h[1]["validation"]["frames"][0]["visibility"].update(
                true_positive=1), "do not sum to visible support"),
            ("too many false positives", lambda h: h[1]["validation"]["frames"][0]["visibility"].update(
                false_positive=20), "exceed known background support"),
            ("nonfinite EPE", lambda h: h[1]["validation"]["frames"][0].update(epe_px=float("inf")),
             "nonfinite diagnostic metric"),
            ("different threshold", lambda h: h[1]["validation"]["frames"][0]["visibility"].update(
                threshold=.5), "one shared fixed threshold"),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                rows = paired_history()
                mutate(rows)
                with self.assertRaisesRegex(ValueError, message):
                    curve.summarize(rows, {"steps": 20})


if __name__ == "__main__":
    unittest.main()
