"""Focused offline tests for the flow-terminal continuation boundary."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_flow_terminal as flow_terminal
from .quality_adaptation_train import TrainConfig, epoch_batches


class FlowTerminalContractTests(unittest.TestCase):
    def test_config_is_fixed_but_keeps_operator_caps(self):
        base = TrainConfig(device="cpu", max_steps=700, max_seconds=120,
                           max_total_seconds=600, max_checkpoint_bytes=64 * 1024**2,
                           max_output_bytes=256 * 1024**2)
        fixed = flow_terminal.continuation_config(base)
        self.assertEqual((fixed.seed, fixed.epochs, fixed.batch_size), (521, 3, 1))
        self.assertEqual((fixed.learning_rate, fixed.weight_decay), (1e-5, 1e-4))
        self.assertEqual((fixed.visibility_weight, fixed.gradient_clip), (0.0, 1.0))
        self.assertEqual((fixed.amp, fixed.checkpoint_every_steps), (False, 8))
        self.assertEqual(fixed.max_steps, base.max_steps)
        self.assertEqual(fixed.max_seconds, base.max_seconds)
        self.assertEqual(fixed.max_total_seconds, base.max_total_seconds)

    def test_confidence_parity_checks_native_float32_threshold_and_tolerance(self):
        baseline = np.full((flow_terminal.SIZE, flow_terminal.SIZE), 0.3001, dtype=np.float32)
        same = baseline.copy()
        self.assertTrue(flow_terminal.confidence_parity(same, baseline)["passed"])
        changed = baseline.copy()
        changed[0, 0] += np.float32(2e-6)
        result = flow_terminal.confidence_parity(changed, baseline)
        self.assertFalse(result["passed"])
        self.assertIn("confidence_difference_exceeds_tolerance", result["reasons"])
        threshold_crossing = baseline.copy()
        threshold_crossing[0, 0] = np.float32(0.2999)
        result = flow_terminal.confidence_parity(threshold_crossing, baseline)
        self.assertIn("native_threshold_decisions_changed", result["reasons"])
        exact_threshold = np.full_like(baseline, np.float32(0.3))
        next_above = exact_threshold.copy()
        next_above[0, 0] = np.nextafter(np.float32(0.3), np.float32(np.inf))
        result = flow_terminal.confidence_parity(next_above, exact_threshold)
        self.assertLess(result["max_abs_difference"], flow_terminal.CONFIDENCE_TOLERANCE)
        self.assertEqual(result["decision_changes"], 1)
        self.assertIn("native_threshold_decisions_changed", result["reasons"])
        self.assertFalse(flow_terminal.confidence_parity(baseline.astype(np.float64), baseline)["passed"])

    def test_resume_cursor_is_bound_to_deterministic_rows_and_step_count(self):
        cfg = flow_terminal.continuation_config(TrainConfig())
        rows = [{"id": f"row-{i}"} for i in range(4)]
        batches = epoch_batches(rows, 0, cfg)
        self.assertEqual(len(batches), 4)
        state = {"epoch": 1, "batch_cursor": 2, "steps": 6, "bad_epochs": 0}
        flow_terminal.validate_resume_cursor(state, rows, cfg)
        with self.assertRaisesRegex(ValueError, "steps do not match"):
            flow_terminal.validate_resume_cursor({**state, "steps": 5}, rows, cfg)
        with self.assertRaisesRegex(ValueError, "cursor exceeds"):
            flow_terminal.validate_resume_cursor({**state, "batch_cursor": 5, "steps": 9}, rows, cfg)

    def test_preflight_uses_train_only_report_and_does_not_inspect_heldout_arrays(self):
        class DatasetDouble:
            identity = "d" * 64
            root = Path("unused")
            document = {"provenance": {"label_origin": "independent_annotations"}}

            def arrays(self, row):
                raise AssertionError(f"unexpected payload read: {row}")

        dataset = DatasetDouble()
        report = {"dataset_sha256": dataset.identity, "splits": {}, "train_eda": {}}
        with patch.object(flow_terminal, "load_dataset", return_value=dataset) as load, \
             patch.object(flow_terminal, "validate_feature_receipt") as receipt, \
             patch.object(flow_terminal, "verify_sources", return_value={"source_lock_sha256": "s"}), \
             patch.object(flow_terminal, "data_report", return_value=report), \
             patch.object(flow_terminal, "runtime_versions", return_value={"torch": "mock"}):
            result = flow_terminal.preflight(Path("fixture-manifest.json"), TrainConfig())
        load.assert_called_once_with(Path("fixture-manifest.json"), inspect_arrays=False)
        receipt.assert_called_once_with(dataset)
        self.assertFalse(result["validation_arrays_opened"])
        self.assertEqual(result["test_payload_reads"], 0)
        self.assertEqual(result["data"], report)

    def test_cloud_guard_runs_before_dataset_or_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch.object(flow_terminal, "cloud_guard", side_effect=RuntimeError("cloud required")), \
                 patch.object(flow_terminal, "preflight") as preflight, \
                 patch.object(flow_terminal, "load_dataset") as load:
                with self.assertRaisesRegex(RuntimeError, "cloud required"):
                    flow_terminal.train(Path("unused.json"), output, TrainConfig(), cloud_execution=False)
            preflight.assert_not_called()
            load.assert_not_called()
            self.assertFalse(output.exists())

    def test_cuda_policy_hash_binds_reviewed_bootstrap_environment_and_numeric_settings(self):
        cfg = flow_terminal.continuation_config(TrainConfig(device="cuda"))
        policy = flow_terminal.policy_document(
            SimpleNamespace(identity="d" * 64), cfg, {"torch": "2.5.1+cu124"})
        expected = policy["runtime_policy"]
        self.assertEqual(expected["cuda_bootstrap_sha256"], flow_terminal.CUDA_BOOTSTRAP_SHA256)
        self.assertEqual(expected["required_environment"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(expected["required_environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertTrue(expected["deterministic_algorithms"])
        self.assertFalse(expected["matmul_allow_tf32"])
        self.assertFalse(expected["cudnn_allow_tf32"])
        self.assertEqual(expected["torch_num_threads"], 1)
        self.assertEqual(expected["torch_num_interop_threads"], 1)
        with patch.dict("os.environ", flow_terminal.CUDA_POLICY_ENV):
            flow_terminal.assert_runtime_environment(cfg)
        bad_env = dict(flow_terminal.CUDA_POLICY_ENV, CUBLAS_WORKSPACE_CONFIG=":16:8")
        with patch.dict("os.environ", bad_env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "pinned policy"):
                flow_terminal.assert_runtime_environment(cfg)

    def test_feature_union_is_opt_in_and_adds_recipe_and_module_policy_pins(self):
        cfg = flow_terminal.continuation_config(TrainConfig())
        recipe_sha = "a" * 64
        dataset = SimpleNamespace(
            identity="d" * 64,
            document={"provenance": {"feature_recipe_sha256": recipe_sha}},
        )
        ordinary = flow_terminal.policy_document(dataset, cfg, {"torch": "fixture"})
        union_policy = flow_terminal.policy_document(
            dataset, cfg, {"torch": "fixture"}, feature_union=True)
        self.assertNotIn("feature_union", ordinary)
        self.assertNotIn("feature_union", ordinary["source_files_sha256"])
        self.assertEqual(union_policy["feature_union"]["recipe_sha256"], recipe_sha)
        self.assertEqual(
            union_policy["feature_union"]["module_sha256"],
            flow_terminal.sha256(Path(flow_terminal.feature_union_module.__file__)),
        )
        self.assertEqual(union_policy["source_files_sha256"]["feature_union"],
                         union_policy["feature_union"]["module_sha256"])

    def test_resume_rejects_union_optin_or_recipe_module_identity_mismatch(self):
        expected = {"enabled": True, "recipe_sha256": "a" * 64,
                    "module_sha256": "b" * 64}
        flow_terminal.validate_feature_union_mode({"feature_union": expected}, expected)
        with self.assertRaisesRegex(ValueError, "feature-union"):
            flow_terminal.validate_feature_union_mode({"feature_union": expected}, None)
        with self.assertRaisesRegex(ValueError, "feature-union"):
            flow_terminal.validate_feature_union_mode({"feature_union": None}, expected)

    def test_export_policy_reconstructs_union_identity_from_persisted_checkpoint(self):
        cfg = flow_terminal.continuation_config(TrainConfig())
        versions = {"torch": "fixture"}
        identity = {"enabled": True, "recipe_sha256": "a" * 64,
                    "module_sha256": flow_terminal.sha256(
                        Path(flow_terminal.feature_union_module.__file__))}
        dataset = SimpleNamespace(identity="d" * 64,
                                  document={"provenance": {
                                      "feature_recipe_sha256": identity["recipe_sha256"]}})
        policy = flow_terminal.policy_document(dataset, cfg, versions, feature_union=True)
        state = {"dataset_sha256": dataset.identity, "feature_union": identity,
                 "config_sha256": flow_terminal.canonical_hash(flow_terminal.asdict(cfg)),
                 "policy_sha256": flow_terminal.canonical_hash(policy),
                 "source_lock_sha256": flow_terminal.sha256(flow_terminal.LOCK),
                 "source_files_sha256": policy["source_files_sha256"],
                 "versions": versions, "runtime_policy": {"device": "cpu"},
                 "optimizer": {}}
        with patch.object(flow_terminal, "check_resume"), \
             patch.object(flow_terminal, "validate_checkpoint_integrity"), \
             patch.object(flow_terminal, "validate_optimizer_state"):
            actual = flow_terminal._checkpoint_policy_check(
                state, cfg, versions, state["runtime_policy"])
        self.assertEqual(actual, state["policy_sha256"])
        changed = dict(state, feature_union={**identity, "recipe_sha256": "c" * 64})
        with patch.object(flow_terminal, "check_resume"), \
             patch.object(flow_terminal, "validate_checkpoint_integrity"), \
             patch.object(flow_terminal, "validate_optimizer_state"):
            with self.assertRaisesRegex(ValueError, "policy/runtime/source identity"):
                flow_terminal._checkpoint_policy_check(
                    changed, cfg, versions, state["runtime_policy"])

    def test_frozen_identity_allows_only_flow_rows_to_differ(self):
        weight = np.arange(3 * 256, dtype=np.float32).reshape(3, 256, 1, 1)
        bias = np.arange(3, dtype=np.float32)
        official = {"head.4.weight": weight.copy(), "head.4.bias": bias.copy(),
                    "head.0.weight": np.ones((1,), dtype=np.float32)}
        candidate = {key: value.copy() for key, value in official.items()}
        candidate["head.4.weight"][:2] += 7
        candidate["head.4.bias"][:2] += 7
        flow_terminal.validate_frozen_state(candidate, official)
        frozen = flow_terminal.frozen_state(official)
        self.assertEqual(frozen["head.4.weight"].shape[0], 1)
        confidence_changed = {key: value.copy() for key, value in candidate.items()}
        confidence_changed["head.4.weight"][2] += 1
        with self.assertRaisesRegex(ValueError, "confidence convolution weights"):
            flow_terminal.validate_frozen_state(confidence_changed, official)
        trunk_changed = {key: value.copy() for key, value in candidate.items()}
        trunk_changed["head.0.weight"][0] += 1
        with self.assertRaisesRegex(ValueError, "frozen official head tensor"):
            flow_terminal.validate_frozen_state(trunk_changed, official)

        signed_zero_base = {key: value.copy() for key, value in official.items()}
        signed_zero_candidate = {key: value.copy() for key, value in official.items()}
        signed_zero_base["head.4.weight"][2, 0, 0, 0] = np.float32(0.0)
        signed_zero_candidate["head.4.weight"][2, 0, 0, 0] = np.float32(-0.0)
        self.assertTrue(np.array_equal(signed_zero_base["head.4.weight"],
                                       signed_zero_candidate["head.4.weight"]))
        with self.assertRaisesRegex(ValueError, "confidence convolution weights"):
            flow_terminal.validate_frozen_state(signed_zero_candidate, signed_zero_base)

    def test_resume_rejects_adamw_hyperparameter_or_step_tampering(self):
        cfg = flow_terminal.continuation_config(TrainConfig())
        group = {
            "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay,
            "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False,
            "maximize": False, "foreach": None, "capturable": False,
            "differentiable": False, "fused": None, "params": [0, 1],
        }
        zero_step = {"state": {}, "param_groups": [group.copy()]}
        flow_terminal.validate_optimizer_state(zero_step, cfg, 0)
        for key, value in (("lr", cfg.learning_rate * 2),
                           ("weight_decay", cfg.weight_decay * 2)):
            tampered = {"state": {}, "param_groups": [dict(group, **{key: value})]}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fixed continuation"):
                flow_terminal.validate_optimizer_state(tampered, cfg, 0)
        stepped = {
            "state": {
                0: {"step": 3.0, "exp_avg": np.zeros(2), "exp_avg_sq": np.zeros(2)},
                1: {"step": 3.0, "exp_avg": np.zeros(1), "exp_avg_sq": np.zeros(1)},
            },
            "param_groups": [group.copy()],
        }
        flow_terminal.validate_optimizer_state(stepped, cfg, 3)
        stepped["state"][1]["step"] = 2.0
        with self.assertRaisesRegex(ValueError, "step differs"):
            flow_terminal.validate_optimizer_state(stepped, cfg, 3)

    def test_checkpoint_integrity_covers_optimizer_and_cursor(self):
        state = {"steps": 8, "batch_cursor": 8,
                 "optimizer": {"param_groups": [{"lr": 1e-5}]}}
        state["integrity_sha256"] = flow_terminal.checkpoint_integrity_digest(state)
        flow_terminal.validate_checkpoint_integrity(state)
        tampered = {**state, "optimizer": {"param_groups": [{"lr": 2e-5}]}}
        with self.assertRaisesRegex(ValueError, "integrity digest"):
            flow_terminal.validate_checkpoint_integrity(tampered)

    def test_incomplete_or_failed_parity_state_cannot_export(self):
        best = {"completed": False, "history": [], "final_evaluation": None}
        with self.assertRaisesRegex(ValueError, "successful final confidence parity"):
            flow_terminal.require_exportable_state({"completed": False}, best)
        completed_but_bad = {
            "completed": True,
            "history": [{"validation": {"adapted": {"object_macro_epe_px": 22.0}}}],
            "final_evaluation": {"passed": False},
        }
        with self.assertRaisesRegex(ValueError, "successful final confidence parity"):
            flow_terminal.require_exportable_state({"completed": True}, completed_but_bad)
        # EPE is reported for independent serving review; it is not a completion gate.
        accepted_experiment = {
            "completed": True,
            "history": [{"validation": {"adapted": {"object_macro_epe_px": 22.0}}}],
            "final_evaluation": {"passed": True},
        }
        flow_terminal.require_exportable_state({"completed": True}, accepted_experiment)


try:
    import torch as _torch
except ImportError:  # The offline runtime may intentionally omit Torch.
    _torch = None


if _torch is None:
    class FlowTerminalTorchFixtureTests(unittest.TestCase):
        @unittest.skip("optional local PyTorch unavailable; cloud CPU fixture covers this path")
        def test_514_parameter_adamw_decay_frozen_confidence_and_strict_roundtrip(self):
            pass
else:
    class FlowTerminalTorchFixtureTests(unittest.TestCase):
        class HeadDouble(_torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = _torch.nn.Sequential(
                    _torch.nn.Conv2d(256, 256, 1),
                    _torch.nn.Identity(), _torch.nn.Identity(), _torch.nn.ReLU(),
                    _torch.nn.Conv2d(256, 3, 1, bias=True),
                )

            def forward(self, tokens, crop_size):
                pred = self.head(tokens[2])
                return pred[:, :2], _torch.sigmoid(pred[:, 2])

        def test_514_parameter_adamw_decay_frozen_confidence_and_strict_roundtrip(self):
            torch = _torch
            torch.manual_seed(4)
            attestation = flow_terminal.configure_runtime_policy(
                flow_terminal.continuation_config(TrainConfig(device="cpu")), torch)
            self.assertEqual(attestation["threads"], 1)
            self.assertEqual(attestation["interop_threads"], 1)
            self.assertTrue(attestation["deterministic_algorithms"])
            official = self.HeadDouble()
            official_state = {key: value.detach().cpu().clone()
                              for key, value in official.state_dict().items()}
            head = self.HeadDouble()
            head.load_state_dict(official_state, strict=True)
            terminal = flow_terminal.attach_flow_terminal(head, torch)
            parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
            self.assertEqual(sum(parameter.numel() for parameter in parameters), 514)
            self.assertEqual(len(parameters), 2)
            frozen_weight = terminal.confidence_weight.detach().clone()
            frozen_bias = terminal.confidence_bias.detach().clone()
            frozen_trunk = head.head[0].weight.detach().clone()

            # A zero-gradient AdamW step demonstrates decoupled decay touches the
            # two flow parameters, while confidence buffers and frozen trunk stay put.
            optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.1)
            for parameter in parameters:
                parameter.grad = torch.zeros_like(parameter)
            before_flow = terminal.flow_weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before_flow, terminal.flow_weight))
            self.assertTrue(torch.equal(frozen_weight, terminal.confidence_weight))
            self.assertTrue(torch.equal(frozen_bias, terminal.confidence_bias))
            self.assertTrue(torch.equal(frozen_trunk, head.head[0].weight))

            tokens = [None] * 12
            tokens[2] = torch.randn(1, 256, 2, 2)
            flow, confidence = head(tokens, (2, 2))
            (flow.square().mean() + confidence.mean()).backward()
            optimizer.step()
            ordinary = flow_terminal.ordinary_head_state(head)
            fresh = self.HeadDouble()
            fresh.load_state_dict(ordinary, strict=True)
            with torch.inference_mode():
                actual = head(tokens, (2, 2))
                roundtrip = fresh(tokens, (2, 2))
            self.assertTrue(torch.equal(actual[0], roundtrip[0]))
            self.assertTrue(torch.equal(actual[1], roundtrip[1]))

            resumed = self.HeadDouble()
            resumed.load_state_dict(official_state, strict=True)
            resumed_terminal = flow_terminal.attach_flow_terminal(resumed, torch)
            flow_terminal.restore_flow_state(resumed, ordinary, official_state)
            self.assertTrue(torch.equal(resumed_terminal.flow_weight, terminal.flow_weight))
            self.assertTrue(torch.equal(resumed_terminal.flow_bias, terminal.flow_bias))
            resumed_optimizer = torch.optim.AdamW(
                [p for p in resumed.parameters() if p.requires_grad], lr=0.1, weight_decay=0.1)
            resumed_optimizer.load_state_dict(optimizer.state_dict())
            self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
