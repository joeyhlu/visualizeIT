"""CPU-only integration checks; analytical candidates, no models or training."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

from .quality_cloud import estimate_cost, require_cloud
from .quality_pipeline import CandidateTraceError, CandidateTraceSink, RankingOptions
from .quality_offline import run_stage, stage_commands


def _psutil_test_double():
    class Process:
        def __init__(self, _pid):
            pass

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def children(self, recursive=False):
            return []

    return SimpleNamespace(Process=Process, NoSuchProcess=type('NoSuchProcess', (Exception,), {}),
        AccessDenied=type('AccessDenied', (Exception,), {}),
        wait_procs=lambda processes, timeout=None: (processes, []))


class PipelineTests(unittest.TestCase):
    def test_cost_counts_preparation_and_storage(self):
        result = estimate_cost(gpu_hours=2, gpu_usd_per_hour=.5, storage_gib=10,
                               storage_usd_per_gib_month=.1, storage_days=30)
        # Arithmetic: 2*.5 GPU + 10*.1*(30/30) storage = $2.
        self.assertEqual(result['estimated_out_of_pocket_usd'], 2)
        self.assertFalse(result['paid_resource_approval'])

    def test_cloud_guard_rejects_local(self):
        with patch('bench.quality_cloud.platform.system', return_value='Windows'):
            with self.assertRaises(RuntimeError): require_cloud()

    def test_trace_retains_empty_frame_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'trace.jsonl'
            sink = CandidateTraceSink(target, object_id='mug', sequence_id='seq',
                provenance={'test_fixture': True}, source_frame_ids={0: 127})
            sink({'frame_id': 0, 'candidates': [], 'render_state': 'suppressed',
                  'memory': {'last_valid_id': None}})
            row = json.loads(target.read_text())
            self.assertEqual(row['source_frame_id'], 127)
            self.assertEqual(row['candidates'], [])
            self.assertEqual(row['object_id'], 'mug')
            self.assertEqual(row['memory'], {'last_valid_id': None})
            self.assertIn('resource_memory', row)
            with self.assertRaises(FileExistsError):
                CandidateTraceSink(target, object_id='mug', sequence_id='seq', provenance={})

    def test_trace_budget_fails_before_append(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'trace.jsonl'
            sink = CandidateTraceSink(target, object_id='mug', sequence_id='seq', provenance={}, max_bytes=1)
            with self.assertRaises(CandidateTraceError): sink({'frame_id': 0, 'candidates': []})
            self.assertEqual(target.stat().st_size, 0)

    def test_stage_order_and_complete_mode(self):
        with patch('bench.quality_runner.read_input', return_value={'object': 'fixture'}):
            rows = stage_commands(Path('bundle'), Path('output'), stages=['segment', 'pose'], limit=2)
            self.assertEqual([s for s, _ in rows], ['segment', 'pose'])
            command = rows[-1][1]
            self.assertEqual(command[command.index('--mode') + 1], 'complete')
            self.assertNotIn('--controlled-initial-pose', command)
            self.assertNotIn('--gotrack-inference-config', command)
            with self.assertRaises(ValueError):
                stage_commands(Path('bundle'), Path('output'), stages=['pose', 'segment'])

    def test_profile_offline_pose_command_forwards_explicit_profile_and_render_flags(self):
        from types import SimpleNamespace
        context = SimpleNamespace(profile='gotrack_integer_legacy_synthetic_v1',
                                  requested_head_sha256='a' * 64)
        options = object()
        with patch('bench.quality_runner.read_input', return_value={'object': 'mug', 'frame_ids': [0]}), \
             patch('bench.quality_runner._validate_inference_runner_options'), \
             patch('bench.quality_gotrack_inference.options_from_values', return_value=options), \
             patch('bench.quality_gotrack_inference.prepare_inference', return_value=context):
            rows = stage_commands(Path('bundle'), Path('output'), stages=['pose'],
                gotrack_head=Path('head.pt'), gotrack_head_sha256='a' * 64,
                gotrack_inference_config=Path('reviewed.json'),
                gotrack_inference_config_sha256='b' * 64, cloud_execution=True)
        command = rows[0][1]
        for flag in ('--gotrack-inference-config', '--gotrack-inference-config-sha256',
                     '--cloud-execution', '--gotrack-head', '--gotrack-head-sha256',
                     '--unlit-templates', '--disable-multisampling'):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index('--gotrack-inference-config-sha256') + 1], 'b' * 64)
        self.assertEqual(command[command.index('--gotrack-head-sha256') + 1], 'a' * 64)

    def test_configured_pose_stage_accepts_trace_none_without_model_construction(self):
        from types import SimpleNamespace
        from . import quality_runner
        from .quality_gotrack_inference import GoTrackInferenceOptions, PROFILE
        from .quality_time import PHYSICAL
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle'
            bundle.mkdir()
            output = root / 'pose.json'
            context = SimpleNamespace(profile=PROFILE, bundle=bundle.resolve(),
                                      identity_dict=lambda: {'profile': PROFILE})
            expected = object()
            options = GoTrackInferenceOptions(config_path=root / 'config.json',
                                               config_sha256='b' * 64,
                                               cloud_execution=True)
            manifest = {'object': 'mug', '_timing': {'clock_mode': PHYSICAL}}
            with patch.object(quality_runner, 'read_input', return_value=manifest), \
                 patch.object(quality_runner, '_validate_inference_runner_options'), \
                 patch.object(quality_runner, '_pnp_tracking_settings', return_value={}), \
                 patch.object(quality_runner, '_physical_mask_cache', return_value=({}, {})), \
                 patch('bench.quality_gotrack_inference.validate_physical_mask_producer'), \
                 patch.object(quality_runner, '_pose_stage_physical', return_value=expected) as execute:
                result = quality_runner.pose_stage(bundle, root / 'masks', output, 'cpu', 'complete',
                    ranking_options=RankingOptions(), inference_options=options,
                    inference_context=context)
            self.assertIs(result, expected)
            execute.assert_called_once()

    def test_wrong_mask_producer_rejects_before_torch_import_and_writes_failure(self):
        import builtins
        from types import SimpleNamespace
        from . import quality_runner
        from .quality_gotrack_inference import (ARTIFACT_REGISTRY, EXPORT_SHA256,
            EXPECTED_SOURCE18_MASK_PRODUCER, GoTrackInferenceOptions, PROFILE)
        from .quality_time import PHYSICAL
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'pose.json'
            accepted = {'input_sha256': '1' * 64, 'mesh_path': 'object.glb',
                        'mesh_sha256': '2' * 64, 'video_path': 'synthetic.mp4',
                        'video_sha256': '3' * 64}
            producer = copy.deepcopy(EXPECTED_SOURCE18_MASK_PRODUCER)
            record = copy.deepcopy(producer)
            record['input_manifest_sha256'] = accepted['input_sha256']
            record['source_sha256'] = {'object.glb': accepted['mesh_sha256'],
                                       'synthetic.mp4': accepted['video_sha256']}
            record['provenance'] = {'input_manifest_sha256': accepted['input_sha256'],
                                    **copy.deepcopy(producer['provenance'])}
            record['provenance']['adapter_sha256']['quality_sam2'] = 'f' * 64
            identity = {'schema': 'gotrack_inference_contract_v1', 'profile': PROFILE,
                        'config_sha256': 'b' * 64,
                        'head': {'requested_sha256': None,
                                 'base_checkpoint_sha256': 'c' * 64},
                        'head_installation': {'requested': False, 'installed': False}}
            context = SimpleNamespace(profile=PROFILE,
                contract={'mask_producer': producer},
                artifact=ARTIFACT_REGISTRY[EXPORT_SHA256], accepted_input=accepted,
                identity_dict=lambda: copy.deepcopy(identity))
            options = GoTrackInferenceOptions(config_path=root / 'config.json',
                config_sha256='b' * 64, cloud_execution=True)
            manifest = {'object': 'mug', 'frame_ids': [0],
                        '_timing': {'clock_mode': PHYSICAL}}
            argv = ['quality_runner', 'pose', '--bundle', str(root / 'bundle'),
                    '--masks', str(root / 'masks'), '--output', str(output),
                    '--device', 'cpu', '--gotrack-inference-config', str(root / 'config.json'),
                    '--gotrack-inference-config-sha256', 'b' * 64,
                    '--cloud-execution', '--unlit-templates', '--disable-multisampling']
            real_import = builtins.__import__

            def reject_torch(name, *args, **kwargs):
                if name == 'torch':
                    raise AssertionError('torch imported before mask producer validation')
                return real_import(name, *args, **kwargs)

            with patch.object(sys, 'argv', argv), \
                 patch.object(quality_runner, 'read_input', return_value=manifest), \
                 patch.object(quality_runner, '_validate_inference_runner_options'), \
                 patch.object(quality_runner, '_pnp_tracking_settings', return_value={}), \
                 patch.object(quality_runner, '_physical_mask_cache', return_value=(record, {})), \
                 patch('bench.quality_gotrack_inference.options_from_values', return_value=options), \
                 patch('bench.quality_gotrack_inference.prepare_inference', return_value=context), \
                 patch('builtins.__import__', side_effect=reject_torch):
                with self.assertRaises(SystemExit):
                    quality_runner.main()
            failure = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(failure['status'], 'failed')
            self.assertFalse(failure['provenance']['gotrack_inference']['head_installation']['installed'])

    def test_offline_pose_receipt_uses_child_installation_identity_on_failure(self):
        import copy
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            pose_output = output / 'pose.json'
            requested = 'a' * 64
            preflight = {
                'schema': 'gotrack_inference_contract_v1',
                'profile': 'gotrack_integer_legacy_synthetic_v1',
                'config_sha256': 'b' * 64,
                'head': {'requested_sha256': requested,
                         'base_checkpoint_sha256': 'c' * 64,
                         'installed_sha256': None},
                'head_installation': {'requested': True, 'installed': False},
            }
            installed = copy.deepcopy(preflight)
            installed['head']['installed_sha256'] = requested
            installed['head_installation']['installed'] = True
            pose_output.write_text(json.dumps({'provenance': {'gotrack_inference': installed}}),
                                   encoding='utf-8')
            success_dir = output / 'success'
            success_dir.mkdir()
            success_output = success_dir / 'pose.json'
            success_output.write_text(json.dumps({'provenance': {'gotrack_inference': installed}}),
                                      encoding='utf-8')
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}):
                success = run_stage('pose', [sys.executable, '-c', 'print("completed")'],
                    success_dir, timeout_s=10, storage_limit_bytes=1048576,
                    storage_root=output, inference_provenance=preflight,
                    inference_output=success_output)
            self.assertEqual(success['status'], 'complete')
            self.assertEqual(success['gotrack_inference']['head']['installed_sha256'], requested)
            code = 'raise SystemExit(1)'
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}), \
                 self.assertRaises(RuntimeError):
                run_stage('pose', [sys.executable, '-c', code], output,
                    timeout_s=10, storage_limit_bytes=1048576, storage_root=output,
                    inference_provenance=preflight, inference_output=pose_output)
            receipt = json.loads((output / 'pose.receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['inference_identity_source'], 'pose_output')
            self.assertEqual(receipt['gotrack_inference']['head']['installed_sha256'], requested)

    def test_pose_child_zero_exit_without_output_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            preflight = {
                'schema': 'gotrack_inference_contract_v1',
                'profile': 'gotrack_integer_legacy_synthetic_v1',
                'config_sha256': 'b' * 64,
                'head': {'requested_sha256': None,
                         'base_checkpoint_sha256': 'c' * 64,
                         'installed_sha256': None},
                'head_installation': {'requested': False, 'installed': False},
            }
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}), \
                 self.assertRaisesRegex(RuntimeError, 'pose_inference_output_missing'):
                run_stage('pose', [sys.executable, '-c', 'print("done")'], output,
                    timeout_s=10, storage_limit_bytes=1048576, storage_root=output,
                    inference_provenance=preflight,
                    inference_output=output / 'missing-pose.json')
            receipt = json.loads((output / 'pose.receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['failure_reason'], 'pose_inference_output_missing')
            self.assertEqual(receipt['inference_identity_source'], 'preflight_only')

    def test_legacy_options_are_inactive(self):
        self.assertFalse(RankingOptions().validate().active)
        with self.assertRaises(ValueError): RankingOptions('learned').validate()

    def test_ordering_research_requires_explicit_checkpoint_and_frozen_features(self):
        from .quality_pipeline import configure_ranking
        with self.assertRaises(ValueError):
            RankingOptions('ordering-research').validate()
        checkpoint = Path('research.json')
        with patch('bench.quality_pose_ranker.make_ranker') as factory:
            options = RankingOptions('ordering-research', checkpoint).validate()
            refiner = MagicMock()
            ranker, sink = configure_ranking(options, refiner=refiner,
                                             manifest={}, provenance={})
            self.assertTrue(refiner.collect_ranker_features)
            self.assertIs(ranker, factory.return_value)
            self.assertIsNone(sink)
            factory.assert_called_with('ordering-research', checkpoint)

    def test_ordering_research_offline_command_retains_explicit_opt_in(self):
        with patch('bench.quality_runner.read_input', return_value={'object': 'fixture'}), \
             patch('bench.quality_pose_ranker.make_ranker') as factory:
            with self.assertRaises(ValueError):
                stage_commands(Path('bundle'), Path('output'), stages=['pose'],
                               ranker='ordering-research')
            command = stage_commands(Path('bundle'), Path('output'), stages=['pose'],
                                     ranker='ordering-research', checkpoint=Path('research.json'))[0][1]
            self.assertEqual(command[command.index('--ranker') + 1], 'ordering-research')
            self.assertIn('--ranker-checkpoint', command)
            factory.assert_called_with('ordering-research', Path('research.json'))

    def test_virtual_smoke_pose_uses_only_mesh_and_camera(self):
        from .quality_runner import _model_only_smoke_pose
        from .quality_contract import project
        vertices = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.1, .1)])
        k = np.array([[500., 0., 160.], [0., 500., 120.], [0., 0., 1.]])
        pose = _model_only_smoke_pose(vertices, k, 320, 240)
        uv, z = project(vertices, pose, k)
        # A centered cube's virtual view must put every corner in front/in frame.
        self.assertTrue((z > .01).all())
        self.assertTrue(((uv > 0) & (uv < [320, 240])).all())
        np.testing.assert_array_equal(pose[:3, :3], np.eye(3))
        with self.assertRaises(ValueError):
            _model_only_smoke_pose(np.zeros((8, 3)), k, 320, 240)

    def test_physical_segment_cli_dispatches_existing_producer(self):
        from . import quality_runner
        from .quality_time import PHYSICAL
        manifest = {'_timing': {'clock_mode': PHYSICAL}}
        with tempfile.TemporaryDirectory() as folder:
            argv = ['quality_runner', 'segment', '--bundle', 'fixture', '--output', folder, '--device', 'cpu']
            with patch.object(sys, 'argv', argv), \
                 patch.dict(sys.modules, {'torch': MagicMock()}), \
                 patch.object(quality_runner, 'read_input', return_value=manifest), \
                 patch.object(quality_runner, 'device_info', return_value={}), \
                 patch.object(quality_runner, 'CACHE', Path(folder)), \
                 patch('bench.quality_sam2.run') as producer:
                quality_runner.main()
            producer.assert_called_once_with(Path('fixture'), Path(folder), 'cpu', None, None, False)

    def test_stage_process_success_and_failure_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}):
                receipt = run_stage('fixture', [sys.executable, '-c', 'print("fixture only")'], output,
                                    timeout_s=10, storage_limit_bytes=1048576, storage_root=output)
            self.assertEqual(receipt['status'], 'complete')
            self.assertIsNone(receipt['billing_stopped'])
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}), \
                 self.assertRaises(RuntimeError):
                run_stage('failure', [sys.executable, '-c', 'raise SystemExit(3)'], output,
                          timeout_s=10, storage_limit_bytes=1048576, storage_root=output)
            self.assertEqual(json.loads((output / 'failure.receipt.json').read_text())['exit_code'], 3)

    def test_short_stage_over_budget_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            code = 'from pathlib import Path; Path(' + repr(str(output / 'growth')) + ').write_bytes(b"x"*2048)'
            with patch.dict(sys.modules, {'psutil': _psutil_test_double()}), \
                 self.assertRaisesRegex(RuntimeError, 'storage_limit'):
                run_stage('growth', [sys.executable, '-c', code], output,
                          timeout_s=10, storage_limit_bytes=1024, storage_root=output)
            receipt = json.loads((output / 'growth.receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed')


if __name__ == '__main__': unittest.main()
