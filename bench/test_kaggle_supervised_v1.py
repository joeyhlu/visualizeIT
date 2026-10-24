"""Model-free supervision evidence. No Torch, GPU, training or downloads."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_supervised_v1 as bridge
from scripts import kaggle_adaptation as k
from scripts import kaggle_supervision_v1 as supervision

ROOT = Path(__file__).resolve().parents[1]


def request_sources():
    return dict(bridge=p.pin(Path(bridge.__file__)),
        supervisor=p.pin(Path(supervision.__file__)), scanner=p.pin(Path(raw.__file__)))


def request_base():
    return dict(source_root=str(ROOT), adapter=p.pin(ROOT / 'scripts/kaggle_adaptation.py'),
        supervision_sources=request_sources(), storage_limit_bytes=18*p.GIB,
        limits=dict(max_seconds=1200, max_rss_bytes=8*p.GIB,
            max_output_bytes=8*p.GIB, reserve_bytes=p.GIB, max_log_bytes=1024**2))


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.cache/supervision-model-free-tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.facade = supervision.Facade(k, raw)

    def limits(self, **changes):
        return dict(max_seconds=3, max_rss_bytes=8*p.GIB, max_output_bytes=1024**2,
            reserve_bytes=1, max_log_bytes=65536, **changes) if not changes else {
                **self.limits(), **changes}

    def run_child(self, program, *, limits=None, scanner=raw, adapter=k):
        return supervision.supervise(adapter, scanner, [sys.executable, '-I', '-B', '-c', program],
            cwd=self.root, output_root=self.root, working=self.root,
            log_path=self.root/'child.log', limits=limits or self.limits(), env=k.clean_environment())

    def test_facade_authenticates_original_sources(self):
        self.assertIs(self.facade.adapter, k)
        self.assertIs(self.facade.scanner, raw)
        with patch.object(k, '__file__', __file__):
            with self.assertRaisesRegex(ValueError, 'adapter changed'):
                supervision.Facade(k, raw)

    def test_atomic_rename_retries_full_original_runtime_scan(self):
        roots = (self.root,)
        links = {self.root/'link': self.root/'target'}
        calls = []
        def scan(observed, *, approved_links):
            calls.append((observed, approved_links))
            if len(calls) < 3:
                raise FileNotFoundError('atomic rename')
            return 123
        adapter = SimpleNamespace(unique_tree_bytes=scan)
        self.assertEqual(supervision.stable_tree_bytes(adapter, roots, approved_links=links), 123)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(value == roots and policy is links for value, policy in calls))

    def test_persistent_runtime_churn_fails_after_three_complete_attempts(self):
        with patch.object(k, 'unique_tree_bytes', side_effect=FileNotFoundError('churn')) as scan:
            with self.assertRaises(FileNotFoundError):
                self.facade.unique_tree_bytes((self.root,))
        self.assertEqual(scan.call_count, 3)

    def test_runtime_permission_errors_never_retry(self):
        with patch.object(k, 'unique_tree_bytes', side_effect=PermissionError('denied')) as scan:
            with self.assertRaises(PermissionError):
                self.facade.unique_tree_bytes((self.root,))
        self.assertEqual(scan.call_count, 1)

    def test_original_output_scanner_restarts_full_scan(self):
        original = raw._storage_scan
        calls = []
        (self.root/'regular').write_bytes(b'12345')
        def renamed(roots):
            calls.append(roots)
            if len(calls) == 1:
                temporary = self.root/'atomic.tmp'
                temporary.write_bytes(b'123')
                os.replace(temporary, self.root/'published')
                raise FileNotFoundError('entry disappeared after enumeration')
            return original(roots)
        with patch.object(raw, '_storage_scan', side_effect=renamed):
            self.assertEqual(raw._usage((self.root,)), 8)
        self.assertEqual(len(calls), 2)

    def test_output_scanner_persistent_churn_is_cancellation(self):
        with patch.object(raw, '_storage_scan', side_effect=FileNotFoundError('churn')) as scan:
            with self.assertRaises(raw.ResourceError):
                raw._usage((self.root,))
        self.assertEqual(scan.call_count, 3)

    def test_hardlink_dedup_and_nested_roots(self):
        target = self.root/'target'
        target.write_bytes(b'abcdef')
        try:
            os.link(target, self.root/'alias')
        except PermissionError:
            if os.name == 'nt':
                self.skipTest('Windows sandbox denies hardlink creation; required on Linux')
            raise
        self.assertEqual(raw._usage((self.root, target)), 6)
        self.assertEqual(self.facade.unique_tree_bytes((self.root, target)), 6)

    @unittest.skipIf(os.name == 'nt', 'Linux production symlink policy')
    def test_output_symlink_fails_closed_runtime_approved_link_allowed(self):
        target = self.root/'target'
        target.write_bytes(b'abcdef')
        link = self.root/'link'
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            raw._usage((self.root,))
        with self.assertRaises(k.ContractError):
            self.facade.unique_tree_bytes((self.root,))
        self.assertEqual(self.facade.unique_tree_bytes((self.root,), approved_links={link:target}), 6)

    def test_completed_child_reader_log_lifetime(self):
        result = self.run_child("print('complete')")
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['log_errors'], [])
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual((self.root/'child.log').read_text().strip(), 'complete')

    def test_deadline_cancels_child(self):
        result = self.run_child('import time;time.sleep(20)', limits=self.limits(max_seconds=.1))
        self.assertEqual(result['reason'], 'deadline_exceeded', result)
        self.assertEqual(result['status'], 'incomplete')

    def test_rss_cancels_child(self):
        with patch.object(k, 'group_rss', return_value=100):
            result = self.run_child('import time;time.sleep(20)', limits=self.limits(max_rss_bytes=1))
        self.assertEqual(result['reason'], 'process_group_rss_exceeded', result)

    def test_output_cancels_child(self):
        result = self.run_child("import pathlib,time;pathlib.Path('payload').write_bytes(b'a'*10000);time.sleep(20)",
            limits=self.limits(max_output_bytes=10))
        self.assertEqual(result['reason'], 'output_storage_exceeded', result)
        self.assertTrue((self.root/'payload').is_file())

    def test_log_cap_cancels_child(self):
        result = self.run_child("import sys,time;sys.stdout.write('x'*100000);sys.stdout.flush();time.sleep(20)",
            limits=self.limits(max_log_bytes=10))
        self.assertEqual(result['reason'], 'log_limit_or_log_io_error', result)
        self.assertEqual((self.root/'child.log').stat().st_size, 10)

    def test_reserve_cancels_child(self):
        with patch.object(k, 'free_by_filesystem', return_value={1:0}):
            result = self.run_child('import time;time.sleep(20)')
        self.assertEqual(result['reason'], 'filesystem_reserve_exhausted', result)

    def test_baseexception_scanner_cancellation_preserves_reason(self):
        with patch.object(raw, '_usage', side_effect=raw.ResourceError('three unstable scans')):
            result = self.run_child('import time;time.sleep(20)')
        self.assertEqual(result['status'], 'incomplete', result)
        self.assertEqual(result['reason'], 'ResourceError: three unstable scans')
        self.assertEqual(result['cleanup_errors'], [])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process-group evidence')
    def test_descendant_process_group_is_reaped_and_pipe_drains(self):
        program = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',\"import time;time.sleep(20)\"]);print('spawned',flush=True);time.sleep(20)"
        started = time.monotonic()
        result = self.run_child(program, limits=self.limits(max_seconds=.2))
        self.assertLess(time.monotonic()-started, 10)
        self.assertEqual(result['reason'], 'deadline_exceeded', result)
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(result['log_errors'], [])


class BridgeTests(unittest.TestCase):
    def test_actual_provenance_hash_mismatch_rejected(self):
        request = request_base()
        provenance = bridge.verify_supervision(request)
        self.assertEqual(provenance['validation_adapter'], request['adapter'])
        changed = copy.deepcopy(request)
        changed['supervision_sources']['bridge']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
            bridge.verify_supervision(changed)

    def test_foreign_actual_source_path_rejected(self):
        request = request_base()
        request['supervision_sources']['bridge'] = request['supervision_sources']['supervisor']
        with self.assertRaisesRegex(ValueError, 'actually loaded'):
            bridge.verify_supervision(request)

    def test_original_raw_closure_and_entrypoints_unchanged(self):
        request = request_base()
        self.assertEqual(p.sha256(Path(raw.__file__)), supervision.SCANNER_SHA256)
        self.assertEqual(p.sha256(Path(k.__file__)), supervision.ADAPTER_SHA256)
        for relative, digest in bridge.ORIGINAL_HELPERS.items():
            self.assertEqual(p.sha256(ROOT/relative), digest)
        self.assertNotIn('bench/quality_adaptation_supervised_v1.py', raw.source_paths())
        self.assertNotIn('scripts/kaggle_supervision_v1.py', raw.source_paths())
        self.assertEqual(request['source_root'], str(ROOT))

    def test_extraction_failure_preserved_without_finalization(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temporary:
            root = Path(temporary)
            request = dict(features_root=str(root/'features'), limits=dict(max_seconds=10))
            state = dict(captures=[dict(id='x',split='train',status='available')])
            failure = dict(status='failed', returncode=-15, reason='primary accounting failure')
            with patch.object(bridge, 'verify_execution'), patch.object(bridge, 'crop_manifest', return_value={'path':'manifest'}), \
                 patch.object(bridge, 'preparation_lock', return_value=nullcontext()), patch.object(bridge, 'production_run', return_value=failure):
                self.assertIs(bridge.extract(request,state,root), failure)
                self.assertFalse((root/'features/extraction-binding.json').exists())

    def test_extract_exception_not_masked_by_finalizer(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temporary:
            root = Path(temporary)
            request = dict(features_root=str(root/'features'), limits=dict(max_seconds=10))
            state = dict(captures=[dict(id='x',split='train',status='available')])
            with patch.object(bridge, 'verify_execution'), patch.object(bridge, 'crop_manifest', return_value={'path':'manifest'}), \
                 patch.object(bridge, 'preparation_lock', return_value=nullcontext()), patch.object(bridge, 'production_run', side_effect=raw.ResourceError('primary')):
                with self.assertRaises(raw.ResourceError):
                    bridge.extract(request,state,root)

    def test_numerical_labels_loop_is_exact_original_ast(self):
        def loop(function):
            tree = ast.parse(inspect.getsource(function))
            return next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                and isinstance(node.target,ast.Tuple) and ast.unparse(node.target)=='(ordinal, source)')
        self.assertEqual(ast.dump(loop(bridge.labels)), ast.dump(loop(p.labels)))
        text = inspect.getsource(bridge.labels)
        self.assertLess(text.index('p.verify_prelabel('), text.index("metadata(request['label_plan'])"))

    def test_incomplete_exit_zero_assembler_cannot_publish_label_receipt(self):
        # Execute the actual assembler/finalization tail without importing neural
        # dependencies. A zero exit code does not override a watchdog cancellation.
        function = ast.parse(inspect.getsource(bridge.labels)).body[0]
        first = next(i for i, node in enumerate(function.body)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == 'result' for t in node.targets))
        tail = ast.FunctionDef(name='assembly_tail', args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=function.body[first:], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[]))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temporary:
            root = Path(temporary)
            result = dict(status='incomplete', reason='log_limit_or_log_io_error', returncode=0)
            namespace = dict(bridge.labels.__globals__, features=root, execution=root,
                request={}, production_run=lambda *args, **kwargs: result)
            exec(compile(module, '<actual-label-assembly-tail>', 'exec'), namespace)
            with self.assertRaisesRegex(ValueError, 'assembler did not complete'):
                namespace['assembly_tail']()
            self.assertEqual(p.read_json(root/'assemble-observation.json'), result)
            self.assertFalse((root/'label-receipt.json').exists())

    def test_all_execution_routes_use_explicit_facade(self):
        for function in (bridge.production_run, bridge.run_labels, bridge.execute, bridge.pilot_execute):
            text = inspect.getsource(function)
            self.assertIn('source_runtime(', text)
            self.assertNotIn('p.source_runtime(', text)
        text = inspect.getsource(bridge.pilot_execute)
        self.assertIn("execute('train', bounded", text)
        self.assertIn('old.inspect_training(k,', text)
        self.assertIn('k.unique_tree_bytes(roots, approved_links=links)', text)
        self.assertIn('old.reconcile(k,', text)
        self.assertIn('full_scope_bound(', text)
        self.assertIn('checkpoint = p.pin(', text)

    def test_training_stage_wiring_and_original_resume_export_validation(self):
        # Deliberate source-contract checks supplement subprocess guard tests.
        # They do not claim any real checkpoint/training behavior.
        text = inspect.getsource(bridge.execute)
        self.assertIn("if stage not in ('preflight', 'train', 'resume', 'export')", text)
        self.assertIn('joined = verify_prepared(request)', text)
        self.assertIn('k.validate_limits(', text)
        self.assertIn('p.quota_check(request, k', text)
        self.assertIn('verify_resume(output, joined[', text)
        self.assertIn('checkpoint_receipt(output, joined[', text)
        self.assertIn('k.command(stage, training_request(request))', text)
        self.assertIn('k.supervise(', text)
        self.assertIs(bridge.verify_resume, bridge.prepared.verify_resume)
        self.assertIs(bridge.checkpoint_receipt, bridge.prepared.checkpoint_receipt)
        self.assertIs(bridge.training_request, bridge.prepared.training_request)

    def test_prepared_validator_stays_authoritative(self):
        request = request_base()
        request['features_root'] = str(ROOT)
        value = object()
        with patch.object(bridge, 'verify_binding') as binding, \
             patch.object(bridge.prepared, 'verify_prepared', return_value=value) as original:
            self.assertIs(bridge.verify_prepared(request,payloads=False), value)
        binding.assert_called_once_with(request, request['features_root'])
        original.assert_called_once_with(request,payloads=False)

    def test_pilot_step_cap_is_original_one_to_four(self):
        request = dict(schema='kaggle_pilot_request_v2',training=dict(limits=request_base()['limits']))
        request['training']['limits']['max_seconds'] = 900
        request['training']['limits']['max_output_bytes'] = 2*p.GIB
        for steps in (1,2,3,4):
            request['stop_after_steps'] = steps
            self.assertEqual(bridge.pilot_v2.validate_caps(request), steps)
        for steps in (0,5,True,1.5):
            request['stop_after_steps'] = steps
            with self.assertRaises(ValueError):
                bridge.pilot_v2.validate_caps(request)

    def test_no_production_runtime_patch(self):
        for module in (bridge,supervision):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                    targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets:
                        self.assertFalse(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                            and target.value.id in ('p','raw','k','prepared','pilot_v2'))


if __name__ == '__main__':
    unittest.main()
