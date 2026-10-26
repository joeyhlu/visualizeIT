"""No-model tests for the finite, fail-closed offline diagnostic adapter."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PATH=Path(__file__).with_name('tracking_neural_diagnostic.py')
spec=importlib.util.spec_from_file_location('tracking_neural_diagnostic_test_target',PATH)
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class SafeLoaderTests(unittest.TestCase):
    def test_forces_safe_cpu_weights_only_even_when_production_requests_false(self):
        calls=[]
        def load(*args,**kwargs):calls.append((args,kwargs));return 'safe'
        torch=SimpleNamespace(load=load)
        with diagnostic.safe_base_loader(torch,Path('/tmp/base.pt')):
            self.assertEqual(torch.load('/tmp/base.pt',weights_only=False,map_location='cuda'),'safe')
        self.assertIs(torch.load,load)
        self.assertTrue(calls[0][1]['weights_only'])
        self.assertEqual(calls[0][1]['map_location'],'cpu')

    def test_other_checkpoint_path_rejected_and_hook_restored_on_failure(self):
        original=lambda *a,**kw:None
        torch=SimpleNamespace(load=original)
        with self.assertRaises(ValueError):
            with diagnostic.safe_base_loader(torch,Path('/tmp/base.pt')):
                torch.load('/tmp/other.pt')
        self.assertIs(torch.load,original)


class TensorComparisonTests(unittest.TestCase):
    def make_case(self,root,name,*,delta=0.,query_delta=0.,dtype=np.float32,forward=True):
        arrays={k:np.zeros((1,2,3),dtype=dtype) for k in ('query','template','rendered_mask','flow','confidence')}
        arrays['flow']+=delta;arrays['query']+=query_delta
        path=root/(name+'.npz');np.savez_compressed(path,**arrays)
        return dict(caseId='case',forwards=[dict(path=path.name)] if forward else [],poseMetres=np.eye(4).tolist())

    def test_reference_export_identical_and_baseline_difference_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.make_case(root,'a');b=self.make_case(root,'b');c=self.make_case(root,'c',delta=2.)
            result=diagnostic.compare_case(root,a,b,c)
            self.assertTrue(result['referenceExportParity'])
            self.assertTrue(result['baselineVsAdapted']['firstInputsByteEqual'])
            self.assertEqual(result['baselineVsAdapted']['firstFlowMaxAbsDifference'],2.)

    def test_numerically_close_but_different_inputs_are_not_exact_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.make_case(root,'a');b=self.make_case(root,'b',query_delta=1e-8);c=self.make_case(root,'c')
            self.assertFalse(diagnostic.compare_case(root,a,b,c)['referenceExportParity'])

    def test_dtype_mismatch_or_no_forward_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.make_case(root,'a');b=self.make_case(root,'b',dtype=np.float64);c=self.make_case(root,'c')
            self.assertFalse(diagnostic.compare_case(root,a,b,c)['referenceExportParity'])
            b=self.make_case(root,'b',forward=False)
            self.assertFalse(diagnostic.compare_case(root,a,b,c)['referenceExportParity'])


@unittest.skipUnless(importlib.util.find_spec('psutil') is not None,
                     'psutil belongs to the optional isolated neural runtime')
class SupervisorTests(unittest.TestCase):
    def supervise_code(self,code,*,max_seconds=2.,max_rss=128*1024**2):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'out';real_popen=subprocess.Popen
            def popen(_cmd,**kwargs):return real_popen([sys.executable,'-c',code],**kwargs)
            with patch.object(diagnostic.subprocess,'Popen',side_effect=popen):
                try:diagnostic.supervise(root/'not-used-plan.json',output,max_seconds,max_rss)
                except SystemExit:pass
            return diagnostic.read(output/'supervision.json')

    def test_success_exit_and_rss_evidence(self):
        result=self.supervise_code('print("bounded process success")')
        self.assertEqual(result['exitCode'],0)
        self.assertIsNone(result['killedReason'])
        self.assertLess(result['elapsedSeconds'],2.)

    def test_wall_timeout_terminates_child_and_preserves_reason(self):
        result=self.supervise_code('import time;time.sleep(30)',max_seconds=.2)
        self.assertEqual(result['killedReason'],'wall-time limit exceeded')
        self.assertNotEqual(result['exitCode'],0)
        self.assertLess(result['elapsedSeconds'],6.)

    def test_rss_cap_terminates_child_and_preserves_reason(self):
        result=self.supervise_code('import time;a=bytearray(20*1024**2);time.sleep(30)',max_rss=1024**2)
        self.assertEqual(result['killedReason'],'process-group RSS limit exceeded')
        self.assertNotEqual(result['exitCode'],0)

    def test_unbounded_limits_rejected_before_process_launch(self):
        for seconds,rss in [(0,1024**2),(901,1024**2),(1,8*1024**3)]:
            with self.subTest(seconds=seconds,rss=rss),self.assertRaises(ValueError):
                diagnostic.supervise('not-read','not-created',seconds,rss)


if __name__=='__main__':unittest.main()
