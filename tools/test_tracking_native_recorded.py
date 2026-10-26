"""Tiny supervision/security tests; no model, renderer, weights or CUDA."""
import json
import importlib.util
import copy
import gc
import weakref
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import subprocess
import os
import signal
import sys
import tempfile
import unittest

from tools import tracking_native_recorded as runner


class SecurityAndPolicyTests(unittest.TestCase):
    def test_safe_base_scope_leaves_head_loads_untouched_and_restores(self):
        calls=[]
        def load(*a,**kw):calls.append((a,kw));return 'value'
        torch=SimpleNamespace(load=load)
        def base(device):return torch.load('verified-base',weights_only=False,map_location=device)
        neural=SimpleNamespace(load_base_network=base)
        with runner.safe_base_loader(neural,torch):
            self.assertIs(torch.load,load)
            self.assertEqual(neural.load_base_network('cuda'),'value')
            self.assertIs(torch.load,load)
            torch.load('head-BytesIO',weights_only=True,map_location='cpu')
        self.assertIs(neural.load_base_network,base)
        self.assertEqual(calls[0][1],dict(weights_only=True,map_location='cpu'))
        self.assertEqual(calls[1],(('head-BytesIO',),dict(weights_only=True,map_location='cpu')))

    def test_safe_load_failure_never_falls_back(self):
        calls=[]
        def load(*a,**kw):calls.append(kw);raise ValueError('unsafe object rejected')
        torch=SimpleNamespace(load=load)
        def base():return torch.load('base',weights_only=False)
        neural=SimpleNamespace(load_base_network=base)
        with self.assertRaises(ValueError),runner.safe_base_loader(neural,torch):neural.load_base_network()
        self.assertIs(torch.load,load);self.assertIs(neural.load_base_network,base)
        self.assertEqual(calls,[dict(weights_only=True,map_location='cpu')])

    def test_profiles_are_measured_bounded_and_stage_specific(self):
        self.assertEqual(runner.stage_budget('pilot','run'),(900,900))
        self.assertEqual(runner.stage_budget('recorded300','segment'),(1800,1800))
        self.assertEqual(runner.stage_budget('recorded300','run'),(2700,2700))
        self.assertEqual(runner.stage_budget('recorded300','gate'),(900,900))
        for profile,command,value in [('pilot','run',901),('recorded300','segment',1801),('recorded300','run',2701),('recorded300','gate',901),('pilot','run',0)]:
            with self.assertRaises(ValueError):runner.stage_budget(profile,command,value)

    def test_pair_budget_and_real_native_run_contract(self):
        self.assertEqual(runner.stage_budget('recorded300pair','run',roles='primary_neural,frozen_baseline'),(5400,5400))
        for roles in (None,'frozen_baseline','primary_neural','frozen_baseline,primary_neural'):
            with self.assertRaises(ValueError):runner.stage_budget('recorded300pair','run',roles=roles)
        with self.assertRaises(ValueError):runner.stage_budget('recorded300pair','run',5401,'primary_neural,frozen_baseline')
        from bench import workbench_contract as wc,workbench_cli as cli
        from bench.test_workbench_contract import clip_document
        clip=wc.ClipSpec.from_dict(clip_document())
        normal=cli.build_run_spec(clip,mode='regression',roles=('primary_neural','frozen_baseline'))
        self.assertEqual(wc.RunSpec.from_dict(normal.to_dict()).roles,('primary_neural','frozen_baseline'))
        with self.assertRaisesRegex(wc.ContractError,'comparator'):
            cli.build_run_spec(clip,mode='regression',roles=('frozen_baseline',))

    def test_all_resource_bounds(self):
        defaults=dict(rss=1,elapsed=1,free_disk=100,output_bytes=1,max_seconds=2,max_rss=2,max_output=2,min_free_disk=50)
        self.assertIsNone(runner.guard_reason(**defaults))
        for change,message in [({'rss':3},'RSS'),({'elapsed':3},'wall-time'),({'free_disk':49},'reserve'),({'output_bytes':3},'output')]:
            self.assertIn(message,runner.guard_reason(**{**defaults,**change}))

    def test_native_plan_rejects_unrelated_commands_and_watches_derived_outputs(self):
        with self.assertRaises(ValueError):runner.native_plan(['inspect'])
        command,paths,_roles=runner.native_plan(['--workspace','/tmp/native-recorded-test','segment','clip','--mask-set','m'])
        self.assertEqual(command,'segment');self.assertIn(Path('/tmp/native-recorded-test/masks'),paths)

    def test_non_cpu_device_is_refused_before_model_execution(self):
        with self.assertRaisesRegex(ValueError,'CPU-only'):
            runner.native_plan(['segment','clip','--mask-set','m','--device','cuda'])

    def test_observational_hooks_do_not_modify_or_capture_outputs(self):
        class Handle:
            def __init__(self):self.removed=False
            def remove(self):self.removed=True
        class Module:
            def register_forward_hook(self,fn):self.fn=fn;self.handle=Handle();return self.handle
        class Network:pass
        net=Network()
        for key in ('backbone','decoder','pose_head'):setattr(net,key,Module())
        loader=lambda device:net;gotrack=SimpleNamespace(load_network=loader)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'counts.json'
            with runner.observe_network_loads(gotrack,p) as counts:
                self.assertIs(gotrack.load_network('cpu'),net)
                for key in ('backbone','decoder','pose_head'):
                    self.assertIsNone(getattr(net,key).fn(None,('input',),'unchanged-output'))
                self.assertEqual(counts,{'network_0':dict(backbone=1,decoder=1,pose_head=1)})
            self.assertIs(gotrack.load_network,loader)
            self.assertTrue(all(getattr(net,key).handle.removed for key in ('backbone','decoder','pose_head')))
            self.assertEqual(json.loads(p.read_text()),counts)


    def test_deepcopied_installed_head_hook_cleanup_preserves_other_hooks(self):
        class Handle:
            def __init__(self,hooks,key):self.hooks,self.key=hooks,key
            def remove(self):self.hooks.pop(self.key,None)
        class Module:
            def __init__(self):self._forward_hooks={}
            def register_forward_hook(self,fn):
                key=len(self._forward_hooks);self._forward_hooks[key]=fn
                return Handle(self._forward_hooks,key)
        class Network:pass
        net=Network()
        for key in ('backbone','decoder','pose_head'):setattr(net,key,Module())
        gotrack=SimpleNamespace(load_network=lambda device:net)
        with tempfile.TemporaryDirectory() as d:
            with runner.observe_network_loads(gotrack,Path(d)/'counts.json') as counts:
                loaded=gotrack.load_network('cpu');loaded.pose_head=copy.deepcopy(loaded.pose_head)
                unrelated=lambda *_:None;loaded.pose_head.register_forward_hook(unrelated)
                for key in ('backbone','decoder','pose_head'):
                    for fn in getattr(loaded,key)._forward_hooks.values():fn(None,None,None)
                self.assertEqual(counts['network_0']['pose_head'],1)
            self.assertEqual(list(loaded.pose_head._forward_hooks.values()),[unrelated])
            self.assertFalse(loaded.backbone._forward_hooks)

    def test_observer_does_not_retain_prior_network(self):
        class Handle:
            def remove(self):pass
        class Module:
            def register_forward_hook(self,fn):return Handle()
        class Network:pass
        net=Network()
        for key in ('backbone','decoder','pose_head'):setattr(net,key,Module())
        reference=weakref.ref(net);gotrack=SimpleNamespace(load_network=lambda device:net)
        with tempfile.TemporaryDirectory() as d,runner.observe_network_loads(gotrack,Path(d)/'counts.json'):
            loaded=gotrack.load_network('cpu');net=None;del loaded;gc.collect()
            self.assertIsNone(reference())

    def test_sampled_accounting_ignores_only_a_disappeared_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'counts.json.part';target.write_text('temporary');original=Path.stat;calls=0
            def vanished(path,*args,**kwargs):
                nonlocal calls
                if path==target:
                    calls+=1
                    if calls>=2:raise FileNotFoundError('atomic publication removed temp')
                return original(path,*args,**kwargs)
            with patch.object(Path,'stat',vanished):self.assertEqual(runner.output_size([Path(d)]),0)

    def test_accounting_permission_errors_remain_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'counts.json.part';target.write_text('temporary');original=Path.stat;calls=0
            def denied(path,*args,**kwargs):
                nonlocal calls
                if path==target:
                    calls+=1
                    if calls>=2:raise PermissionError('must not hide accounting errors')
                return original(path,*args,**kwargs)
            with patch.object(Path,'stat',denied),self.assertRaises(PermissionError):runner.output_size([Path(d)])


@unittest.skipUnless(importlib.util.find_spec('psutil'), 'optional supervisor psutil dependency is absent')
class TinyProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def run_child(self,program,**kw):
        return runner.supervise([sys.executable,'-c',program],self.root/'run',root=self.root,
            **{**dict(min_free_disk=0,max_seconds=2,sample_seconds=.02),**kw})

    def test_success_and_nonzero_exit_are_preserved(self):
        result=self.run_child("print('tiny, no model')")
        self.assertEqual(result['exitCode'],0);self.assertIsNone(result['killedReason'])
        self.assertIn('tiny',(self.root/'run/worker.log').read_text())
        with self.assertRaises(FileExistsError):self.run_child('pass')

    def test_nonzero_child_is_not_reported_as_success(self):
        result=self.run_child('raise SystemExit(7)')
        self.assertEqual(result['exitCode'],7);self.assertIsNone(result['killedReason'])

    def test_timeout_stops_child_and_keeps_partial_evidence(self):
        result=runner.supervise([sys.executable,'-c','import time;time.sleep(10)'],self.root/'run',root=self.root,
            min_free_disk=0,max_seconds=.05,sample_seconds=.01)
        self.assertIn('wall-time',result['killedReason']);self.assertNotEqual(result['exitCode'],0)
        self.assertTrue((self.root/'run/supervision.json').exists());self.assertLess(result['elapsedSeconds'],2)

    def test_rss_and_output_caps_stop_tiny_children(self):
        result=self.run_child('import time;time.sleep(10)',max_rss=1)
        self.assertIn('RSS',result['killedReason'])
        # Separate immutable output directory for another supervised child.
        result=runner.supervise([sys.executable,'-c',"import time;print('x'*4096,flush=True);time.sleep(10)"],self.root/'other',root=self.root,
            min_free_disk=0,max_seconds=2,sample_seconds=.01,max_output=3000)
        self.assertIn('output',result['killedReason'])

    def check_orphan_cleanup(self, escaped=False):
        import psutil
        program = f"import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session={escaped!r});print(p.pid,flush=True);time.sleep(.08)"
        result=self.run_child(program)
        pid=int((self.root/'run/worker.log').read_text().strip())
        try:
            self.assertEqual(result['exitCode'],0)
            self.assertIn('descendants',result['killedReason'])
            for _ in range(50):
                try:
                    status=psutil.Process(pid).status()
                except psutil.NoSuchProcess:
                    break
                if status==psutil.STATUS_ZOMBIE:break
                import time;time.sleep(.01)
            else:self.fail('owned descendant was not stopped')
        finally:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass

    def test_descendant_cannot_outlive_successful_root(self):
        self.check_orphan_cleanup()

    def test_observed_detached_descendant_is_also_stopped(self):
        self.check_orphan_cleanup(escaped=True)

    def test_source_drift_is_recorded(self):
        p=self.root/'source.txt';p.write_text('before')
        result=self.run_child(f"from pathlib import Path;Path({str(p)!r}).write_text('after')",frozen_sources={'source.txt':runner.digest(p)})
        self.assertEqual(result['sourceDrift'],['source.txt'])

    def test_raised_wall_bound_requires_explicit_declared_ceiling(self):
        with self.assertRaises(ValueError):self.run_child('pass',max_seconds=1800)
        result=runner.supervise([sys.executable,'-c','pass'],self.root/'full',root=self.root,min_free_disk=0,
            max_seconds=1800,wall_ceiling=1800,sample_seconds=.02)
        self.assertEqual(result['exitCode'],0)


if __name__=='__main__':unittest.main()
