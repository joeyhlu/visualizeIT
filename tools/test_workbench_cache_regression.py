"""Codex tests for cache-only behavior; never execute a real replay."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock
from tools import workbench_cache_regression as cr


class CacheRegressionTests(unittest.TestCase):
    def sample(self):
        return [{'run_id':'old','backend_role':'primary_neural','frame_id':'s1','pose_m':None,
                 'mask_state':'observed','render_state':'suppressed','timestamp_s':1.,
                 'backend_identity':{'export_sha256':cr.HEAD_SHA256},'metrics':{'tracking_ms':20},
                 'evidence':{'overlay':'evidence/primary_neural/s1.png'}}]

    def test_only_run_id_changes_are_allowed(self):
        a=self.sample();b=copy.deepcopy(a);b[0]['run_id']='new'
        self.assertTrue(cr.compare_rows(a,b)['equal'])
        for field,value in [('pose_m',[[1]]),('mask_state','unavailable'),('render_state','rendered'),
                            ('timestamp_s',2.),('metrics',{'tracking_ms':0}),('evidence',{}),
                            ('backend_identity',{'export_sha256':'older'})]:
            b=copy.deepcopy(a);b[0][field]=value
            with self.subTest(field=field):self.assertFalse(cr.compare_rows(a,b)['equal'])

    def test_missing_extra_and_duplicate_frames(self):
        a=self.sample()
        self.assertFalse(cr.compare_rows(a,[])['equal'])
        self.assertFalse(cr.compare_rows([],a)['equal'])
        with self.assertRaises(cr.RegressionError):cr.compare_rows(a,a+a)

    def test_real_original_full_accounting(self):
        from bench import workbench_cli as cli
        ws=cli.Workspace(cr.ROOT/cr.WORKSPACE)
        for key,spec in cr.CASES.items():
            _,rows,_=cli.load_results(ws,spec['run_id'])
            data=[r.to_dict() for r in rows]
            with self.subTest(case=key):
                self.assertEqual(len(data),2*spec['frames'])
                self.assertTrue(cr.compare_rows(data,data)['equal'])

    def test_guards_raise_baseexception_before_any_model_or_gpu(self):
        from bench import workbench_neural as wn,workbench_telemetry as wt,workbench_cli as cli,workbench_cache as cache
        targets=[(wn,'GoTrackMetricBackend'),(wn,'NeuralSession'),(wn,'segment_prompted'),
                 (wt,'gpu_stage'),(cli.Pipeline,'_cache_store'),(cache.StageCache,'store')]
        originals=[getattr(o,k) for o,k in targets]
        for owner,name in targets:
            with self.subTest(name=name):
                with cr.cache_only_guards() as attempts:
                    with self.assertRaises(cr.CacheOnlyViolation):getattr(owner,name)()
                    self.assertEqual(len(attempts),1)
        self.assertEqual([getattr(o,k) for o,k in targets],originals)
        self.assertFalse(issubclass(cr.CacheOnlyViolation,Exception))

    def test_guards_restore_after_unhandled_miss(self):
        from bench import workbench_telemetry as wt
        original=wt.gpu_stage
        with self.assertRaises(cr.CacheOnlyViolation):
            with cr.cache_only_guards():wt.gpu_stage('primary_neural')
        self.assertIs(wt.gpu_stage,original)

    def test_tampered_plan_rejected_before_json(self):
        with mock.patch.object(Path,'is_file',return_value=True),mock.patch.object(Path,'is_symlink',return_value=False),\
             mock.patch.object(Path,'stat',return_value=mock.Mock(st_size=8)),\
             mock.patch.object(Path,'read_bytes',return_value=b'bad json'),\
             mock.patch.object(cr,'loads_strict',side_effect=AssertionError('parsed')):
            with self.assertRaisesRegex(cr.RegressionError,'SHA-256'):cr.load_plan(Path('x'),'0'*64)

    def test_strict_json_nonfinite_duplicate(self):
        for raw in [b'{"x":NaN}',b'{"x":1,"x":2}']:
            with self.assertRaises(cr.RegressionError):cr.loads_strict(raw,'manufactured')

    def test_plan_refuses_torch_residency_before_weights(self):
        with mock.patch.dict(sys.modules,{'torch':mock.Mock()}),\
             mock.patch.object(cr,'shared_identity',side_effect=AssertionError('weights')):
            with self.assertRaisesRegex(cr.RegressionError,'before torch'):cr.build_plan()

    def test_ineligible_case_never_constructs_pipeline(self):
        with mock.patch('bench.workbench_cli.execute_run',side_effect=AssertionError('pipeline')):
            r=cr.run_case({'cases':{'mug':{'eligible':False,'run_id':'old','ineligible_reason':'changed source'}}},'mug')
        self.assertEqual(r['status'],'ineligible')

    def test_cli_failed_case_returns_nonzero(self):
        for status in ['failed','blocked','ineligible']:
            result={'summary':'x','summary_sha256':'0'*64,'counts':dict.fromkeys(['completed','failed','blocked','ineligible'],0)}
            result['counts'][status]=1
            with self.subTest(status=status),mock.patch.object(cr,'run',return_value=result):
                self.assertNotEqual(cr.main(['run','--plan','x','--expected-plan-sha256','0'*64,'--out','y']),0)

    def test_actual_cache_miss_route_is_failed_without_inference(self):
        from bench import workbench_cli as cli,workbench_neural as wn,workbench_telemetry as wt
        shared={'head':{'path':'.cache/video-workbench/models/flow-terminal-20261021/head.pt'}}
        case={'key':'mug','run_id':cr.CASES['mug']['run_id'],'eligible':True,'mask_set':'sam2-workbench-01',
              'prospective_output_bytes':100,'admission':{'contract_sha256':'fixed'}}
        admission=mock.Mock(admitted=True,body={'contract_sha256':'fixed'})
        def missing_cache(*args,**kwargs):
            wt.gpu_stage('primary_neural')
            raise AssertionError('guard let model path continue')
        with mock.patch.object(cr,'shared_identity',return_value=shared),\
             mock.patch.object(cr,'case_identity',return_value=case),\
             mock.patch.object(cli.Workspace,'require_budget'),\
             mock.patch.object(wn,'load_admission',return_value=admission),\
             mock.patch.object(cli,'execute_run',side_effect=missing_cache):
            result=cr.run_case({'cases':{'mug':case},'shared':shared},'mug')
        self.assertEqual(result['status'],'failed')
        self.assertIn('cache miss',result['reason'])
        self.assertEqual(len(result['guard_attempts']),1)

    def test_complete_shared_closure_includes_runtime_tool_and_key_modules(self):
        original={'runtime':{'Pillow':'1'},'tool_sha256':'fixed','current_key_modules':{'render':'fixed'}}
        case={'run_id':'old'};plan={'shared':original,'cases':{'mug':case}}
        for key,value in [('runtime',{'Pillow':'2'}),('tool_sha256','changed'),('current_key_modules',{'render':'changed'})]:
            changed={**original,key:value}
            with self.subTest(key=key),mock.patch.object(cr,'shared_identity',return_value=changed),\
                 mock.patch.object(cr,'case_identity',return_value=case):
                result=cr.closure_status(plan,'mug')
            self.assertFalse(result['ok']);self.assertTrue(result['case_equal']);self.assertIn(key,result['shared_differs'])

    def test_runtime_comparison_rejects_pillow_and_missing_or_extra_versions(self):
        original={'python':'3.10','Pillow':'10'}
        self.assertEqual(cr.compare_versions(original,dict(original)),{})
        for current in [{'python':'3.10','Pillow':'11'},{'python':'3.10'},{**original,'new':'1'},None]:
            with self.subTest(current=current):self.assertTrue(cr.compare_versions(original,current))

    def test_summary_invalid_destination_prevents_any_case_execution(self):
        plan={'shared':{'tool_sha256':'fixed'},'cases':{'mug':{'eligible':True,'prospective_output_bytes':10}}}
        with mock.patch.object(cr,'load_plan',return_value=plan),mock.patch.object(cr,'sha256_file',return_value='fixed'),\
             mock.patch.object(cr,'validate_destination',side_effect=cr.RegressionError('outside repository')),\
             mock.patch.object(cr,'run_case') as execute:
            with self.assertRaises(cr.RegressionError):cr.run(Path('plan'),'0'*64,Path('out'),'mug')
        execute.assert_not_called()

    def test_post_execution_plan_mutation_revokes_success(self):
        plan={'shared':{'tool_sha256':'fixed'},'cases':{'mug':{'eligible':True,'prospective_output_bytes':10}},'claims':{}}
        captured=[]
        def write(_target,data):captured.append(cr.loads_strict(data,'summary'));return 'receipt'
        with mock.patch.object(cr,'load_plan',return_value=plan),mock.patch.object(cr,'sha256_file',return_value='fixed'),\
             mock.patch.object(cr,'validate_destination',return_value=cr.ROOT/'.cache/mock-summary.json'),\
             mock.patch.object(cr,'plan_file_unchanged',side_effect=[True,False]),\
             mock.patch.object(cr,'run_case',return_value={'status':'completed'}),mock.patch.object(cr,'_write_new',side_effect=write):
            result=cr.run(Path('plan'),'0'*64,Path('out'),'mug')
        self.assertEqual(result['counts']['completed'],0);self.assertEqual(result['counts']['failed'],1)
        self.assertFalse(captured[0]['all_requested_completed']);self.assertEqual(cr.exit_code(result),1)

    def test_real_invalid_summary_paths_rejected_before_writability_probe(self):
        for destination in [cr.ROOT.parent/'outside.json',cr.ROOT/'docs/video-workbench.md',cr.ROOT/'absent-parent/out.json']:
            with self.subTest(path=destination),mock.patch.object(cr.os,'open',side_effect=AssertionError('probe')):
                with self.assertRaises(cr.RegressionError):cr.validate_destination(destination)


if __name__=='__main__':unittest.main()
