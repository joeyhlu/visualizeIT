"""Codex integration tests for explicit per-clip controls; execution is stubbed."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bench import workbench_service as service,workbench_neural as neural,workbench_contract as wc
from bench.test_workbench_contract import clip_document

class ClipControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.global_bundle=self.root/'global.json';self.global_bundle.write_text('{}')
        self.mapped_bundle=self.root/'mapped.json';self.mapped_bundle.write_text('{"mapped":true}')
        self.bundle=SimpleNamespace(family='fixture_640x480',seed_pose_m=np.eye(4),seed_frame_id='f1',
                    glb_sha256='b'*64,obj_id=2,principal_point_origin='integer_centers',metric_provenance='dataset_metric_cad')
        loader=patch.object(neural.MetricBundle,'from_file',return_value=self.bundle)
        loader.start();self.addCleanup(loader.stop)

    def app(self,**options):
        app=service.WorkbenchService(self.root/'workspace',run_options=options)
        for name in ('clip-a','clip-b'):
            doc=clip_document();doc['clip_id']=name;doc['onboarding']['selection']['frame_id']='f1'
            app.ws.save_clip(wc.ClipSpec.from_dict(doc))
        return app

    def execute(self,app,name='clip-a',roles=None,**extra):
        request=app.create_run({'clip_id':name,'mode':'smoke','roles':roles or ['primary_neural','frozen_baseline'],**extra})
        def synchronous(*,target,args,**kwargs):return SimpleNamespace(start=lambda:target(*args))
        with patch.object(service.threading,'Thread',side_effect=synchronous),\
             patch.object(app.cli,'execute_run',return_value={'status':'blocked'}) as execute:
            job=app.create_job(request['run_id'])
        return request,job,execute

    def test_duplicates_and_malformed_mapping_rejected(self):
        for mapping in (['clip-a=x','clip-a=y'],['../escape=x'],['missing-separator']):
            with self.subTest(mapping=mapping),self.assertRaises(service.WorkbenchError):
                service.parse_clip_mapping(mapping,'bundle')

    def test_lookalike_clip_cannot_inherit_bound_global(self):
        app=self.app(metric_bundle=str(self.global_bundle),metric_bundle_clip='clip-a')
        first,binding=app._resolve_control(app.clip('clip-a'),'f1')
        second,other=app._resolve_control(app.clip('clip-b'),'f1')
        self.assertTrue(first['ready']);self.assertIsNotNone(binding)
        self.assertFalse(second['ready']);self.assertIsNone(other)

    def test_unbound_legacy_bundle_is_unavailable(self):
        app=self.app(metric_bundle=str(self.global_bundle))
        summary,binding=app._resolve_control(app.clip('clip-a'),'f1')
        self.assertFalse(summary['ready']);self.assertIsNone(binding)

    def test_explicit_mapping_wins_and_api_cannot_choose_path(self):
        app=self.app(metric_bundle=str(self.global_bundle),metric_bundle_clip='clip-a',
                     metric_bundles_by_clip={'clip-a':str(self.mapped_bundle)})
        request,job,execute=self.execute(app,metric_bundle=str(self.global_bundle))
        self.assertEqual(execute.call_count,1)
        self.assertEqual(Path(execute.call_args.args[3].metric_bundle),self.mapped_bundle)
        self.assertEqual(request['metric_control']['source'],'per_clip')
        self.assertNotIn(str(self.root),json.dumps(request['metric_control']))

    def test_changed_mapped_bundle_blocks_without_global_fallback(self):
        app=self.app(metric_bundle=str(self.global_bundle),metric_bundle_clip='clip-a',
                     metric_bundles_by_clip={'clip-a':str(self.mapped_bundle)})
        self.mapped_bundle.write_text('{"changed":true}')
        request,job,execute=self.execute(app)
        execute.assert_not_called();self.assertEqual(job['status'],'blocked')
        self.assertIn('changed',job['blocked_reason'])
        self.assertEqual(job['frame_result_count'],4)

    def test_relative_only_run_does_not_receive_metric_geometry(self):
        app=self.app(metric_bundles_by_clip={'clip-a':str(self.mapped_bundle)})
        _,_,execute=self.execute(app,roles=['relative_geometry'])
        self.assertIsNone(execute.call_args.args[3].metric_bundle)

    def test_selection_frame_mismatch_blocks_mapping(self):
        app=self.app(metric_bundles_by_clip={'clip-a':str(self.mapped_bundle)})
        summary,binding=app._resolve_control(app.clip('clip-a'),'f0')
        self.assertEqual(summary['status'],'blocked');self.assertIsNone(binding)

    def test_mask_defaults_stay_per_clip(self):
        app=self.app(mask_set='observed-global',masks_by_clip={'clip-b':'observed-mapped'})
        directory=app.ws.masks_dir/'clip-a'/'observed-global';directory.mkdir(parents=True)
        (directory/'manifest.json').write_text('{}')
        self.assertEqual(app._default_masks('clip-a')['default'],'observed-global')
        self.assertEqual(app._default_masks('clip-b')['default'],'observed-mapped')
        self.assertFalse(app._default_masks('clip-b')['present'])

if __name__=='__main__':unittest.main()
