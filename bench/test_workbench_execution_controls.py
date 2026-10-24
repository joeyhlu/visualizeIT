"""Codex tests of supplemental service admission; no model or renderer execution."""
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from bench import workbench_service as service
from bench import test_workbench_clip_controls as fixtures

class ExecutionServiceTests(unittest.TestCase):
    setUp=fixtures.ClipControlsTests.setUp
    app=fixtures.ClipControlsTests.app
    execute=fixtures.ClipControlsTests.execute
    def configured(self):
        base=self.app()
        directory=self.root/'execution-control';directory.mkdir()
        for name in service.EXECUTION_CONTROL_FILES:(directory/name).write_bytes(name.encode())
        digest=lambda name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        identity={'kind':service.EXECUTION_CONTROL_KIND,'clip_id':'clip-a','object_id':2,'canonical_ycb_id':'fixture',
            'bundle_sha256':digest('metric-bundle.json'),'glb_sha256':digest('model.glb'),
            'sidecar_sha256':digest('execution-control-sidecar.json'),'receipt_sha256':digest('emission-receipt.json'),
            'adapter_sha256':service.EXECUTION_ADAPTER_SHA256,'map_sha256':service.EXECUTION_MAP_SHA256,
            'family':'fixture_640x480','obj_id':2,'seed_frame_id':'f1','metric_provenance':'dataset_metric_cad',
            'principal_point_origin':'integer_centers','flags':{'execution_only':True}}
        gate=patch.object(service.WorkbenchService,'_validate_execution_control',return_value=identity)
        mocked=gate.start();self.addCleanup(gate.stop)
        app=service.WorkbenchService(base.ws.path,run_options={'execution_controls_by_clip':{'clip-a':str(directory)}})
        return app,directory,mocked

    def test_execution_identity_bound_and_revalidated_before_pipeline(self):
        app,directory,gate=self.configured()
        request,job,execute=self.execute(app)
        self.assertEqual(execute.call_count,1)
        self.assertEqual(execute.call_args.args[3].metric_bundle,str(directory/'metric-bundle.json'))
        self.assertGreaterEqual(gate.call_count,4) # startup, request, job admission, pipeline
        self.assertEqual(request['metric_control']['control_kind'],service.EXECUTION_CONTROL_KIND)

    def test_changed_execution_sidecar_prevents_any_pipeline(self):
        app,directory,_=self.configured()
        (directory/'execution-control-sidecar.json').write_bytes(b'changed')
        _,job,execute=self.execute(app)
        execute.assert_not_called();self.assertEqual(job['status'],'blocked')

    def test_execution_seed_mismatch_is_blocked(self):
        app,_,_=self.configured()
        summary,binding=app._resolve_control(app.clip('clip-a'),'f0',full=True)
        self.assertFalse(summary['ready']);self.assertIsNone(binding)

    def test_conflicting_geometry_configuration_is_rejected(self):
        app,directory,_=self.configured()
        with self.assertRaises(service.WorkbenchError):
            service.WorkbenchService(app.ws.path,run_options={'execution_controls_by_clip':{'clip-a':str(directory)},
                'metric_bundles_by_clip':{'clip-a':str(self.mapped_bundle)}})

class ExactClipBindingTests(unittest.TestCase):
    def setUp(self):
        self.app=object.__new__(service.WorkbenchService)
        self.app._control_mismatches=lambda *args:[] # existing family/size gate has separate coverage
        self.K=[576.,0.,319.5,0.,576.,239.5,0.,0.,1.]
        self.frame=SimpleNamespace(source_frame_index=9,image_sha256='a'*64)
        self.clip=SimpleNamespace(clip_id='clip-ycb',intrinsics=SimpleNamespace(K=self.K),
            provenance={'dataset':service.YCB_DATASET,'object_id':2,'canonical_id':'fixture',
                'inference_manifest_sha256':'b'*64,'scene_id':'000059','onboarding_frame_ids':['s000009']},
            frame=lambda fid:self.frame)
        row={'logicalIndex':9,'frameId':9,'rgb':{'sha256':'a'*64},'calibration':{'cam_K':self.K}}
        yc=SimpleNamespace(OBJECTS={2:('fixture',None)},DEFAULT_MANIFEST='inference-only',
            load_manifest=lambda path:({},'b'*64),select_object=lambda manifest,oid:{'sceneId':59,'onboarding':[row]})
        self.adapter=SimpleNamespace(yc=yc,FINAL_ONBOARDING_LOGICAL=9)
        self.bundle=SimpleNamespace(obj_id=2,seed_frame_id='s000009')
        self.result={'object_id':2,'seed_frame_id':'s000009'}

    def issues(self):return self.app._execution_binding_issues(self.adapter,self.clip,self.result,self.bundle)

    def test_exact_clip_binding_accepts_authenticated_opening(self):self.assertEqual(self.issues(),[])

    def test_foreign_object_rejected(self):
        self.clip.provenance['object_id']=4
        self.assertTrue(self.issues())

    def test_changed_manifest_or_scene_rejected(self):
        for key,value in [('inference_manifest_sha256','c'*64),('scene_id','000060')]:
            original=self.clip.provenance[key];self.clip.provenance[key]=value
            self.assertTrue(self.issues());self.clip.provenance[key]=original

    def test_changed_seed_rgb_or_source_frame_rejected(self):
        self.frame.image_sha256='c'*64;self.assertTrue(self.issues())
        self.frame.image_sha256='a'*64;self.frame.source_frame_index=10;self.assertTrue(self.issues())

    def test_wrong_calibration_or_nonopening_seed_rejected(self):
        self.clip.intrinsics.K=[1.]*9;self.assertTrue(self.issues())
        self.clip.intrinsics.K=self.K;self.clip.provenance['onboarding_frame_ids']=[];self.assertTrue(self.issues())

if __name__=='__main__':unittest.main()
