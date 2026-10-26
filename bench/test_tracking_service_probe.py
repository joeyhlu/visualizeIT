"""Temporary-resource tests for bounded production-service HTTP replay."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from . import tracking_service_probe as probe
from .prototype_benchmark import digest
from .prototype_service import ASSET_ARRAYS, load_asset_registry, load_calibration_registry


def fixture(root, *, synthetic=True, evaluation_frames=13):
    root=Path(root); root.mkdir()
    rng=np.random.default_rng(282)
    Image.fromarray(rng.integers(0,256,(96,128,3),dtype=np.uint8)).save(root/'rgb.png')
    Image.fromarray(np.full((96,128),1000,dtype=np.uint16)).save(root/'depth.png')
    Image.fromarray(np.full((96,128),255,dtype=np.uint8)).save(root/'mask.png')
    (root/'object.ply').write_text('''ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
-100 -100 0
100 -100 0
0 100 0
0 0 100
3 0 2 1
3 0 1 3
3 1 2 3
3 2 0 3
''',encoding='ascii')
    def ref(name):return dict(path=name,sha256=digest(root/name))
    k=[100.,0,63.5,0,100.,47.5,0,0,1.]
    obj=dict(objectId=1,name='Temporary synthetic service fixture',model=ref('object.ply'),
             referenceId='fixture-source-reference',onboarding=[],frames=[])
    for index in range(10+evaluation_frames):
        row=dict(frameId=index,logicalIndex=index,timestampKind='physical',
                 timestampSeconds=index/30,rgb=ref('rgb.png'),
                 calibration=dict(cam_K=k,depth_scale=1.))
        if index<10:
            row.update(depth=ref('depth.png'),onboardingMask=ref('mask.png'),
                evaluationPose=dict(cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[0.,0.,1000.]))
            obj['onboarding'].append(row)
        else:obj['frames'].append(row)
    manifest=dict(schemaVersion=1,benchmarkId='temporary-test-only',dataset='Temporary procedural fixture',
        evidenceKind='synthetic_stress' if synthetic else 'dataset_reference',objects=[obj])
    (root/'inference.json').write_text(json.dumps(manifest))
    # Any accidental evaluator read/parse fails immediately; no evaluator resource exists.
    (root/'evaluator.json').write_text('THIS FILE MUST NEVER BE OPENED BY THE PROBE')
    return manifest


class TrackingServiceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.dataset=self.root/'dataset'
        self.manifest=fixture(self.dataset)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self,name='assets',**kwargs):
        return probe.prepare_asset(self.dataset,self.root/name,frame_count=3,**kwargs)

    def test_export_uses_standard_production_asset_contract_and_truthful_calibration(self):
        prepared=self.prepare()
        loaded=load_asset_registry(prepared['assetsRoot'])
        self.assertEqual(set(loaded),{1})
        self.assertEqual(loaded[1].reference_id,prepared['referenceId'])
        self.assertEqual(loaded[1].evidence_kind,'synthetic_control')
        self.assertGreaterEqual(prepared['referenceAnchors'],12)
        with np.load(loaded[1].asset_path,allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files),ASSET_ARRAYS)
            basis=arrays['benchFromSource']
            np.testing.assert_allclose(arrays['modelPositions']@basis[:3,:3].T+basis[:3,3],arrays['positions'])
        calibration=load_calibration_registry(prepared['calibrationsPath'])
        self.assertEqual(calibration[prepared['calibrationId']].kind,'synthetic')
        real=self.root/'real-fixture'
        fixture(real,synthetic=False)
        prepared_real=probe.prepare_asset(real,self.root/'real-assets',frame_count=2)
        real_calibration=load_calibration_registry(prepared_real['calibrationsPath'])
        self.assertEqual(real_calibration[prepared_real['calibrationId']].kind,'dataset_reference')

    def test_wire_contains_only_rgb_identity_and_normalizes_prerecorded_clock(self):
        prepared=self.prepare()
        record=copy.deepcopy(prepared['records'][0])
        record.update(evaluationPose={'secret':'forbidden'},depth={'secret':'forbidden'},mask='forbidden')
        packet,body=probe.make_packet(prepared,record,'test-session')
        self.assertEqual(set(packet),probe.WIRE_FIELDS)
        self.assertEqual(packet['timestampKind'],'frame_index_no_fps')
        self.assertIsNone(packet['captureTimestampSeconds'])
        self.assertNotIn(b'forbidden',body)
        self.assertNotIn(b'evaluationPose',body)

    def test_actual_current_service_http_variable_length_contract_and_close(self):
        # Actual production service, actual current tracker, no fake pose output.
        with mock.patch.object(probe,'_exchange',wraps=probe._exchange) as exchange:
            result=probe.run_probe([self.dataset],self.root/'report.json',frame_count=3)
        report=result['datasets'][0]
        self.assertEqual(report['framesSent'],3)
        self.assertEqual(report['acceptedFrameRequests'],3)
        self.assertTrue(result['allContractsPassed'])
        self.assertTrue(report['allRequestsHTTP200'])
        self.assertTrue(report['serverClosed'])
        self.assertFalse(result['accuracyMeasured'])
        self.assertFalse(result['liveFPSMeasured'])
        self.assertFalse(result['evaluatorOpened'])
        self.assertEqual(sum(report['stateCounts'].values()),3)
        # Fixture anchors are planar; lost/recovering is valid honest evidence.
        self.assertTrue(all(row['contractValid'] for row in report['frames']))
        port=exchange.call_args_list[0].args[0]
        with socket.socket() as client:
            self.assertNotEqual(client.connect_ex(('127.0.0.1',port)),0)

    def test_server_closes_when_exchange_raises(self):
        prepared=self.prepare()
        captured=[]
        original=probe._exchange
        def fail_after_health(port,method,endpoint,body=None):
            captured.append(port)
            if method=='POST':raise OSError('intentional interrupted HTTP test')
            return original(port,method,endpoint,body)
        with mock.patch.object(probe,'_exchange',side_effect=fail_after_health):
            with self.assertRaisesRegex(OSError,'interrupted HTTP'):
                probe.probe_prepared(prepared)
        with socket.socket() as client:
            self.assertNotEqual(client.connect_ex(('127.0.0.1',captured[0])),0)

    def test_response_contract_rejects_stale_wrong_identity_nonfinite_and_fake_live(self):
        packet={key:None for key in probe.IDENTITY_FIELDS}
        packet.update(sessionId='s',objectId=1,referenceId='r',calibrationId='c',frameId=10,
                      logicalIndex=10,timestampKind='frame_index_no_fps')
        payload=dict(**packet,poseState='lost',processingSeconds=.01,cameraFromObject=None,
                     projectionSegments=None,renderState='suppressed',freshCameraFrameVerified=False)
        self.assertTrue(probe.interpret_response(200,payload,packet)['contractValid'])
        mutations=({'frameId':11},{'processingSeconds':float('nan')},
                   {'cameraFromObject':np.eye(4).tolist()},{'freshCameraFrameVerified':True},
                   {'projectionSegments':[[1,2,3,4]]})
        for update in mutations:
            with self.subTest(update=update):
                bad=dict(payload,**update)
                self.assertFalse(probe.interpret_response(200,bad,packet)['contractValid'])
        tracking=dict(payload,poseState='tracking',units='metres',cameraFromObject=np.eye(4).tolist(),
                      cameraConvention='CV X right, Y down, Z forward; column vectors')
        self.assertTrue(probe.interpret_response(200,tracking,packet)['contractValid'])
        tracking['cameraFromObject']=np.eye(4,dtype=bool).tolist()
        self.assertFalse(probe.interpret_response(200,tracking,packet)['contractValid'])

    def test_cap_label_leakage_resource_hash_and_fresh_output_guards(self):
        for count in (0,21,True):
            with self.assertRaises(ValueError):
                probe.prepare_asset(self.dataset,self.root/'unused',frame_count=count)
        with self.assertRaisesRegex(ValueError,'source dataset'):
            probe.run_probe([self.dataset],self.dataset/'new-report.json',frame_count=1)
        self.prepare()
        with self.assertRaisesRegex(ValueError,'fresh production asset'):
            self.prepare()
        self.manifest['objects'][0]['frames'][0]['evaluationPose']={}
        (self.dataset/'inference.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'Evaluator labels'):
            probe.prepare_asset(self.dataset,self.root/'leak-assets',frame_count=1)
        del self.manifest['objects'][0]['frames'][0]['evaluationPose']
        (self.dataset/'inference.json').write_text(json.dumps(self.manifest))
        (self.dataset/'rgb.png').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            probe.prepare_asset(self.dataset,self.root/'bad-assets',frame_count=1)


if __name__=='__main__':
    unittest.main()
