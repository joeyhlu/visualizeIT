"""Small independent HB input controls; no real dataset, labels or models."""
from pathlib import Path
import binascii
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock
import zlib
import numpy as np
from tools import workbench_hb_input_preflight as hb

SCRATCH=Path(__file__).resolve().parents[1]/'.cache/video-workbench/scratch'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


class HBPreflightTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='hb-input-test-',dir=SCRATCH)
        self.root=Path(self.temp.name).resolve();self.addCleanup(self.cleanup)
    def cleanup(self):
        assert self.root.is_relative_to(SCRATCH.resolve());self.temp.cleanup()
    def stream(self,raw=b'manufactured RGB bytes'*100,method=8,tail=b''):
        if method==8:
            encoder=zlib.compressobj(wbits=-15);compressed=encoder.compress(raw)+encoder.flush()
        else:compressed=raw
        compressed+=tail;path=self.root/'payload.bin';path.write_bytes(compressed)
        return path,{'method':method,'expanded_bytes':len(raw),'compressed_bytes':len(compressed),
                     'crc32':binascii.crc32(raw)&0xffffffff},raw,compressed
    def test_stored_and_deflate_are_exact_and_sha_is_observed(self):
        for method in [0,8]:
            with self.subTest(method=method):
                path,entry,raw,compressed=self.stream(method=method);chunks=[]
                got=hb.verified_member_stream(path,entry,chunks.append)
                self.assertEqual(b''.join(chunks),raw)
                self.assertEqual(got['expanded_sha256_observed'],sha(raw));self.assertEqual(got['compressed_sha256_observed'],sha(compressed))
                self.assertTrue(got['crc32_and_sizes_consistent_with_local_index'])
                self.assertIs(got['cryptographic_source_identity_verified'],False)
    def test_wrong_crc_size_truncation_and_trailing_rejected(self):
        for mode in ['crc','expanded_size','compressed_size','truncated','trailing']:
            path,entry,raw,compressed=self.stream(tail=b'extra' if mode=='trailing' else b'')
            if mode=='crc':entry['crc32']^=1
            if mode=='expanded_size':entry['expanded_bytes']-=1
            if mode=='compressed_size':entry['compressed_bytes']+=1
            if mode=='truncated':path.write_bytes(compressed[:-1]);entry['compressed_bytes']-=1
            with self.subTest(mode=mode),self.assertRaises(hb.PreflightError):hb.verified_member_stream(path,entry)
    def test_overexpansion_stops_before_sink_receives_excess(self):
        path,entry,_,_=self.stream(raw=b'a'*200000);entry['expanded_bytes']=8;received=[]
        with self.assertRaises(hb.PreflightError):hb.verified_member_stream(path,entry,received.append)
        self.assertLessEqual(sum(map(len,received)),8)
    def test_frame_guard_precedes_any_file_open(self):
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace')
        for frames in [[10],[-1],[0,10],[True],[0,0],[],list(range(11))]:
            with self.subTest(frames=frames),mock.patch.object(Path,'open',side_effect=AssertionError('must not read')):
                with self.assertRaises(hb.UsageError):hb.preview_opening(ctx,'nonexistent-plan','0'*64,frames,'new')
        self.assertEqual(hb.validate_frames([9,0]),[0,9])
    def header(self,faces=1,encoding='ascii',appearance=True,units='mm'):
        lines=['ply',f'format {encoding} 1.0',f'comment units {units}','element vertex 3',
               'property float x','property float y','property float z']
        if appearance:lines+=['property uchar red','property uchar green','property uchar blue']
        lines += [f'element face {faces}','property list uchar int vertex_indices','end_header']
        return ('\n'.join(lines)+'\n').encode()
    def test_header_feasibility_never_claims_geometry_or_admission(self):
        for encoding in ['ascii','binary_little_endian']:
            got=hb.evaluate_header(hb.parse_ply_header(self.header(encoding=encoding)))
            self.assertTrue(got['header_feasible_for_future_reviewed_conversion'])
            self.assertFalse(got['geometry_verified']);self.assertFalse(got['admitted'])
            self.assertEqual(got['face_count'],1)
    def test_face_cap_encoding_and_missing_appearance_block_feasibility(self):
        for kwargs in [{'faces':200001},{'encoding':'binary_big_endian'},{'appearance':False},{'faces':0}]:
            with self.subTest(kwargs=kwargs):
                got=hb.evaluate_header(hb.parse_ply_header(self.header(**kwargs)))
                self.assertFalse(got['header_feasible_for_future_reviewed_conversion']);self.assertFalse(got['admitted'])
    def test_malformed_unbounded_or_ambiguous_ply_header_rejected(self):
        for raw in [b'not ply',self.header().replace(b'end_header\n',b''),
                    self.header().replace(b'element face 1',b'element face -1'),
                    self.header().replace(b'element face 1',b'element face nan'),
                    self.header().replace(b'property float x',b'property float x\nproperty float x')]:
            with self.subTest(),self.assertRaises(hb.PreflightError):hb.parse_ply_header(raw)
    def test_exact_rgb_channel_convention_and_shape(self):
        bgr=np.zeros((480,640,3),np.uint8);bgr[:]=[7,31,201]
        got=hb.decode_rgb(b'manufactured',decode=lambda raw:bgr)
        self.assertEqual(got[0,0].tolist(),[201,31,7]);self.assertTrue(got.flags.c_contiguous)
        for bad in [None,bgr[:10],bgr.astype(float),np.zeros((480,640,4),np.uint8)]:
            with self.subTest(),self.assertRaises(hb.PreflightError):hb.decode_rgb(b'fake',decode=lambda raw:bad)
    def test_png_native_size_channels_and_crc(self):
        from bench.vision import cv2
        okay,encoded=cv2.imencode('.png',np.zeros((480,640,3),np.uint8));self.assertTrue(okay)
        raw=encoded.tobytes();self.assertEqual(hb.check_png_structure(raw)['width'],640)
        for bad in [raw[:-1],raw[:29]+b'\0\0\0\0'+raw[33:]]:
            with self.assertRaises(hb.PreflightError):hb.check_png_structure(bad)
        _,rgba=cv2.imencode('.png',np.zeros((480,640,4),np.uint8))
        with self.assertRaises(hb.PreflightError):hb.check_png_structure(rgba.tobytes())
    def test_calibration_is_dataset_only_and_conflicts_rejected(self):
        row={'kind':'dataset_reference','width':640,'height':480,
             'K':[537.4799,0,318.8965,0,536.1447,238.3781,0,0,1], 'objectId':'never copied'}
        result=hb.select_calibration({'schemaVersion':1,'calibrations':[row]})
        self.assertFalse(result['phone_calibration']);self.assertNotIn('objectId',result)
        changed=dict(row,K=[1,0,1,0,1,1,0,0,1])
        with self.assertRaises(hb.PreflightError):hb.select_calibration({'schemaVersion':1,'calibrations':[row,changed]})
    def test_payload_path_escape_and_unknown_method_rejected(self):
        with self.assertRaises(hb.PreflightError):hb.safe_member_path(self.root,'../escape.bin','fixture')
        member='val_primesense/000002/rgb/000000.png';archive=hb.RGB_ARCHIVE
        row={'sourceMethod':99,'sourceFlags':0,'sourceCompressedBytes':1,'expandedBytes':1,'crc32':0,
             'method':99,'compressedBytes':1,'datasetRevision':hb.DATASET_REVISION,'archive':archive,
             'sourceMember':member,'allowedRoles':['rgb'],
             'location':{'kind':'payload','path':'payloads/'+sha((archive+':'+member).encode())+'.bin'}}
        with self.assertRaises(hb.PreflightError):hb.validate_member_row(row,member,archive,'rgb',hb.MAX_RGB_EXPANDED)

    def metadata_fixture(self,model_ids=range(1,11)):
        folder=self.root/'metadata';folder.mkdir();payloads=folder/'payloads';payloads.mkdir()
        resources={}
        for member,archive,role in [(f'val_primesense/000002/rgb/{i:06d}.png',hb.RGB_ARCHIVE,'rgb') for i in range(120)]+[(f'models/obj_{i:06d}.ply',hb.MODEL_ARCHIVE,'model') for i in model_ids]:
            token=sha((archive+':'+member).encode());relative='payloads/'+token+'.bin'
            (folder/relative).write_bytes(b'x')
            resources[member]={'sourceMethod':0,'sourceFlags':0,'sourceCompressedBytes':1,'expandedBytes':1,
                'crc32':binascii.crc32(b'x')&0xffffffff,'method':0,'compressedBytes':1,
                'datasetRevision':hb.DATASET_REVISION,'archive':archive,'sourceMember':member,
                'allowedRoles':[role],'location':{'kind':'payload','path':relative}}
        acq=folder/'acquisition-plan.json';cal=folder/'calibrations.json'
        acq.write_text(json.dumps({'schemaVersion':1,'corpus':hb.CORPUS,'datasetRevision':hb.DATASET_REVISION,
                                  'status':'planned_only','resources':resources}),encoding='utf-8')
        cal.write_text(json.dumps({'schemaVersion':1,'calibrations':[{'kind':'dataset_reference','width':640,'height':480,
            'K':[537.4799,0,318.8965,0,536.1447,238.3781,0,0,1]}]}),encoding='utf-8')
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace');ctx.derived.mkdir(parents=True)
        return ctx,acq,cal,payloads

    def test_metadata_plan_does_not_open_any_payload(self):
        ctx,acq,cal,payloads=self.metadata_fixture();original=Path.open;opened=[]
        def checked(path,*args,**kwargs):
            if path.suffix=='.bin':opened.append(str(path));raise AssertionError('metadata-only plan read payload')
            return original(path,*args,**kwargs)
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic),mock.patch.object(Path,'open',checked):
            plan,complete=hb.build_plan(ctx,acq,cal)
        self.assertEqual(opened,[]);self.assertTrue(complete)
        self.assertFalse(plan['status']['compressed_content_verified']);self.assertFalse(plan['status']['neural_admitted'])
        self.assertEqual(plan['selection']['opening'],list(range(10)))
        self.assertEqual(plan['selection']['scoring'],list(range(10,120)))
        self.assertIsNone(plan['selection']['physical_timestamp'])

    def test_wrong_ten_model_ids_cannot_claim_complete_inventory(self):
        ctx,acq,cal,_=self.metadata_fixture(model_ids=range(2,12))
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic):plan,complete=hb.build_plan(ctx,acq,cal)
        self.assertFalse(complete);self.assertFalse(plan['models']['inventory_complete'])

    def test_candidate_new_input_limit_is_enforced(self):
        ctx,acq,cal,_=self.metadata_fixture()
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic),mock.patch.object(hb,'NEW_INPUT_LIMIT',129):
            with self.assertRaises(hb.PreflightError):hb.build_plan(ctx,acq,cal)

    def test_guard_blocks_read_write_modes_and_restores_on_error(self):
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace');bad=self.root/'scene_gt.json'
        bad.write_bytes(b'{"manufactured":true}')
        for opener in [lambda:open(bad,'r+b'),lambda:os.open(bad,os.O_RDWR)]:
            with self.subTest(),hb.guarded_stage(ctx):
                with self.assertRaises(hb.GuardError):
                    handle=opener()
                    # Only used to close an unexpectedly allowed open before failing.
                    if isinstance(handle,int):os.close(handle)
                    else:handle.close()
        self.assertFalse(hb._GUARD['active'])

    def test_both_payload_stages_guard_metadata_rebuild(self):
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace');observed=[]
        ctx.derived.mkdir(parents=True)
        def stop(*args,**kwargs):
            observed.append(hb._GUARD['active']);raise hb.PreflightError('manufactured stop before data')
        for call in [lambda:hb.inspect_mesh(ctx,'plan.json','0'*64,1,str(ctx.derived/'mesh.json')),
                     lambda:hb.preview_opening(ctx,'plan.json','0'*64,[0],str(ctx.derived/'preview'))]:
            with mock.patch.object(hb,'load_plan',side_effect=stop):
                try:call()
                except hb.PreflightError:pass
        self.assertEqual(observed,[True,True]);self.assertFalse(hb._GUARD['active'])

    def test_both_payload_stages_refuse_low_budget_before_payload_open(self):
        ctx,acq,cal,_=self.metadata_fixture()
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic):
            plan,_=hb.build_plan(ctx,acq,cal);path=self.root/'plan.json';raw=hb.dump_json(plan);path.write_bytes(raw)
            calls=[lambda:hb.inspect_mesh(ctx,str(path),sha(raw),1,str(ctx.derived/'mesh.json')),
                   lambda:hb.preview_opening(ctx,str(path),sha(raw),[0],str(ctx.derived/'preview'))]
            for call in calls:
                with mock.patch.object(hb,'workspace_budget',side_effect=hb.PreflightError('manufactured low disk')) as budget, \
                     mock.patch.object(hb,'verified_member_stream') as stream:
                    try:call()
                    except hb.PreflightError:pass
                    budget.assert_called()
                    stream.assert_not_called()

    def test_nested_guard_cannot_disable_the_outer_guard(self):
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace')
        with hb.guarded_stage(ctx):
            with self.assertRaises(hb.GuardError):
                with hb.guarded_stage(ctx):pass
            self.assertTrue(hb._GUARD['active'])
        self.assertFalse(hb._GUARD['active'])

    def test_direct_forbidden_plan_rejected_before_read(self):
        ctx=hb.Context(root=self.root,workspace=self.root/'workspace');ctx.derived.mkdir(parents=True)
        forbidden=self.root/'scene_gt.json';forbidden.write_bytes(b'{"manufactured":true}')
        original=hb.read_bounded;seen=[]
        def read(path,*args):seen.append(Path(path));return original(path,*args)
        with mock.patch.object(hb,'read_bounded',side_effect=read):
            with self.assertRaises(hb.PreflightError):
                hb.inspect_mesh(ctx,str(forbidden),sha(forbidden.read_bytes()),1,str(ctx.derived/'header.json'))
        self.assertNotIn(forbidden,seen)

    def test_nested_forbidden_acquisition_reference_rejected_before_read(self):
        ctx,acq,cal,_=self.metadata_fixture();bad=acq.parent/'scene_gt.json';bad.write_bytes(b'{}')
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic):
            plan,_=hb.build_plan(ctx,acq,cal);plan['inputs']['acquisition_plan']['path']=bad.relative_to(self.root).as_posix()
            path=self.root/'plan.json';raw=hb.dump_json(plan);path.write_bytes(raw);seen=[];original=hb.read_bounded
            def read(candidate,*args):seen.append(Path(candidate));return original(candidate,*args)
            with mock.patch.object(hb,'read_bounded',side_effect=read):
                with self.assertRaises(hb.PreflightError):
                    hb.inspect_mesh(ctx,str(path),sha(raw),1,str(ctx.derived/'header.json'))
            self.assertNotIn(bad,seen)

    def test_combined_input_budget_requires_explicit_strict_existing_bytes(self):
        with self.assertRaises(hb.PreflightError):hb.input_budget(1,hb.NEW_INPUT_LIMIT)
        for bad in [True,-1,1.5,'10']:
            with self.subTest(bad=bad),self.assertRaises(hb.PreflightError):hb.input_budget(1,bad)
        unknown=hb.input_budget(1,None)
        self.assertIn('unverified_parent_required',str(unknown))
        checked=hb.input_budget(1,100);self.assertNotIn('unverified_parent_required',str(checked))

    def replace_member(self,acq,member,raw):
        doc=json.loads(acq.read_bytes());row=doc['resources'][member]
        (acq.parent/row['location']['path']).write_bytes(raw)
        row.update(sourceCompressedBytes=len(raw),compressedBytes=len(raw),expandedBytes=len(raw),
                   crc32=binascii.crc32(raw)&0xffffffff)
        acq.write_text(json.dumps(doc),encoding='utf-8')

    def test_successful_manufactured_mesh_receipt_has_no_peak_or_admission_claim(self):
        ctx,acq,cal,_=self.metadata_fixture()
        self.replace_member(acq,'models/obj_000001.ply',self.header())
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':120,'rgb_expanded_bytes':120}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic):
            plan,_=hb.build_plan(ctx,acq,cal);path=self.root/'plan.json';raw=hb.dump_json(plan);path.write_bytes(raw)
            out=ctx.derived/'header.json'
            with mock.patch.object(hb,'workspace_budget',return_value={'synthetic_budget':True}):
                result=hb.inspect_mesh(ctx,str(path),sha(raw),1,str(out))
        doc=json.loads(out.read_bytes());self.assertEqual(result['exit'],0)
        self.assertFalse(doc['feasibility']['admitted']);self.assertFalse(doc['feasibility']['geometry_verified'])
        self.assertNotIn('peak_payload_buffer_bound_bytes',doc['memory'])
        self.assertIs(doc['memory']['absolute_peak_measured'],False)
        self.assertIs(doc['payload_integrity']['cryptographic_source_identity_verified'],False)
        self.assertEqual(len(doc['payload_opens']),1)

    def test_successful_manufactured_preview_preserves_exact_opening_identity(self):
        from bench.vision import cv2
        from bench.workbench_geometry import array_sha256
        ctx,acq,cal,_=self.metadata_fixture();expected={}
        for index,colour in [(0,[7,31,201]),(9,[27,61,151])]:
            bgr=np.full((480,640,3),colour,np.uint8);ok,encoded=cv2.imencode('.png',bgr);self.assertTrue(ok)
            self.replace_member(acq,f'val_primesense/000002/rgb/{index:06d}.png',encoded.tobytes())
            expected[index]=array_sha256(np.ascontiguousarray(bgr[:,:,::-1]))
        source=json.loads(acq.read_bytes())['resources'];total=sum(row['compressedBytes'] for key,row in source.items() if '/rgb/' in key)
        synthetic={'000002':{'rgb_frames':120,'rgb_encoded_bytes':total,'rgb_expanded_bytes':total}}
        with mock.patch.object(hb,'ALLOWED_SCENES',synthetic):
            plan,_=hb.build_plan(ctx,acq,cal);path=self.root/'plan.json';raw=hb.dump_json(plan);path.write_bytes(raw)
            out=ctx.derived/'preview'
            with mock.patch.object(hb,'workspace_budget',return_value={'synthetic_budget':True}):
                result=hb.preview_opening(ctx,str(path),sha(raw),[9,0],str(out))
        doc=json.loads((out/'preview-receipt.json').read_bytes());self.assertEqual(result['exit'],0)
        self.assertEqual([row['index'] for row in doc['frames']],[0,9])
        for row in doc['frames']:
            self.assertEqual(row['decoded']['rgb_array_sha256'],expected[row['index']])
        self.assertEqual(len(doc['payload_opens']),2)
        self.assertEqual(set(p.name for p in out.iterdir()),{'opening-contact-sheet.png','preview-receipt.json'})
        self.assertFalse(doc['flags']['neural_admitted']);self.assertFalse(doc['flags']['cryptographic_source_identity_verified'])


if __name__=='__main__':unittest.main()
