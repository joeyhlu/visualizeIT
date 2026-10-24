"""Independent manufactured portability fixtures; no models or video inputs."""
from pathlib import Path
import copy
import builtins
import hashlib
import json
import os
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from bench import workbench_contract as wc, workbench_geometry as wg

SCRATCH=Path(__file__).resolve().parents[1]/'.cache/video-workbench/scratch'
sha=lambda data:hashlib.sha256(data).hexdigest()

def encode_glb(document, binary):
    js=json.dumps(document,separators=(',',':')).encode()
    js+=b' '*((-len(js))%4); binary+=b'\0'*((-len(binary))%4)
    chunks=struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binary),0x004E4942)+binary
    return struct.pack('<III',0x46546C67,2,12+len(chunks))+chunks


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        from tools import workbench_portability_audit as audit
        self.audit=audit
        SCRATCH.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='portability-test-',dir=SCRATCH)
        self.root=Path(self.temp.name).resolve();self.addCleanup(self.cleanup)

    def cleanup(self):
        assert self.root.is_relative_to(SCRATCH.resolve());self.temp.cleanup()

    def prepared(self, triangles=1, document_mutator=None, artwork_mutator=None, embedded_png=None):
        positions=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4').tobytes()
        indices=np.tile(np.array([0,1,2],dtype='<u2'),triangles).tobytes()
        binary=positions+indices
        doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(binary)}],
             'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(positions)},
                            {'buffer':0,'byteOffset':len(positions),'byteLength':len(indices)}],
             'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3','min':[0,0,0],'max':[1,1,0]},
                          {'bufferView':1,'componentType':5123,'count':3*triangles,'type':'SCALAR'}],
             'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1,'mode':4}]}],
             'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0}
        if embedded_png is not None:
            binary+=b'\0'*((-len(binary))%4);offset=len(binary);binary+=embedded_png
            doc['buffers'][0]['byteLength']=len(binary)
            doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(embedded_png)})
            doc['images']=[{'bufferView':2,'mimeType':'image/png'}]
        if document_mutator: document_mutator(doc)
        glb=self.root/'model.glb';glb.write_bytes(encode_glb(doc,binary))
        bundle={'schema':'workbench_metric_bundle_v1','family':'ycb_640x480','glb_path':'model.glb',
                'glb_sha256':sha(glb.read_bytes()),'obj_id':2,'principal_point_origin':'integer_centers',
                'seed_pose_m':np.eye(4).tolist(),'seed_label':'controlled_init:manufactured fixture',
                'seed_frame_id':'seed','metric_provenance':'dataset_metric_cad'}
        art=wg.builtin_artwork('checkerboard',256); digest=wg.array_sha256(art)
        marker={'face_id':0,'barycentric':[1/3]*3,'tangent':[1,0,0],'bitangent':[0,1,0],
                'normal':[0,0,1],'size_object_units':0.1,'artwork_sha256':digest}
        record={'schema':'workbench_neural_artwork_v1','units':'metres','glb_sha256':bundle['glb_sha256'],
                'marker':marker,'anchor_pixel':[10,10],'anchor_frame_id':'seed','seed_label':bundle['seed_label'],
                'artwork':{'kind':'checkerboard','edge_px':256,'sha256':digest,'generation':'fixed_builtin_v1'},
                'support_source':'metric CAD geometry (not observation-derived)',
                'principal_point_origin':'integer_centers','effective_K':np.eye(3).ravel().tolist(),
                'patch_faces':1,'mesh_faces':triangles,'anchor_mask_checked':True}
        if artwork_mutator: artwork_mutator(record)
        record['record_sha256']=wc.canonical_sha256(record)
        b=self.root/'bundle.json';a=self.root/'artwork.json'
        b.write_bytes(wc.canonical_json(bundle));a.write_bytes(wc.canonical_json(record))
        return b,a,glb

    def invoke(self,b,a):
        return self.audit.audit_prepared_control(b,a,root=self.root)

    def surface(self, *, descriptor_rows=2, unavailable=False, depth=None):
        folder=self.root/'surface';folder.mkdir()
        arrays={'vertices':np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32),
                'faces':np.array([[0,1,2]],np.int32),'face_support_views':np.array([3],np.int32),
                'descriptor_points':np.zeros((2,3),np.float32),
                'descriptor_values':np.zeros((descriptor_rows,32),np.uint8)}
        names={'vertices':'vertices.npy','faces':'faces.npy','face_support_views':'face_support.npy',
               'descriptor_points':'bank_points.npy','descriptor_values':'bank_descriptors.npy'}
        refs={}
        if not unavailable:
            for key,array in arrays.items():
                path=folder/names[key]
                with path.open('xb') as stream: np.save(stream,array,allow_pickle=False)
                raw=path.read_bytes();refs[key]={'path':path.name,'sha256':sha(raw),'bytes':len(raw)}
        artwork={'kind':'checkerboard','edge_px':256,'sha256':wg.array_sha256(wg.builtin_artwork('checkerboard',256)),
                 'generation':'fixed_builtin_v1'}
        cameras=[]
        if depth is not None:
            raw=b'unsupported depth encoding'; ref={'path':'depth.bin','sha256':sha(raw),'bytes':len(raw)}
            if depth!='missing':(folder/'depth.bin').write_bytes(raw)
            cameras=[{'frame_id':'seed','image_sha256':'a'*64,'mask_sha256':'b'*64,'width':3,'height':3,
                      'K':[100,0,1,0,100,1,0,0,1],'pose':np.eye(4).tolist(),'depth':ref}]
        body={'schema':wc.SURFACE_SCHEMA,'package_id':'manufactured','clip_id':'fixture',
              'status':'unavailable' if unavailable else 'available',
              'unavailable_reason':'manufactured failed preparation' if unavailable else None,
              'units':'object_unit','metric_scale_available':False,
              'normalization':None if unavailable else {'center':[0,0,0],'scale':1,'method':'manufactured'},
              **{key:refs.get(key) for key in arrays},'vertex_count':0 if unavailable else 3,
              'face_count':0 if unavailable else 1,'cameras':cameras,'admission':{} if unavailable else {'passed':True},
              'marker':None,'artwork':None if unavailable else artwork,'frozen':not unavailable,
              'frozen_after_frame_id':None if unavailable else 'seed','prefix_frame_ids':['seed'],
              'reconstruction':{},'preview':{},'evaluator_inputs_read':False}
        wc.SurfacePackage.from_dict(body)
        (folder/'surface.json').write_bytes(wc.canonical_json(body))
        return folder

    def require_not_pass(self, call):
        try: result=call()
        except (ValueError, RuntimeError, OSError): return
        self.assertIn(result['closure_status'],['FAIL','UNVERIFIED','UNAVAILABLE'])

    def test_valid_triangle_closure_and_exact_complete_payload_accounting(self):
        b,a,g=self.prepared(); result=self.invoke(b,a)
        self.assertEqual(result['closure_status'],'VERIFIED')
        self.assertEqual(result['bytes_target'],'PASS'); self.assertEqual(result['triangles_target'],'PASS')
        self.assertEqual(result['artwork_target'],'PASS');self.assertIs(result['mobile_verified'],False)
        expected=sum(p.stat().st_size for p in (b,a,g))
        self.assertEqual(result['accounting']['unique_path_bytes'],expected)
        self.assertEqual(result['accounting']['generated_artwork_raw_bytes'],256*256*3)
        self.assertEqual(result['accounting']['portable_budget_bytes'],expected+256*256*3)
        self.assertIsNone(result['accounting']['allocated_disk_bytes'])

    def test_actual_triangle_failure_is_distinct_from_verified_closure(self):
        b,a,_=self.prepared(triangles=31606); result=self.invoke(b,a)
        self.assertEqual(result['closure_status'],'VERIFIED');self.assertEqual(result['triangles_target'],'FAIL')
        self.assertIs(result['mobile_verified'],False)

    def test_uninspected_required_extension_images_are_unknown_not_zero(self):
        def required_texture_extension(doc):
            doc['extensionsRequired']=['KHR_texture_transform']
            doc['images']=[{'uri':f'not-opened-{i}.png'} for i in range(4)]
            doc['textures']=[{'source':i} for i in range(4)]
        b,a,_=self.prepared(document_mutator=required_texture_extension)
        result=self.invoke(b,a)
        self.assertEqual(result['closure_status'],'UNVERIFIED')
        for key in ('bytes_target','triangles_target','artwork_target'):
            self.assertIsNone(result[key])
        diagnostic=result['embedded_texture_diagnostic']
        self.assertEqual(diagnostic['inspection_status'],'not_inspected')
        for key in ('images','max_edge_px','over_1024','unknown_dimensions'):
            self.assertIsNone(diagnostic[key],key)

    def test_inspected_glb_without_images_has_measured_zero_counts(self):
        b,a,_=self.prepared();result=self.invoke(b,a)
        diagnostic=result['embedded_texture_diagnostic']
        self.assertEqual(diagnostic['inspection_status'],'inspected')
        for key in ('images','over_1024','unknown_dimensions'):
            self.assertEqual(diagnostic[key],0,key)
        self.assertIsNone(diagnostic['max_edge_px'])
        self.assertEqual(result['closure_status'],'VERIFIED')

    def test_unsupported_buffer_images_are_not_reported_absent(self):
        b,a,_=self.prepared(document_mutator=lambda doc:doc['buffers'][0].update(uri='never-opened.bin'))
        result=self.invoke(b,a)
        self.assertEqual(result['closure_status'],'UNVERIFIED')
        diagnostic=result['embedded_texture_diagnostic']
        self.assertEqual(diagnostic['inspection_status'],'not_inspected')
        self.assertIsNone(diagnostic['images'])

    def test_semantic_artwork_tamper_cannot_hide_behind_self_consistent_record_hash(self):
        b,a,_=self.prepared(artwork_mutator=lambda rec:rec['artwork'].update(sha256='0'*64))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_bundle_hash_tamper_rejected_even_at_same_file_length(self):
        b,a,g=self.prepared();raw=g.read_bytes();g.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_out_of_range_indices_fail_before_optimistic_triangle_count(self):
        b,a,g=self.prepared(); raw=bytearray(g.read_bytes())
        # The last triangle lives in the BIN payload with two bytes of final padding.
        raw[-4:-2]=struct.pack('<H',99);g.write_bytes(raw)
        bundle=json.loads(b.read_bytes());bundle['glb_sha256']=sha(raw);b.write_bytes(wc.canonical_json(bundle))
        record=json.loads(a.read_bytes());record['glb_sha256']=sha(raw);record.pop('record_sha256')
        record['record_sha256']=wc.canonical_sha256(record);a.write_bytes(wc.canonical_json(record))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_external_uri_is_never_opened(self):
        b,a,_=self.prepared(document_mutator=lambda doc:doc['buffers'][0].update(uri='forbidden-evaluator.json'))
        original=Path.read_bytes
        def read(path):
            if path.name=='forbidden-evaluator.json': self.fail('external evaluator URI was read')
            return original(path)
        with mock.patch.object(Path,'read_bytes',read):self.require_not_pass(lambda:self.invoke(b,a))

    def test_unsupported_required_extension_never_passes(self):
        b,a,_=self.prepared(document_mutator=lambda doc:doc.update(extensionsRequired=['UNSUPPORTED_extension']))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_marker_face_bound_and_seed_identity_checked(self):
        b,a,_=self.prepared(artwork_mutator=lambda rec:rec['marker'].update(face_id=2))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_root_escape_rejected_without_opening_escaped_asset(self):
        b,a,_=self.prepared();bundle=json.loads(b.read_bytes());bundle['glb_path']='../forbidden.glb'
        b.write_bytes(wc.canonical_json(bundle));self.require_not_pass(lambda:self.invoke(b,a))

    def test_small_surface_counts_manifest_all_arrays_and_generated_artwork(self):
        folder=self.surface();result=self.audit.audit_surface_package(folder,root=self.root)
        expected=sum(p.stat().st_size for p in folder.iterdir())
        self.assertEqual(result['closure_status'],'VERIFIED')
        self.assertEqual(result['accounting']['unique_path_bytes'],expected)
        self.assertEqual(result['accounting']['portable_budget_bytes'],expected+256*256*3)
        self.assertEqual(result['triangles_target'],'PASS');self.assertEqual(result['bytes_target'],'PASS')

    def test_unavailable_surface_never_passes_because_manifest_is_small(self):
        folder=self.surface(unavailable=True);result=self.audit.audit_surface_package(folder,root=self.root)
        self.assertEqual(result['closure_status'],'UNAVAILABLE')
        for key in ['bytes_target','triangles_target','artwork_target']:self.assertIsNone(result[key])

    def test_descriptor_point_and_value_rows_must_match(self):
        folder=self.surface(descriptor_rows=3)
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_unknown_sidecar_blocks_standalone_package_acceptance(self):
        folder=self.surface();(folder/'unknown-sidecar.bin').write_bytes(b'not silently omitted')
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_missing_camera_depth_is_a_real_dependency_failure(self):
        folder=self.surface(depth='missing')
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_unsupported_depth_encoding_cannot_claim_verified_runtime_closure(self):
        folder=self.surface(depth='unsupported')
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_same_length_numeric_file_corruption_is_not_just_a_size_check(self):
        folder=self.surface();path=folder/'vertices.npy';raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_existing_output_is_preserved(self):
        b,a,g=self.prepared()
        inventory={'schema':'workbench_prepared_cad_target_inventory_v1','rows':[{
            'case':'fixture','bundle':str(b),'bundle_sha256':sha(b.read_bytes()),
            'glb':str(g),'glb_sha256':sha(g.read_bytes()),'glb_bytes':g.stat().st_size,
            'triangles':1,'triangles_match_serving_record':True,'artwork_texture_edge_px':256}]}
        manifest=self.root/'inventory.json';manifest.write_bytes(wc.canonical_json(inventory))
        output=self.root/'existing.json';output.write_bytes(b'unchanged evidence')
        try:self.audit.main(['--prepared-inventory',str(manifest),'--artwork','fixture='+str(a),'--out',str(output)])
        except SystemExit: pass
        self.assertEqual(output.read_bytes(),b'unchanged evidence')

    def test_prepared_audit_never_attempts_model_imports_or_scored_evidence_reads(self):
        b,a,_=self.prepared(); attempted=[]; original_import=builtins.__import__; original_read=Path.read_bytes
        def checked_import(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','torchvision','pyrender','OpenGL','sam2'}:
                attempted.append('import '+name);raise AssertionError('forbidden model/runtime import')
            return original_import(name,*args,**kwargs)
        def checked_read(path):
            if path.name in {'results.jsonl','report.json','evaluator.json','scene_gt.json'} or path.suffix in {'.pt','.pth'}:
                attempted.append('read '+str(path));raise AssertionError('forbidden inference/evaluator input')
            return original_read(path)
        with mock.patch.object(builtins,'__import__',checked_import),mock.patch.object(Path,'read_bytes',checked_read):
            result=self.invoke(b,a)
        self.assertEqual(attempted,[]);self.assertEqual(result['closure_status'],'VERIFIED')

    def test_hardlinks_duplicates_and_equal_content_do_not_shrink_transfer_budget(self):
        first=self.root/'first.bin';linked=self.root/'linked.bin';equal=self.root/'equal.bin'
        first.write_bytes(b'a'*100);os.link(first,linked);equal.write_bytes(first.read_bytes())
        refs=[]
        for index,path in enumerate([first,first,linked,equal]):
            st=path.stat();refs.append({'role':str(index),'path':str(path),'bytes':100,
                                       'file_id':(st.st_dev,st.st_ino)})
        result=self.audit.account(refs,256*256*3)
        self.assertEqual(result['reference_bytes'],400)
        self.assertEqual(result['unique_path_bytes'],300)
        self.assertEqual(result['unique_file_object_bytes'],200)
        self.assertEqual(result['portable_budget_bytes'],300+256*256*3)
        refs[-1]['file_id']=None;unknown=self.audit.account(refs,0)
        self.assertIsNone(unknown['unique_file_object_bytes'])
        self.assertEqual(unknown['portable_budget_bytes'],300)

    def test_imported_artwork_without_locator_remains_unverified(self):
        folder=self.surface();p=folder/'surface.json';doc=json.loads(p.read_bytes())
        doc['artwork']['generation']='imported';p.write_bytes(wc.canonical_json(doc))
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_object_dtype_cannot_execute_pickle(self):
        folder=self.surface();path=folder/'vertices.npy'
        with path.open('wb') as stream:np.save(stream,np.array([{'manufactured':'object'}],dtype=object),allow_pickle=True)
        p=folder/'surface.json';doc=json.loads(p.read_bytes());raw=path.read_bytes()
        doc['vertices'].update(bytes=len(raw),sha256=sha(raw));p.write_bytes(wc.canonical_json(doc))
        self.require_not_pass(lambda:self.audit.audit_surface_package(folder,root=self.root))

    def test_large_embedded_cad_material_does_not_fail_small_overlay_artwork(self):
        from bench import workbench_image_io as imageio
        png=imageio.encode_png_rgba(np.zeros((2048,2,4),np.uint8))
        b,a,g=self.prepared(embedded_png=png);result=self.invoke(b,a)
        self.assertEqual(result['closure_status'],'VERIFIED');self.assertEqual(result['artwork_target'],'PASS')
        self.assertEqual(result['geometry']['embedded_images'][0]['height'],2048)
        self.assertEqual(result['accounting']['unique_path_bytes'],sum(p.stat().st_size for p in (b,a,g)))

    def test_unsupported_optional_geometry_semantics_never_pass(self):
        mutations=[lambda doc:doc['nodes'][0].update(extensions={'EXT_mesh_gpu_instancing':{}}),
                   lambda doc:doc.update(extensions={'unknown_optional_semantics':{}}),
                   lambda doc:doc.update(skins=[{'joints':[0]}]),
                   lambda doc:doc['meshes'][0]['primitives'][0].update(targets=[{'POSITION':0}])]
        for mutate in mutations:
            with self.subTest():
                b,a,_=self.prepared(document_mutator=mutate)
                self.require_not_pass(lambda:self.invoke(b,a))

    def test_prepared_bundle_semantics_cannot_be_hidden_by_matching_artwork(self):
        variants=[('principal_point_origin','invented'),('metric_provenance','relative_scale'),
                  ('seed_label','evaluator:scored_pose'),('seed_pose_m',None),
                  ('seed_pose_m',[[1,0],[0,1]]),('seed_frame_id','bad\nidentifier')]
        for key,value in variants:
            with self.subTest(key=key,value=value):
                b,a,_=self.prepared();bundle=json.loads(b.read_bytes());record=json.loads(a.read_bytes())
                bundle[key]=value
                target='anchor_frame_id' if key=='seed_frame_id' else key
                if target in record:record[target]=value
                b.write_bytes(wc.canonical_json(bundle));record.pop('record_sha256')
                record['record_sha256']=wc.canonical_sha256(record);a.write_bytes(wc.canonical_json(record))
                self.require_not_pass(lambda:self.invoke(b,a))

    def test_artwork_seed_label_must_match_bundle(self):
        b,a,_=self.prepared(artwork_mutator=lambda record:record.update(seed_label='controlled_init:different_source'))
        self.require_not_pass(lambda:self.invoke(b,a))

    def test_successful_cli_never_opens_unrelated_clip_records(self):
        b,a,g=self.prepared();manifest=self.root/'inventory.json'
        bundle=json.loads(b.read_bytes());bundle['glb_path']=g.relative_to(SCRATCH.parents[2]).as_posix()
        b.write_bytes(wc.canonical_json(bundle))
        manifest.write_bytes(wc.canonical_json({'schema':'workbench_prepared_cad_target_inventory_v1','rows':[{
            'case':'fixture','bundle':str(b.relative_to(SCRATCH.parents[2])),'bundle_sha256':sha(b.read_bytes()),'glb':str(g.relative_to(SCRATCH.parents[2])),
            'glb_sha256':sha(g.read_bytes()),'glb_bytes':g.stat().st_size,'triangles':1,
            'bundle_plus_glb_bytes':b.stat().st_size+g.stat().st_size,
            'triangles_match_serving_record':True,'artwork_texture_edge_px':256}]}))
        original=Path.open;attempts=[]
        def checked(path,*args,**kwargs):
            if path.name=='clip.json':attempts.append(str(path));raise AssertionError('unrelated clip record read')
            return original(path,*args,**kwargs)
        out=self.root/'audit.json'
        # Host free space is an external variable; this manufactured publication
        # control supplies a budget explicitly, and does not certify a real audit.
        with mock.patch.object(Path,'open',checked),mock.patch.object(self.audit.shutil,'disk_usage',return_value=SimpleNamespace(free=10<<30)):
            status=self.audit.main(['--prepared-inventory',str(manifest),'--artwork','fixture='+str(a),'--out',str(out)])
        self.assertEqual(attempts,[]);self.assertEqual(status,0,json.loads(out.read_bytes())['rows'][0]['closure_reasons'])
        self.assertEqual(json.loads(out.read_bytes())['rows'][0]['closure_status'],'VERIFIED')


if __name__=='__main__': unittest.main()
