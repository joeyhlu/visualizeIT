"""Independent manufactured UV/metric checks, without torch, GL or evaluation labels."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tools import workbench_ycb_uv_proof as proof

class UVProofTests(unittest.TestCase):
    def test_asymmetric_texel_centres_and_bilinear_middle(self):
        texture=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],np.uint8)
        uv=np.array([[.25,.75],[.75,.75],[.25,.25],[.75,.25],[.5,.5],[-10,10]])
        got=proof.sample_bilinear_bop(texture,uv)
        np.testing.assert_allclose(got,[[1,0,0],[0,1,0],[0,0,1],[1,1,1],[.5,.5,.5],[1,0,0]],atol=1e-15)

    def test_metric_projection_requires_consistent_units(self):
        K=np.array([[200.,0,320],[0,200,240],[0,0,1]])
        X=np.array([[100.,50.,200.],[0.,0.,0.]])
        t=np.array([0.,0.,1000.])
        expected=np.array([[320+200/12,240+100/12],[320,240]])
        uv,_=proof.project(K,np.eye(3),t,X)
        np.testing.assert_allclose(uv,expected)
        metre,_=proof.project(K,np.eye(3),t*.001,X*.001)
        np.testing.assert_allclose(metre,uv)
        for vertices,translation in ((X,t*.001),(X*.001,t)):
            wrong,_=proof.project(K,np.eye(3),translation,vertices)
            self.assertGreater(np.max(abs(wrong-uv)),10.)

    def test_sloping_triangle_intersection_is_perspective_correct(self):
        mesh={'positions_mm':np.array([[-100.,-100.,1000.],[100.,-100.,1000.],[0.,100.,2000.]]),
              'triangles':np.array([[0,1,2]]),'uv':np.array([[0.,0.],[1.,0.],[.5,1.]])}
        inputs={'mesh':mesh,'texture':np.full((2,2,3),128,np.uint8)}
        image={'K':np.array([[100.,0,320.],[0,100.,240.],[0,0,1]]),'R':np.eye(3),'t_mm':np.zeros(3)}
        got=proof.reference_image(inputs,image)
        self.assertTrue(got['coverage'][240,320])
        self.assertAlmostEqual(got['depth_mm'][240,320],1500.,places=8)
        np.testing.assert_allclose(got['uv'][240,320],[.5,.5],atol=1e-12)

    def inputs_and_labels(self):
        inputs={'object_id':2,'texture_info':{'sha256':'a'*64},'texture':np.zeros((4,4,3),np.uint8),
                'mesh':{'positions_mm':np.array([[0.,0.,1000.],[100.,0.,1000.],[0.,100.,1000.]]),
                        'triangles':np.array([[0,1,2]]),'uv':np.array([[0.,0.],[1.,0.],[0.,1.]])},
                'images':[{'frame_id':fid,'rgb_sha256':str(fid)*16,'K':np.diag([10.,10.,1.]),
                           'R':np.eye(3),'t_mm':np.zeros(3)} for fid in (361,370)]}
        labels={'schema':proof.LANDMARK_SCHEMA,'object_id':2,'texture_sha256':'a'*64,
                'selected_by':'manufactured arithmetic fixture','landmarks':[]}
        for i,(u,v) in enumerate(((.2,.2),(.3,.2),(.2,.3))):
            labels['landmarks'].append({'landmark_id':f'point{i}','triangle':0,'barycentric':[1-u-v,u,v],
                'texture_xy':[4*u-.5,4*(1-v)-.5],
                'images':[{'frame_id':img['frame_id'],'rgb_sha256':img['rgb_sha256'],
                           'visible':True,'rgb_xy':[u,v]} for img in inputs['images']]})
        return inputs,labels

    def evaluate(self,inputs,labels):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'landmarks.json'
            path.write_text(json.dumps(labels),encoding='utf-8')
            return proof.evaluate_landmarks(path,inputs)

    def test_missing_or_occluded_landmarks_are_inconclusive(self):
        inputs,labels=self.inputs_and_labels()
        self.assertFalse(proof.evaluate_landmarks(None,inputs)['passed'])
        for lm in labels['landmarks']:lm['images'][0]['visible']=False
        self.assertFalse(self.evaluate(inputs,labels)['passed'])

    def test_three_distinct_analytic_landmarks_pass_arithmetic_only(self):
        inputs,labels=self.inputs_and_labels()
        self.assertTrue(self.evaluate(inputs,labels)['passed'])

    def test_repeated_landmark_does_not_satisfy_three_distinct_points(self):
        inputs,labels=self.inputs_and_labels()
        labels['landmarks']=[copy.deepcopy(labels['landmarks'][0]) for _ in range(3)]
        with self.assertRaises(proof.ProofError):self.evaluate(inputs,labels)

    def test_one_bad_image_blocks_correspondence(self):
        inputs,labels=self.inputs_and_labels()
        for lm in labels['landmarks']:lm['images'][1]['rgb_xy']=[50,50]
        self.assertFalse(self.evaluate(inputs,labels)['passed'])

    def test_wrong_texture_or_non_onboarding_observation_is_rejected(self):
        for mutation in ('texture','frame'):
            inputs,labels=self.inputs_and_labels()
            if mutation=='texture':labels['texture_sha256']='b'*64
            else:labels['landmarks'][0]['images'][0]['frame_id']=999
            with self.subTest(mutation=mutation),self.assertRaises(proof.ProofError):self.evaluate(inputs,labels)

    def test_duplicate_point_with_different_ids_is_rejected(self):
        inputs,labels=self.inputs_and_labels()
        labels['landmarks']=[copy.deepcopy(labels['landmarks'][0]) for _ in range(3)]
        for i,lm in enumerate(labels['landmarks']):lm['landmark_id']=str(i)
        with self.assertRaises(proof.ProofError):self.evaluate(inputs,labels)

    def test_shared_vertex_through_different_triangles_is_one_point(self):
        inputs,labels=self.inputs_and_labels()
        inputs['mesh']['triangles']=np.array([[0,1,2],[0,2,1],[0,1,2]])
        inputs['mesh']['uv'][0]=[.25,.75]
        for i,lm in enumerate(labels['landmarks']):
            lm.update(triangle=i,barycentric=[1.,0.,0.],texture_xy=[.5+i*.1,.5+i*.1])
            for obs in lm['images']:obs['rgb_xy']=[i,i]
        with self.assertRaises(proof.ProofError):self.evaluate(inputs,labels)

    def test_repeated_image_coordinate_does_not_count_as_distinct(self):
        inputs,labels=self.inputs_and_labels()
        for lm in labels['landmarks']:
            for obs in lm['images']:obs['rgb_xy']=[.25,.25]
        with self.assertRaises(proof.ProofError):self.evaluate(inputs,labels)

    def test_landmark_shapes_numeric_types_and_visibility_are_strict(self):
        for mutation in ('shape','number_string','visibility','nonfinite'):
            inputs,labels=self.inputs_and_labels()
            lm=labels['landmarks'][0]
            if mutation=='shape':lm['images'][0]['rgb_xy']=[.2]
            elif mutation=='number_string':lm['barycentric']=['0.6',.2,.2]
            elif mutation=='visibility':lm['images'][0]['visible']='true'
            else:lm['barycentric']=[float('nan'),.2,.2]
            with self.subTest(mutation=mutation),self.assertRaises((proof.ProofError,proof.yc.ControlError)):
                self.evaluate(inputs,labels)

    def samples(self,n=128):
        return {'y':np.zeros(n,dtype=np.int64),'x':np.arange(n,dtype=np.int64),
                'face_id':np.zeros(n,dtype=np.int64),'barycentric':np.full((n,3),1/3,np.float64),
                'depth_mm':np.ones(n,np.float64),'uv':np.full((n,2),.5,np.float64),
                'rgb':np.zeros((n,3),np.float64)}

    def test_sample_count_derives_from_distinct_finite_coordinates(self):
        self.assertTrue(proof.validate_samples(self.samples())['enough'])
        self.assertFalse(proof.validate_samples(self.samples(1))['enough'])
        for mutation in ('duplicate','outside','nonfinite','dtype','shape'):
            arrays=self.samples()
            if mutation=='duplicate':arrays['x'][:]=0
            elif mutation=='outside':arrays['x'][0]=640
            elif mutation=='nonfinite':arrays['rgb'][0,0]=np.inf
            elif mutation=='dtype':arrays['x']=arrays['x'].astype(float)
            else:arrays['uv']=np.ones((128,1))
            with self.subTest(mutation=mutation),self.assertRaises(proof.ProofError):proof.validate_samples(arrays)

    def test_stale_or_invented_prepare_rejected_before_output(self):
        for mutation in ('sha','threshold','tool','inventory'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); prepared=root/'prepared';prepared.mkdir();out=root/'render'
                record={'schema':proof.SCHEMA_PREPARE,'object_id':2,
                        'tool_sha256':proof.sha256_file(Path(proof.__file__)),
                        'thresholds':dict(proof.THRESHOLDS),'reference_method':proof.REFERENCE_METHOD,
                        'reviewed_source_pins':proof.REVIEWED_SOURCE_PINS,
                        'candidate_pins':{str(k):list(v) for k,v in proof.CANDIDATE_PINS.items()},
                        'files':{'../outside':'0'*64}}
                if mutation=='threshold':record['thresholds']['min_valid_pixels']=1
                elif mutation=='tool':record['tool_sha256']='0'*64
                raw=json.dumps(record).encode();(prepared/'prepared.json').write_bytes(raw)
                digest='0'*64 if mutation=='sha' else proof.sha256_bytes(raw)
                with self.assertRaises(proof.ProofError):
                    proof.render(prepared,out,expected_prepared_sha256=digest)
                self.assertFalse(out.exists())

    def test_asset_basename_and_exact_image_inventory_checked_before_samples(self):
        for mutation in ('basename','zero_images','one_image','three_images'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);prepared=root/'prepared';prepared.mkdir();assets=root/'assets';assets.mkdir()
                (assets/'candidate-model.glb').write_bytes(b'candidate');(assets/'other.glb').write_bytes(b'other')
                images=[{'frame_id':fid,'logical_index':logical,'rgb_sha256':str(fid)*16,
                         'K':np.eye(3),'R':np.eye(3),'t_mm':np.zeros(3)} for logical,fid in ((0,361),(9,370))]
                inputs={'candidate_dir':assets,'glb_sha256':'a'*64,'receipt_sha256':'b'*64,
                        'manifest_sha256':'c'*64,'ply_sha256':'d'*64,'texture_info':{'sha256':'e'*64},'images':images}
                record={'schema':proof.SCHEMA_PREPARE,'object_id':2,
                        'tool_sha256':proof.sha256_file(Path(proof.__file__)),
                        'thresholds':dict(proof.THRESHOLDS),'reference_method':proof.REFERENCE_METHOD,
                        'reviewed_source_pins':proof.REVIEWED_SOURCE_PINS,
                        'candidate_pins':{str(k):list(v) for k,v in proof.CANDIDATE_PINS.items()},
                        'inputs':{'candidate_glb_path':str(assets/('other.glb' if mutation=='basename' else 'candidate-model.glb')),
                                  'candidate_glb_sha256':'a'*64,'candidate_receipt_sha256':'b'*64,
                                  'manifest_sha256':'c'*64,'ply_sha256':'d'*64,'texture':{'sha256':'e'*64}},
                        'images':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in image.items()} for image in images],
                        'image_inventory':[{k:image[k] for k in ('frame_id','logical_index','rgb_sha256')} for image in images],
                        'source_convention':{'passed':True},'source_convention_pass':True,
                        'onboarding_correspondence':{'status':'inconclusive'},'onboarding_correspondence_pass':'inconclusive',
                        'serving_render_pass':'pending_render_phase','final_status':'inconclusive','files':{}}
                if mutation=='zero_images':record['images']=[]
                elif mutation=='one_image':record['images']=record['images'][:1]
                elif mutation=='three_images':record['images'].append(copy.deepcopy(record['images'][0]))
                for image in images:
                    for suffix in proof.PER_IMAGE_FILES:
                        name=f"frame-{image['frame_id']:06d}"+suffix
                        (prepared/name).write_bytes(b'fixture')
                        record['files'][name]=proof.sha256_bytes(b'fixture')
                raw=json.dumps(record).encode();(prepared/'prepared.json').write_bytes(raw)
                with patch.object(proof,'authenticate',return_value=inputs),\
                     patch.object(proof,'source_convention',return_value={'passed':True}),\
                     patch.object(proof,'evaluate_landmarks',return_value={'status':'inconclusive'}),\
                     patch.object(proof,'reference_image') as reference:
                    with self.assertRaises(proof.ProofError):
                        proof.validate_prepared(prepared,proof.sha256_bytes(raw),texture_dir=root,manifest=root/'manifest.json')
                    reference.assert_not_called()

if __name__=='__main__':unittest.main()
