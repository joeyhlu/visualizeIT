"""Codex causal/lifecycle tests with a fake predictor; no SAM, Torch or GPU."""
from pathlib import Path
from unittest.mock import patch
import tempfile,unittest
import numpy as np
from tools import workbench_sam_forward_isolation as isolation

class Frames:
    def __init__(self,label,count,released,on_read):
        self.label,self.count,self.released,self.on_read=label,count,released,on_read
        self.closed=False
    def __len__(self):return self.count
    def release(self,index):self.released=index
    def read(self,index):self.on_read(self.label,index);return index
    def close(self):self.closed=True

class Predictor:
    def __init__(self,fault=None):self.fault=fault;self.states=[];self.log=[];self.first=None;self.readers=[];self.device_calls=0
    def make_frames(self,label,count,released,on_read):
        self.log.append(('reader',label,count,released))
        reader=Frames(label,count-1 if self.fault=='reader_length' else count,released,on_read)
        self.readers.append(reader);return reader
    def open_state(self,frames):
        frames.read(0)
        if self.fault=='open_state':raise RuntimeError('open failure')
        if self.fault=='reuse' and self.first is not None:return self.first
        state={'frames':frames,'reverse_memory':False};self.states.append(state)
        if self.first is None:self.first=state
        self.log.append(('init',frames.label));return state
    def add_prompt(self,state,index,points,labels):state['frames'].read(index);self.log.append(('prompt',index))
    def propagate(self,state,start,reverse):
        frames=state['frames'];self.log.append(('propagate',frames.label,reverse))
        if not reverse and state['reverse_memory']:raise RuntimeError('contaminated forward state')
        if reverse:state['reverse_memory']=True
        order=list(range(start,-1,-1)) if reverse else list(range(start,len(frames)))
        if self.fault=='missing' and not reverse:order=order[:-1]
        if self.fault=='future' and not reverse:frames.read(start+1)
        if self.fault=='reverse_future' and reverse:frames.read(start+1)
        for index in order:
            frames.read(index);mask=np.ones((3,3),dtype=bool)
            if self.fault in ['mid_forward','mid_forward_cleanup'] and not reverse and index==12:raise RuntimeError('forward compute failure')
            if self.fault=='mid_reverse' and reverse and index==7:raise RuntimeError('reverse compute failure')
            if self.fault=='seed_mismatch' and reverse and index==start:mask[0,0]=False
            yield (index+1 if self.fault=='wrong_index' else index),([2] if self.fault=='wrong_object' else [1]),mask
    def to_mask(self,logits):return logits
    def close_state(self,state,frames):
        self.log.append(('close',frames.label))
        if state is not None:state.clear()
        frames.close()
        if self.fault in ['cleanup','mid_forward_cleanup']:raise RuntimeError('cleanup injected')
        return {'closed':True}
    def close_frames(self,frames):frames.close();return {'closed':True}
    def release_device(self):
        self.device_calls+=1
        if self.fault=='mid_forward_cleanup':raise RuntimeError('device release injected')
        return {'released':True}

class StateIsolationTests(unittest.TestCase):
    def run_fake(self,fault=None):
        adapter=Predictor(fault);writes={}
        def write(index,fid,mask,label):
            self.assertNotIn(index,writes);writes[index]=(fid,mask.copy(),label)
        result=isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
            points=np.array([[1.5,1.5]]),labels=np.array([1]),shape=(3,3),write_mask=write)
        return adapter,writes,result
    def test_fresh_states_full_length_forward_first_and_complete_identity(self):
        adapter,writes,result=self.run_fake()
        self.assertEqual(result['counters'],{'forward_yields':111,'reverse_yields':10,'writes':120,'prompt_adds':2,'state_inits':2,'teardowns':2})
        self.assertEqual([x for x in adapter.log if x[0]=='reader'],[('reader','forward',120,9),('reader','reverse',120,9)])
        self.assertLess(adapter.log.index(('close','forward')),adapter.log.index(('init','reverse')))
        self.assertEqual(adapter.log.count(('propagate','forward',False)),1)
        self.assertEqual(adapter.log.count(('propagate','reverse',True)),1)
        self.assertIsNot(adapter.states[0],adapter.states[1])
        for index,(fid,mask,label) in writes.items():
            self.assertEqual(fid,f's{361+index:06d}');self.assertEqual(label,'forward' if index>=9 else 'reverse')
        self.assertLessEqual(max(result['reads']['reverse']),9)
        self.assertTrue(all(not state for state in adapter.states))
    def test_each_forward_release_follows_previous_emission(self):
        _,_,result=self.run_fake();events=result['events']
        for n,event in enumerate(events):
            if event['event']=='release':
                earlier=[e for e in events[:n] if e['event']=='yield' and e['state']=='forward']
                self.assertEqual(earlier[-1]['local'],event['local']-1)
    def test_future_read_is_rejected_in_both_branches(self):
        for fault in ['future','reverse_future']:
            with self.assertRaises(isolation.IsolationError):self.run_fake(fault)
    def test_reusing_state_cannot_silently_transfer_reverse_memory(self):
        with self.assertRaises(isolation.IsolationError):self.run_fake('reuse')
    def test_wrong_missing_yields_and_object_identity_rejected(self):
        for fault in ['wrong_index','missing','wrong_object']:
            with self.assertRaises(isolation.IsolationError):self.run_fake(fault)
    def test_reverse_duplicate_seed_must_equal_forward(self):
        with self.assertRaises(isolation.IsolationError):self.run_fake('seed_mismatch')
    def test_cleanup_failure_stops_before_reverse_state(self):
        adapter=Predictor('cleanup')
        with self.assertRaises(RuntimeError):
            isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None)
        self.assertNotIn(('init','reverse'),adapter.log)

    def test_reader_is_closed_when_state_initialization_or_length_check_fails(self):
        for fault in ['open_state','reader_length']:
            adapter=Predictor(fault)
            with self.assertRaises((isolation.IsolationError,RuntimeError)):
                isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                    points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None)
            self.assertTrue(adapter.readers[0].closed,fault)

    def test_reused_state_rejection_still_closes_new_reader(self):
        adapter=Predictor('reuse')
        with self.assertRaises(isolation.IsolationError):
            isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None)
        self.assertEqual(len(adapter.readers),2)
        self.assertTrue(all(reader.closed for reader in adapter.readers))

    def test_iterator_close_failure_still_attempts_state_and_reader_cleanup(self):
        class BadClose:
            def __init__(self,inner):self.inner=inner
            def __iter__(self):return self
            def __next__(self):return next(self.inner)
            def close(self):self.inner.close();raise RuntimeError('iterator close failure')
        class Adapter(Predictor):
            def propagate(self,*args):return BadClose(super().propagate(*args))
        adapter=Adapter()
        with self.assertRaises((isolation.IsolationError,RuntimeError)):
            isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None)
        self.assertTrue(adapter.readers[0].closed)
        self.assertTrue(all(not state for state in adapter.states))
        self.assertNotIn(('init','reverse'),adapter.log)

    def test_mid_branch_failures_retain_counters_reads_and_partial_writes(self):
        for fault,expected_forward,expected_reverse,expected_writes in [('mid_forward',3,0,3),('mid_reverse',111,2,112)]:
            adapter=Predictor(fault);ledger=isolation.IsolationLedger(isolation.frame_ids_for())
            with self.assertRaises(RuntimeError):
                isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                    points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None,ledger=ledger)
            snapshot=ledger.snapshot()
            self.assertEqual(snapshot['counters']['forward_yields'],expected_forward)
            self.assertEqual(snapshot['counters']['reverse_yields'],expected_reverse)
            self.assertEqual(snapshot['counters']['writes'],expected_writes)
            self.assertEqual(len(snapshot['written_frame_ids']),expected_writes)
            self.assertTrue(snapshot['events']);self.assertTrue(snapshot['reads'])
            self.assertTrue(all(reader.closed for reader in adapter.readers))

    def test_primary_error_and_all_cleanup_errors_survive_together(self):
        adapter=Predictor('mid_forward_cleanup');ledger=isolation.IsolationLedger(isolation.frame_ids_for())
        with self.assertRaisesRegex(RuntimeError,'forward compute failure') as caught:
            isolation.run_state_isolated(adapter,frame_ids=isolation.frame_ids_for(),prompt_index=9,
                points=np.zeros((1,2)),labels=np.ones(1),shape=(3,3),write_mask=lambda *args:None,ledger=ledger)
        self.assertGreaterEqual(len(caught.exception.isolation_cleanup_errors),2)
        self.assertGreaterEqual(len(ledger.snapshot()['cleanup_errors']),2)
        self.assertGreater(adapter.device_calls,0)
        self.assertTrue(adapter.readers[0].closed);self.assertNotIn(('init','reverse'),adapter.log)

class MaskQualificationTests(unittest.TestCase):
    def setUp(self):
        self.seed=np.zeros((8,8),bool);self.seed[:2,:5]=True
        self.half=np.zeros_like(self.seed);self.half[0,:5]=True
        self.masks={i:self.half.copy() for i in range(120)};self.masks[9]=self.seed.copy()
        self.thresholds={**isolation.THRESHOLDS,'seed_area_exact':10}
    def qualify(self):return isolation.qualify(self.masks,original_seed=self.seed,positive=[(0,0)],negative=[(7,7)],thresholds=self.thresholds)
    def test_half_seed_area_boundary_passes_and_prefix_transition_excluded(self):
        result=self.qualify();self.assertTrue(result['qualified'])
        self.assertEqual(result['gates']['scored_consecutive_iou']['pairs'],109)
    def test_below_half_seed_area_fails_without_hiding_frame(self):
        self.masks[50][0,4]=False
        result=self.qualify();self.assertFalse(result['qualified'])
        self.assertIn(50,result['gates']['scored_area']['frames_below'])
        self.assertEqual(len(result['scored_areas']),110)
    def test_exact_half_iou_passes_and_disjoint_transition_fails(self):
        self.masks[11]=self.seed.copy();self.assertTrue(self.qualify()['qualified'])
        self.masks[12][:]=False;self.masks[12][7,:5]=True
        self.assertFalse(self.qualify()['gates']['scored_consecutive_iou']['passed'])
    def test_seed_content_change_and_missing_frame_rejected(self):
        self.masks[9][1,4]=False;self.masks[9][2,0]=True
        self.assertFalse(self.qualify()['gates']['seed_equals_original']['passed'])
        self.masks.pop(119)
        with self.assertRaises(isolation.IsolationError):self.qualify()
    def test_binary_png_roundtrip_content_hash_and_eight_connectivity(self):
        from bench.workbench_geometry import mask_sha256
        data=isolation.encode_mask_png(self.seed)
        decoded=isolation.decode_mask_png(data,self.seed.shape)
        np.testing.assert_array_equal(decoded,self.seed)
        self.assertEqual(isolation.mask_content_sha256(decoded),mask_sha256(self.seed))
        self.assertEqual(isolation.largest_component_fraction(np.eye(2,dtype=bool)),1.)
        self.assertEqual(isolation.iou(np.zeros((2,2),bool),np.zeros((2,2),bool)),0.)
    def test_nonbinary_wrongshape_and_truncated_png_rejected(self):
        from bench.vision import cv2
        _,encoded=cv2.imencode('.png',np.ones((8,8),np.uint8))
        for data,shape in [(encoded.tobytes(),(8,8)),(isolation.encode_mask_png(self.seed),(9,8)),(b'badPNG',(8,8))]:
            with self.assertRaises(isolation.IsolationError):isolation.decode_mask_png(data,shape)

class WriterAndPlanTests(unittest.TestCase):
    def test_writer_cap_containment_immutability_and_budget_before_write(self):
        with tempfile.TemporaryDirectory(dir=isolation.ROOT/'.cache') as tmp:
            out=Path(tmp)/'out';writer=isolation.OutputWriter(out,cap_bytes=4);writer.create();writer.write('one',b'12')
            for name,data in [('../escape',b'x'),('one',b'x'),('two',b'123')]:
                with self.assertRaises(isolation.IsolationError):writer.write(name,data)
            with self.assertRaises(isolation.IsolationError):isolation.OutputWriter(out).create()
            with self.assertRaises(isolation.IsolationError):isolation._contained(out,out.parent/'escape')
            def deny(_):raise isolation.IsolationError('reserve')
            blocked=isolation.OutputWriter(Path(tmp)/'blocked',budget_check=deny);blocked.create()
            with self.assertRaises(isolation.IsolationError):blocked.write('never',b'x')
            self.assertFalse((blocked.dir/'never').exists())
    def test_changed_sources_or_inputs_stop_before_gpu(self):
        plan={'workspace':'.cache/video-workbench','plan_id':'test','runtime':{'device':'cuda'},'source':'frozen'}
        with patch.object(isolation,'_load_plan',return_value=(plan,isolation.canonical_bytes(plan))),\
             patch.object(isolation,'build_plan',return_value={**plan,'source':'changed'}),\
             patch.object(isolation,'SamAdapter') as model:
            with self.assertRaises(isolation.IsolationError):isolation.run(Path('unused'),'a'*64)
            model.assert_not_called()

class OutputReadbackTests(unittest.TestCase):
    def fixture(self,folder):
        from io import BytesIO
        from PIL import Image
        ids=isolation.frame_ids_for(count=3);writer=isolation.OutputWriter(folder);writer.create();masks={}
        mask=np.ones((3,3),bool)
        for i,fid in enumerate(ids):
            data=isolation.encode_mask_png(mask);name=f'm{i:05d}.png';digest=writer.write(name,data)
            masks[fid]={'path':name,'sha256':digest,'width':3,'height':3,'content_sha256':isolation.mask_content_sha256(mask),
                        'empty':False,'state':'reverse' if i<1 else 'forward','local':i}
        events=[{'event':'yield','state':s} for s in ['forward','forward','reverse','reverse']]+[{'event':'write'}]*3
        writer.write('events.jsonl',b'\n'.join(isolation.canonical_bytes(e) for e in events)+b'\n')
        stream=BytesIO();Image.new('RGB',(3,3)).save(stream,format='PNG');writer.write('preview.png',stream.getvalue())
        manifest={'schema':isolation.MASK_SCHEMA,'clip_id':isolation.CLIP_ID,'evaluator_inputs_read':False,'cancelled':False,'masks':masks}
        writer.write('manifest.json',isolation.canonical_bytes(manifest))
        return writer,ids
    def verify(self,writer,ids):
        return isolation.verify_staged_outputs(writer.dir,recorded=writer.files,frame_ids=ids,prompt_index=1,shape=(3,3),preview_size=(3,3))
    def test_exact_staged_files_receive_real_readback_proof(self):
        with tempfile.TemporaryDirectory(dir=isolation.ROOT/'.cache') as tmp:
            writer,ids=self.fixture(Path(tmp)/'out');result=self.verify(writer,ids)
            self.assertEqual(result['proof']['mask_count'],3)
            self.assertEqual(set(result['proof']['files']),set(writer.files))
    def test_changed_mask_bytes_even_with_identical_pixels_rejected(self):
        from io import BytesIO
        from PIL import Image,PngImagePlugin
        with tempfile.TemporaryDirectory(dir=isolation.ROOT/'.cache') as tmp:
            writer,ids=self.fixture(Path(tmp)/'out');path=writer.dir/'m00002.png';original=path.read_bytes()
            info=PngImagePlugin.PngInfo();info.add_text('changed','bytes only')
            stream=BytesIO();Image.fromarray(np.full((3,3),255,np.uint8)).save(stream,format='PNG',pnginfo=info)
            replacement=stream.getvalue();self.assertNotEqual(original,replacement)
            np.testing.assert_array_equal(isolation.decode_mask_png(original,(3,3)),isolation.decode_mask_png(replacement,(3,3)))
            path.write_bytes(replacement)
            with self.assertRaises(isolation.IsolationError):self.verify(writer,ids)
    def test_modified_nonseed_mask_events_preview_or_manifest_rejected(self):
        for name in ['m00002.png','events.jsonl','preview.png','manifest.json']:
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=isolation.ROOT/'.cache') as tmp:
                writer,ids=self.fixture(Path(tmp)/'out');path=writer.dir/name;path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaises(isolation.IsolationError):self.verify(writer,ids)
    def test_missing_extra_and_hard_linked_outputs_rejected(self):
        import os
        for fault in ['missing','extra','hard_link']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory(dir=isolation.ROOT/'.cache') as tmp:
                writer,ids=self.fixture(Path(tmp)/'out');path=writer.dir/'m00002.png'
                if fault=='missing':path.unlink()
                elif fault=='extra':(writer.dir/'extra.txt').write_text('unexpected')
                else:
                    copy=Path(tmp)/'linked.png';copy.write_bytes(path.read_bytes());path.unlink();os.link(copy,path)
                with self.assertRaises(isolation.IsolationError):self.verify(writer,ids)

if __name__=='__main__':unittest.main()
