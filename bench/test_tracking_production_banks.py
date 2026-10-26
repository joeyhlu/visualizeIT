"""Cache contracts tested with manufactured arrays; no expensive mesh raster."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from . import tracking_production_banks as banks


def arrays(count=16,seed=4):
    p=np.arange(count*3,dtype=np.float64).reshape(count,3)/1000
    return (p,np.zeros((count,32),dtype=np.uint8),np.zeros((8,8),dtype=np.uint8),
            np.zeros((seed,2),dtype=np.float32),p[:seed].copy())


class ProductionBankTests(unittest.TestCase):
    def setUp(self):
        # Tiny tempfile fixtures live on the4.9GiB /tmp mount, not the workspace
        # disk used in real preparation. Test real disk guards separately.
        patcher=patch.object(banks,'MIN_FREE_BYTES',0)
        patcher.start();self.addCleanup(patcher.stop)

    def fixture(self,root):
        dataset=root/'data';dataset.mkdir();cache=root/'cache'
        def ref(name):
            p=dataset/name;p.write_bytes(name.encode());return dict(path=name,sha256=banks.sha(p))
        rgb,depth,model=ref('rgb.bin'),ref('depth.bin'),ref('model.ply')
        onboard=[dict(frameId=i,rgb=rgb,depth=depth,calibration=dict(cam_K=[1.,0,0,0,1.,0,0,0,1.],depth_scale=1.),
            evaluationPose=dict(cam_R_m2c=[1.,0,0,0,1.,0,0,0,1.],cam_t_m2c=[0,0,500])) for i in range(10)]
        obj=dict(objectId=20,name='unit object',model=model,onboarding=onboard,
                 frames=[dict(frameId=10,rgb=dict(path='MUST-NOT-OPEN',sha256='0'*64))])
        (dataset/'inference.json').write_text(json.dumps(dict(schemaVersion=1,objects=[obj])))
        return dataset,cache,obj

    def build(self,dataset,cache,value):
        mesh=object();transform=np.eye(4);builder=lambda *args:value
        with patch.object(banks,'production_api',return_value=(builder,lambda *a:(SimpleNamespace(original=mesh),None,transform))):
            metadata=banks.build_one(dataset,cache,20)
        return metadata

    def test_exact_builder_arrays_and_cache_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,cache,obj=self.fixture(Path(temp));value=arrays(16,12);mesh=object();transform=np.eye(4)
            with patch.object(banks,'production_api') as api:
                from unittest.mock import Mock
                builder=Mock(return_value=value);reader=Mock(return_value=(SimpleNamespace(original=mesh),None,transform));api.return_value=(builder,reader)
                metadata=banks.build_one(dataset,cache,20)
            builder.assert_called_once()
            args=builder.call_args.args
            self.assertEqual(args[0],dataset);self.assertEqual(args[1],obj['onboarding']);self.assertIs(args[2],mesh);self.assertIs(args[3],transform)
            loaded=banks.load_bank(dataset,cache,20)
            self.assertEqual(metadata['status'],'ready');self.assertEqual(metadata['anchors'],16)
            for a,b in zip(value,loaded.arrays):
                self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape);np.testing.assert_array_equal(a,b);self.assertFalse(b.flags.writeable)
            self.assertFalse(metadata['evaluatorOpened']);self.assertFalse(metadata['scoredFramesOpened'])

    def test_low_and_zero_anchor_banks_are_kept_not_repaired(self):
        for n in (0,5,16):
            with tempfile.TemporaryDirectory() as temp,self.subTest(n=n):
                dataset,cache,_=self.fixture(Path(temp));value=arrays(n,min(4,n))
                metadata=self.build(dataset,cache,value);loaded=banks.load_bank(dataset,cache,20)
                self.assertEqual(metadata['status'],'limited');self.assertEqual(metadata['anchors'],n)
                self.assertEqual(len(loaded.arrays[0]),n)

    def test_builder_failure_is_immutable_and_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,cache,_=self.fixture(Path(temp))
            def fail(*args):raise RuntimeError('exact builder failure')
            with patch.object(banks,'production_api',return_value=(fail,lambda *a:(SimpleNamespace(original=object()),None,np.eye(4)))):
                metadata=banks.build_one(dataset,cache,20)
            self.assertEqual(metadata['status'],'failed');self.assertIn('exact builder failure',metadata['failure'])
            with patch.object(banks,'production_api',side_effect=AssertionError('must not retry')):
                reused=banks.build_one(dataset,cache,20)
            self.assertEqual(reused,metadata);self.assertIsNone(banks.load_bank(dataset,cache,20).arrays)

    def test_input_manifest_source_and_runtime_mutations_are_rejected(self):
        for kind in ('manifest','source','runtime','payload'):
            with tempfile.TemporaryDirectory() as temp,self.subTest(kind=kind):
                dataset,cache,_=self.fixture(Path(temp));self.build(dataset,cache,arrays())
                with ExitStack() as stack:
                    if kind=='manifest':(dataset/'inference.json').write_text((dataset/'inference.json').read_text()+' ')
                    elif kind=='source':stack.enter_context(patch.object(banks,'source_hashes',return_value={'changed':'hash'}))
                    elif kind=='runtime':stack.enter_context(patch.object(banks,'package_versions',return_value={'changed':'version'}))
                    else:(dataset/'rgb.bin').write_bytes(b'changed')
                    with self.assertRaises(banks.BankCacheError):banks.load_bank(dataset,cache,20)

    def test_npz_or_array_metadata_tampering_rejected(self):
        for kind in ('npz','arrays','status'):
            with tempfile.TemporaryDirectory() as temp,self.subTest(kind=kind):
                dataset,cache,_=self.fixture(Path(temp));self.build(dataset,cache,arrays())
                directory=banks.folder(cache,20)
                if kind=='npz':
                    with (directory/'reference-bank.npz').open('ab') as f:f.write(b'tampered')
                else:
                    p=directory/'metadata.json';m=banks.read(p)
                    if kind=='arrays':m['arrays']['referencePoints']['sha256']='0'*64
                    else:m['status']='ready' # seed has4: truly limited.
                    m['metadataSHA256']=banks.value_sha({k:v for k,v in m.items() if k!='metadataSHA256'});p.write_bytes(banks.canonical(m))
                with self.assertRaises(banks.BankCacheError):banks.load_bank(dataset,cache,20)

    def test_object_pickle_and_huge_npy_header_are_refused_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'unsafe.npz';value=list(arrays());value[0]=np.array([object()],dtype=object)
            np.savez_compressed(p,**dict(zip(banks.ARRAY_NAMES,value)))
            with self.assertRaises(banks.BankCacheError):banks.read_npz(p)
            header=io.BytesIO();np.lib.format.write_array_header_1_0(header,dict(descr='<f8',fortran_order=False,shape=(10**12,3)))
            with zipfile.ZipFile(p,'w') as z:
                for name in banks.ARRAY_NAMES:z.writestr(name+'.npy',header.getvalue())
            with self.assertRaises(banks.BankCacheError):banks.read_npz(p)

    def test_missing_or_duplicate_onboarding_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,cache,_=self.fixture(Path(temp));p=dataset/'inference.json';m=json.loads(p.read_text());m['objects'][0]['onboarding'].pop();p.write_text(json.dumps(m))
            with self.assertRaises(banks.BankCacheError):banks.binding(dataset,20)
            outside=Path(temp)/'outside';outside.write_bytes(b'outside')
            with self.assertRaises(banks.BankCacheError):banks.resource_path(dataset,dict(path='../outside',sha256=banks.sha(outside)))

    def ablation(self,dataset,root,value,*,mode='production_mesh'):
        run=root/'ablation';run.mkdir();bound,_=banks.binding(dataset,20)
        directory=run/'object-000020';directory.mkdir();path=directory/'reference-bank.npz'
        np.savez_compressed(path,**dict(zip(banks.ARRAY_NAMES,value)))
        versions=bound['packageVersions'];sources=dict(bound['sourceHashes'],**{'bench/tracking_benchmark.py':banks.ABLATION_HARNESS_SHA256})
        freeze=dict(sourceHashes=sources,hardware=dict(numpy=versions['numpy'],opencv=versions['opencv'],pillow=versions['pillow'],python=versions['python']+' runtime'))
        (run/'freeze.json').write_text(json.dumps(freeze))
        seq=dict(objectId=20,anchors=len(value[0]),seedViewAnchors=len(value[3]),initializationSeconds=12.5,
                 bankSHA256=hashlib.sha256(b''.join(a.tobytes() for a in value)).hexdigest(),
                 bankArtifact=dict(path='object-000020/reference-bank.npz',sha256=banks.sha(path)))
        seal=dict(freezeSHA256=banks.sha(run/'freeze.json'),sourceHashes=sources,inferenceSHA256=bound['inferenceManifestSHA256'],configuration=dict(initializationMode=mode),sequences=[seq])
        (run/'inference-seal.json').write_text(json.dumps(seal));return run

    def test_only_sealed_exact_production_ablation_can_be_adopted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,cache,_=self.fixture(root);value=arrays(20,12);run=self.ablation(dataset,root,value)
            entry=banks.adopt_ablation(dataset,cache,20,run)
            self.assertEqual(entry.metadata['origin']['kind'],'adopted_sealed_ablation')
            self.assertEqual(entry.metadata['sourceBuildSeconds'],12.5)
            self.assertEqual(entry.metadata['artifact']['sha256'],banks.sha(run/'object-000020/reference-bank.npz'))
            loaded=banks.load_bank(dataset,cache,20)
            for a,b in zip(value,loaded.arrays):np.testing.assert_array_equal(a,b)

    def test_depth_or_unknown_source_ablation_is_not_silently_recomputed(self):
        for kind in ('depth','source','payload'):
            with tempfile.TemporaryDirectory() as temp,self.subTest(kind=kind):
                root=Path(temp);dataset,cache,_=self.fixture(root);run=self.ablation(dataset,root,arrays(20,12),mode='depth_budget' if kind=='depth' else 'production_mesh')
                if kind=='source':
                    p=run/'inference-seal.json';m=banks.read(p);m['sourceHashes']['bench/tracking_benchmark.py']='0'*64;p.write_text(json.dumps(m))
                if kind=='payload':
                    with (run/'object-000020/reference-bank.npz').open('ab') as f:f.write(b'bad')
                with self.assertRaises(banks.BankCacheError):banks.adopt_ablation(dataset,cache,20,run)

    def test_invalid_parallel_resource_bounds_rejected_without_starting_workers(self):
        for options in (dict(workers=0),dict(workers=5),dict(workers=4,max_worker_rss=2*1024**3),dict(max_seconds=7201)):
            with self.subTest(options=options),self.assertRaises(banks.BankCacheError):
                banks.prepare_banks('not-read','not-created',[20],**options)

    def test_disk_reserve_guard_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(banks,'MIN_FREE_BYTES',10*1024**3):
            with patch.object(banks.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(banks.BankCacheError,'disk reserve'):
                    banks.disk_guard(temp)

    def test_two_fresh_preparation_workers_preserve_tiny_invalid_mesh_failures(self):
        # The actual pool/supervisor runs, but both tiny fixtures fail the PLY
        # header before rasterization or RGB decode. No heavy bank is generated.
        parent=Path(__file__).resolve().parents[1]/'.cache'/'tracking-benchmark'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            dataset,cache,_=self.fixture(Path(temp));p=dataset/'inference.json';m=json.loads(p.read_text())
            other=json.loads(json.dumps(m['objects'][0]));other['objectId']=21;m['objects'].append(other);p.write_text(json.dumps(m))
            report=banks.prepare_banks(dataset,cache,[20,21],workers=2,max_worker_rss=512*1024**2,
                                      max_total_rss=1024**3,max_seconds=30)
            self.assertIsNone(report['reason']);self.assertEqual(report['trackingUpdates'],0)
            self.assertEqual(report['workers'],2)
            self.assertEqual([row['status'] for row in report['entries']],['failed','failed'])
            for oid in (20,21):
                entry=banks.load_bank(dataset,cache,oid)
                self.assertIsNone(entry.arrays);self.assertIn('ASCII BOP PLY',entry.metadata['failure'])

    def test_rss_guard_stops_owned_pool_and_retains_unfinished_entry(self):
        parent=Path(__file__).resolve().parents[1]/'.cache'/'tracking-benchmark'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            dataset,cache,_=self.fixture(Path(temp))
            with patch.object(banks,'_rss',return_value=2*1024**3):
                report=banks.prepare_banks(dataset,cache,[20],workers=1,max_worker_rss=512*1024**2,
                                          max_total_rss=1024**3,max_seconds=30)
            self.assertIn(report['reason'],('per-worker RSS limit exceeded','total preparation RSS limit exceeded'))
            self.assertEqual(report['entries'][0]['status'],'failed')
            self.assertIsNone(banks.load_bank(dataset,cache,20).arrays)


if __name__=='__main__':unittest.main()
