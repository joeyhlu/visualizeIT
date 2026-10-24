"""Explicit SOFTWARE metadata/assets only; never models, datasets or quality evidence."""
from contextlib import ExitStack
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_sharded_v1 as shards
from bench.test_quality_adaptation_portable_prepare import known_role_fixture,write

ROOT=Path(__file__).resolve().parents[1]


class SoftwarePresence:
    def __init__(self,base):
        self.base=Path(base); self.base.mkdir(parents=True,exist_ok=True)
        self.recipe,self.audit=known_role_fixture(self.base)
        self.role_file=Path(self.audit['evidence_descriptors'][0]['path'])
        self.missing=dict(path=(self.base/'absent-historical-admission.json').as_posix(),bytes=5895,sha256='c'*64)
        self.audit['evidence_descriptors'].append(self.missing)
        for asset in self.recipe['assets']:
            if asset['object_id']=='7':
                asset['mesh']['path']=(self.base/'SEALED-TEST7-NEVER-STAT.glb').as_posix()
            else:
                path=self.base/('software-mesh-'+asset['object_id']+'.glb')
                path.write_bytes(('SOFTWARE MANUFACTURED MESH BYTES '+asset['object_id']).encode())
                asset['mesh']=p.pin(path)
            row=next(row for row in self.audit['assets'] if row['physical_object_id']==asset['physical_object_id'])
            row['mesh_sha256']=asset['mesh']['sha256']
        self.audit['recipe_sha256']=p.canonical(self.recipe)
        children={}
        for sid,group in shards.TRAIN_GROUPS.items():
            recipe=shards._projected_recipe(self.recipe,group)
            children[sid]=dict(recipe=recipe,exposure_audit=shards._projected_audit(self.audit,recipe))
        body=dict(schema=shards.PARTITION_SCHEMA,full_recipe=self.recipe,full_recipe_sha256=p.canonical(self.recipe),
            exposure_audit=self.audit,exposure_audit_sha256=p.canonical(self.audit),children=children)
        self.partition=write(self.base/'partition.json',dict(body,partition_sha256=p.canonical(body)))
        imported=copy.deepcopy(children['smoke0']['exposure_audit']);imported['SOFTWARE_IMPORTED_AUDIT']=True
        self.additional=[dict(recipe=write(self.base/'imported-recipe.json',children['smoke0']['recipe']),
            exposure_audit=write(self.base/'imported-audit.json',imported))]
        self.snapshots=[write(self.base/f'software-snapshot-{i}.json',dict(SOFTWARE_FIXTURE=True,variant=i)) for i in (0,1)]
        self.source=self.base/'source'
        helper=self.source/'bench/quality_adaptation_portable_prepare.py';helper.parent.mkdir(parents=True)
        shutil.copyfile(p.__file__,helper)
        self.models={}
        for name in p.ASSET_PRESENCE_MODELS:
            output=self.source/'.cache/model-quality/checkpoints'/name;output.parent.mkdir(parents=True,exist_ok=True)
            data=('SOFTWARE NEVER LOAD AS CHECKPOINT '+name).encode();output.write_bytes(data)
            self.models[name]=(len(data),hashlib.sha256(data).hexdigest())
        self.runtime=write(self.base/'runtime.json',dict(SOFTWARE_RUNTIME_METADATA=True))
        self.sources={'bench/quality_adaptation_portable_prepare.py':p.pin(helper)}
        self.migration=write(self.base/'migration.json',dict(schema='visualizeit_prepared_source_migration_v1',sources={},contexts={}))
        producer=self.base/'software-producer.py';producer.write_text('# SOFTWARE NEVER EXECUTED\n')
        self.producer=p.pin(producer)
        self.review=write(self.base/'review.json',dict(decision='APPROVE',training_approved=False,
            operator_sha256=self.producer['sha256'],source_migration=self.migration,
            files={name:value['sha256'] for name,value in self.sources.items()}))

    def constants(self):
        return dict(MISSING_ASSET_ADMISSION=self.missing,ASSET_PRESENCE_MODELS=self.models,ASSET_PRESENCE_SNAPSHOTS=self.snapshots)

    def make(self):
        proof=p.current_asset_presence_proof(self.partition,additional_audits=self.additional,
            source_root=self.source,runtime=self.runtime,executing_sources=self.sources)
        self.proof=write(self.base/'proof.json',proof)
        doc=p.asset_presence_binding_document(self.partition,additional_audits=self.additional,proof=self.proof,
            producer=self.producer,review=self.review,source_migration=self.migration)
        self.binding=write(self.base/'binding.json',doc)
        return self.binding


class AssetPresenceSoftwareTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'.cache/cloud-prepare/test-tmp';root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.f=SoftwarePresence(Path(self.temp.name))
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(p,**self.f.constants()))
        self.stack.enter_context(p.source_migration_scope(self.f.migration))
        self.f.make()

    def test_default_absent_fails_and_scoped_exact_success_preserves_all_metadata(self):
        before={path:path.read_bytes() for path in self.f.base.rglob('*.json')}
        with self.assertRaises(ValueError):p.bounded_recipe(self.f.recipe,self.f.audit)
        with self.assertRaises(ValueError):p.metadata(self.f.missing)
        with p.asset_presence_successor_scope(self.f.binding):
            result=p.bounded_recipe(self.f.recipe,self.f.audit)
            self.assertEqual(result['plan']['counts']['train']['refinement_states'],312)
            self.assertEqual(result['audit_sha256'],p.canonical(self.f.audit))
            with p.asset_presence_successor_scope(self.f.binding):p.bounded_recipe(self.f.recipe,self.f.audit)
            self.assertIsNotNone(p._ASSET_PRESENCE_SUCCESSOR.get())
        self.assertIsNone(p._ASSET_PRESENCE_SUCCESSOR.get())
        self.assertEqual(before,{path:path.read_bytes() for path in before})

    def test_wrong_audit_or_descriptor_not_authorized(self):
        with p.asset_presence_successor_scope(self.f.binding):
            altered=copy.deepcopy(self.f.audit);altered['SOFTWARE_MUTATION']=True
            with self.assertRaisesRegex(ValueError,'outside explicit'):p.bounded_recipe(self.f.recipe,altered)
            altered=copy.deepcopy(self.f.audit);altered['evidence_descriptors'][-1]['sha256']='d'*64
            with self.assertRaises(ValueError):p.bounded_recipe(self.f.recipe,altered)

    def test_present_corrupt_old_receipt_rejects_before_any_asset_open(self):
        Path(self.f.missing['path']).write_text('CORRUPT SOFTWARE RECEIPT')
        with patch.object(p,'_asset_presence_file') as hashed:
            with self.assertRaisesRegex(ValueError,'requires historical admission to be absent'):
                with p.asset_presence_successor_scope(self.f.binding):pass
            hashed.assert_not_called()

    @unittest.skipIf(os.name=='nt','mandatory Linux no-follow symlink refusal')
    def test_present_symlink_old_receipt_rejected(self):
        Path(self.f.missing['path']).symlink_to(self.f.role_file)
        with self.assertRaisesRegex(ValueError,'symlink'):
            with p.asset_presence_successor_scope(self.f.binding):pass

    def test_missing_historical_role_or_snapshot_rejects_before_assets(self):
        for path in (self.f.role_file,Path(self.f.snapshots[0]['path'])):
            with self.subTest(path=path):
                saved=path.read_bytes();path.unlink()
                try:
                    with patch.object(p,'_asset_presence_file') as hashed,self.assertRaises((ValueError,FileNotFoundError)):
                        with p.asset_presence_successor_scope(self.f.binding):pass
                    hashed.assert_not_called()
                finally:path.write_bytes(saved)

    def test_changed_asset_rejected_and_TEST_path_not_statted(self):
        mesh=next(a['mesh'] for a in self.f.recipe['assets'] if a['object_id']=='1')
        Path(mesh['path']).write_bytes(b'CHANGED SOFTWARE MESH')
        with self.assertRaisesRegex(ValueError,'current asset bytes changed'):
            with p.asset_presence_successor_scope(self.f.binding):pass

    def test_sealed_TEST_never_statted_and_explicit_TEST_proof_rejected(self):
        original=Path.stat
        def checked(path,*args,**kwargs):
            if 'SEALED-TEST7' in str(path):raise AssertionError('sealed TEST stat')
            return original(path,*args,**kwargs)
        with patch.object(Path,'stat',checked),p.asset_presence_successor_scope(self.f.binding):
            p.bounded_recipe(self.f.recipe,self.f.audit)
        doc=p.metadata(self.f.proof);doc['meshes'].append(dict(object_id='7',mesh={'path':'POISON'}))
        changed=write(self.f.base/'test-request-proof.json',doc)
        binding=p.metadata(self.f.binding);binding['proof']=changed
        with self.assertRaisesRegex(ValueError,'exact current'):
            with p.asset_presence_successor_scope(write(self.f.base/'test-request-binding.json',binding)):pass

    def test_stale_proof_partition_and_source_mutations_reject(self):
        proof=p.metadata(self.f.proof);proof['observed_at']=(datetime.now(timezone.utc)-timedelta(minutes=6)).isoformat()
        binding=p.metadata(self.f.binding);binding['proof']=write(self.f.base/'stale-proof.json',proof)
        with self.assertRaisesRegex(ValueError,'fresh current'):
            with p.asset_presence_successor_scope(write(self.f.base/'stale-binding.json',binding)):pass
        changed=p.metadata(self.f.binding);changed['partition']['sha256']='a'*64
        with self.assertRaises(ValueError):
            with p.asset_presence_successor_scope(write(self.f.base/'partition-binding.json',changed)):pass
        Path(self.f.sources['bench/quality_adaptation_portable_prepare.py']['path']).write_text('# CHANGED')
        with self.assertRaises(ValueError):
            with p.asset_presence_successor_scope(self.f.binding):pass

    def test_fresh_interpreter_reconstructs_scope_and_missing_binding_fails(self):
        config=write(self.f.base/'fresh-config.json',dict(constants=self.f.constants(),binding=self.f.binding,
            migration=self.f.migration,recipe=self.f.recipe,audit=self.f.audit))
        program=r'''
from pathlib import Path
import json,sys
from unittest.mock import patch
from bench import quality_adaptation_portable_prepare as p
c=json.loads(Path(sys.argv[1]).read_text())
with patch.multiple(p,**c['constants']),p.source_migration_scope(c['migration']):
    try:p.bounded_recipe(c['recipe'],c['audit'])
    except ValueError:pass
    else:raise AssertionError('fresh worker inherited implicit admission')
    with p.asset_presence_successor_scope(c['binding']):
        result=p.bounded_recipe(c['recipe'],c['audit'])
        assert result['plan']['counts']['train']['refinement_states']==312
    assert p._ASSET_PRESENCE_SUCCESSOR.get() is None and 'torch' not in sys.modules
print('SOFTWARE FRESH WORKER PASS')
'''
        result=subprocess.run([sys.executable,'-E','-s','-B','-c',program,config['path']],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('SOFTWARE FRESH WORKER PASS',result.stdout)


if __name__=='__main__':unittest.main()
