"""Synthetic metadata fixtures; no models, arrays, cloud or training."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import kaggle_label_order_v1 as adapter
from scripts.kaggle_label_order_v1 import ordered_artifacts
from bench import quality_adaptation_portable_prepare as p


class LabelOrderTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'global_manifest': {'records': [dict(id='a', split='train'),
            dict(id='b', split='validation'), dict(id='c', split='test')]}}
        self.inventory = {'rows': [dict(id='b', status='available', labels=dict(path='b.npz', bytes=2, sha256='b'*64)),
            dict(id='a', status='available', labels=dict(path='a.npz', bytes=1, sha256='a'*64)),
            dict(id='c', status='unavailable', failure_reason='sealed')]}
        self.artifacts = [dict(id=r['id'], **r['labels']) for r in self.inventory['rows'] if r['status']=='available']

    def test_changes_only_order_to_execution_order(self):
        original = copy.deepcopy(self.artifacts)
        result = ordered_artifacts(self.plan, self.inventory, self.artifacts)
        self.assertEqual([r['id'] for r in result], ['a', 'b'])
        self.assertEqual({r['id']:r for r in result}, {r['id']:r for r in original})
        self.assertEqual(self.artifacts, original)

    def test_canonical_order_is_idempotent(self):
        expected = list(reversed(self.artifacts))
        self.assertEqual(ordered_artifacts(self.plan, self.inventory, expected), expected)

    def test_payload_difference_rejected(self):
        self.artifacts[0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'beyond ordering'):
            ordered_artifacts(self.plan, self.inventory, self.artifacts)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            ordered_artifacts(self.plan, self.inventory, self.artifacts+self.artifacts[:1])

    def test_missing_artifact_rejected(self):
        with self.assertRaisesRegex(ValueError, 'beyond ordering'):
            ordered_artifacts(self.plan, self.inventory, self.artifacts[:1])

    def test_sealed_test_payload_rejected(self):
        self.inventory['rows'][-1].update(status='available', labels=self.inventory['rows'][0]['labels'])
        with self.assertRaisesRegex(ValueError, 'sealed'):
            ordered_artifacts(self.plan, self.inventory, self.artifacts)

    def test_publication_preserves_original_receipt_and_payloads(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.cache') as temporary, ExitStack() as stack:
            root=Path(temporary); features=root/'features'; execution=root/'identity'
            features.mkdir(); execution.mkdir()
            for name,value in [('prelabel-seal.json',{}),('execution-plan.json',self.plan)]:
                p.atomic_json(execution/name,value)
            inventory=copy.deepcopy(self.inventory)
            for row in inventory['rows']:
                if row['status']=='available':
                    file=features/row['labels']['path'];file.write_bytes(row['id'].encode())
                    row['labels']={**p.pin(file),'path':file.name}
            p.atomic_json(features/'labels.json',inventory);p.atomic_json(features/'manifest.json',{})
            artifacts=[dict(id=r['id'],**r['labels']) for r in inventory['rows'] if r['status']=='available']
            receipt=dict(schema='kaggle_label_receipt_v2',prelabel_seal=p.pin(execution/'prelabel-seal.json'),
                inventory=p.pin(features/'labels.json'),dataset=p.pin(features/'manifest.json'),
                labels_entered_inference=False,test_payload_reads=0,label_artifacts=artifacts)
            p.atomic_json(features/'label-receipt.json',receipt)
            before=(features/'label-receipt.json').read_bytes()
            for owner,name in [(p,'cloud_guard'),(adapter.bridge,'verify_binding'),(adapter.raw,'read_context'),
                    (p,'raw_metadata'),(p,'verify_prelabel')]:
                stack.enter_context(patch.object(owner,name))
            request=dict(features_root=str(features),execution_root=str(execution),
                context=dict(path=str(root/'context.json'),sha256='0'*64),raw_seal={})
            digest=p.sha256(Path(adapter.__file__))
            result=adapter.normalize(request,expected_source_sha256=digest)
            self.assertEqual(result['status'],'order_corrected')
            self.assertEqual((features/'label-receipt-original-capture-order.json').read_bytes(),before)
            corrected=p.read_json(features/'label-receipt.json')
            self.assertEqual([r['id'] for r in corrected['label_artifacts']],['a','b'])
            self.assertEqual((features/'a.npz').read_bytes(),b'a')
            self.assertEqual((features/'b.npz').read_bytes(),b'b')
            self.assertEqual(adapter.normalize(request,expected_source_sha256=digest)['status'],'already_ordered')
            with self.assertRaisesRegex(ValueError,'source changed'):
                adapter.normalize(request,expected_source_sha256='0'*64)


if __name__ == '__main__':
    unittest.main()
