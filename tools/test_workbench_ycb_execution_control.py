"""Independent failure controls for execution-only CAD evidence (no GPU)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import workbench_ycb_execution_control as adapter

class ExecutionControlTests(unittest.TestCase):
    def test_missing_approval_cannot_emit(self):
        with patch.object(adapter,'load_map',side_effect=adapter.AdapterError('missing reviewed map')):
            with patch.object(adapter,'plan') as planner:
                with self.assertRaises(adapter.AdapterError):
                    adapter.emit(2,adapter.DERIVED_ROOT/'unapproved-test-output')
                planner.assert_not_called()

    def test_output_escape_rejected(self):
        with self.assertRaises(adapter.AdapterError):adapter.check_out(adapter.ROOT/'escaped-control')

    def test_missing_planned_bundle_pin_rejected(self):
        entry={key:'0'*64 for key in adapter.ENTRY_KEYS}
        entry.update(object_id=2,canonical_ycb_id=adapter.yc.OBJECTS[2][0])
        entry.pop('bundle_sha256')
        with self.assertRaises(adapter.AdapterError):adapter.map_entry({'entries':{'2':entry}},2)

    def fixture(self,directory):
        record={key:None for key in adapter.RECORD_KEYS}
        record['schema']=adapter.gpu.SCHEMA
        names=adapter.evidence_names(2)
        record['files']={name:adapter.sha256_bytes(name.encode()) for name in names}
        for name in names:(directory/name).write_bytes(name.encode())
        raw=adapter.canonical(record);(directory/adapter.EVIDENCE_RECORD).write_bytes(raw)
        return record,adapter.sha256_bytes(raw)

    def test_exact_artifact_inventory_and_hashes_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);record,digest=self.fixture(directory)
            loaded,artifacts=adapter.load_evidence(directory,digest,2)
            self.assertEqual(set(artifacts),set(record['files']))
            target=next(iter(record['files']))
            (directory/target).write_bytes(b'mutation')
            with self.assertRaises(adapter.AdapterError):adapter.load_evidence(directory,digest,2)

    def test_extra_failure_receipt_blocks_valid_looking_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);_,digest=self.fixture(directory)
            (directory/'failure.json').write_text('{}')
            with self.assertRaises(adapter.AdapterError):adapter.load_evidence(directory,digest,2)

    def test_record_rewrite_rejected_by_external_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);record,digest=self.fixture(directory)
            record['relative_serving_uv_convention_pass']=True
            (directory/adapter.EVIDENCE_RECORD).write_bytes(adapter.canonical(record))
            with self.assertRaises(adapter.AdapterError):adapter.load_evidence(directory,digest,2)

if __name__=='__main__':unittest.main()
