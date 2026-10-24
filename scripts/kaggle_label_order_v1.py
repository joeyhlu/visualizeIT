"""Order-only compatibility for genuine multi-object label receipts.

The original consumer compares ordered artifact lists. Capture order can differ
from its frozen execution order. Preserve the original receipt and every payload;
reject any discrepancy beyond ordering before publishing a corrected receipt.
"""
from pathlib import Path
import os

from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_supervised_v1 as bridge
from bench import quality_adaptation_portable as raw


def ordered_artifacts(plan, inventory, artifacts):
    states = p.indexed(plan['global_manifest']['records'], 'execution states')
    labels = p.indexed(inventory['rows'], 'label inventory')
    supplied = p.indexed(artifacts, 'label artifacts')
    expected = []
    for ident, state in states.items():
        if state['split'] == 'dev':
            continue
        label = labels[ident]
        if label['status'] == 'available':
            if state['split'] not in ('train', 'validation'):
                raise ValueError('sealed/non-training state has a label payload')
            expected.append(dict(id=ident, **label['labels']))
    if supplied != p.indexed(expected, 'expected label artifacts'):
        raise ValueError('label artifacts differ beyond ordering')
    return expected


def normalize(request, *, expected_source_sha256):
    """Authenticate existing evidence, then change only its list order."""
    if p.sha256(Path(__file__)) != expected_source_sha256:
        raise ValueError('label ordering adapter source changed')
    p.cloud_guard(request)
    bridge.verify_binding(request, request['features_root'])
    context = raw.read_context(Path(request['context']['path']), request['context']['sha256'])
    state = p.raw_metadata(context, request['raw_seal'])
    execution, features = Path(request['execution_root']), Path(request['features_root'])
    p.verify_prelabel(state, execution, features)
    path = features / 'label-receipt.json'
    receipt = p.read_json(path)
    if (receipt['schema'] != 'kaggle_label_receipt_v2'
            or receipt['prelabel_seal'] != p.pin(execution / 'prelabel-seal.json')
            or receipt['inventory'] != p.pin(features / 'labels.json')
            or receipt['dataset'] != p.pin(features / 'manifest.json')
            or receipt.get('labels_entered_inference') is not False
            or receipt.get('test_payload_reads') != 0):
        raise ValueError('original label receipt bindings differ')
    inventory = p.metadata(receipt['inventory'])
    plan = p.read_json(execution / 'execution-plan.json')
    ordered = ordered_artifacts(plan, inventory, receipt['label_artifacts'])
    if ordered == receipt['label_artifacts']:
        return dict(status='already_ordered', receipt=p.pin(path))
    for artifact in ordered:
        p.verify_pin({k: artifact[k] for k in ('path', 'bytes', 'sha256')}, features)
    backup = features / 'label-receipt-original-capture-order.json'
    with backup.open('xb') as stream:
        stream.write(path.read_bytes())
    corrected = {**receipt, 'label_artifacts': ordered,
        'artifact_order_correction': dict(schema='kaggle_label_order_correction_v1',
            source=p.pin(Path(__file__)), original_receipt=p.pin(backup),
            execution_plan=p.pin(execution / 'execution-plan.json'), inventory=receipt['inventory'],
            correction='same exact artifacts in frozen execution order')}
    temporary = features / 'label-receipt-canonical-order.pending.json'
    p.atomic_json(temporary, corrected)
    os.replace(temporary, path)
    return dict(status='order_corrected', receipt=p.pin(path), original_receipt=p.pin(backup),
        artifacts=len(ordered))
