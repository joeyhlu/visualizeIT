"""Four synchronized views with honest pending states and independent markers."""
import hashlib
import json
import math
import base64
from pathlib import Path
import xml.etree.ElementTree as ET
from .quality_assets import ROOT, CACHE
from .show3d_player import HTML
from .recovery_player import replace_once
from .quality_evaluate import polygon_mask
from .vision import cv2
from .storage import write_artifact


MUG_PNP_EXPERIMENT = CACHE/'diagnostics/mug-pnp-full-window-v2'
MUG_PNP_FRAME_IDS = tuple(range(827, 1067))
MUG_PNP_BRANCHES = ('control', 'candidate')
MUG_PNP_PREFIX_LENGTHS = (30, 120, 240)
MUG_PNP_STAGE_IDS = tuple(f'{branch}-{length}' for branch in MUG_PNP_BRANCHES
                          for length in MUG_PNP_PREFIX_LENGTHS)
MUG_PNP_PREFIX_KEYS = tuple(f'{branch}_{short}_vs_{long}' for branch in MUG_PNP_BRANCHES
                            for short, long in ((30, 120), (30, 240), (120, 240)))
MUG_PNP_MODES = ('mug_pnp_control', 'mug_pnp_no_extrinsic_guess')
MUG_PNP_MODE_LABELS = {
    'mug_pnp_control': 'Full-window PnP control',
    'mug_pnp_no_extrinsic_guess': 'No-extrinsic-guess ablation',
}
MUG_PNP_FIXED_SETTINGS = {
    'model_memory': False,
    'unlit_templates': True,
    'appearance_check': False,
    'disable_multisampling': True,
}

BOTTLE_POSE_ABLATION_REPORT = CACHE/'diagnostics/bottle-source-observability-pose-ablation-v1/pose_ablation.json'
BOTTLE_POSE_ABLATION_TERMINAL = CACHE/'diagnostics/bottle-r5-parent-v1/terminal.json'
BOTTLE_POSE_ABLATION_FIGURE = ROOT/'.cache/ui-proof/bottle-source-selection-r5.png'
BOTTLE_POSE_ABLATION_REPORT_SHA256 = 'A394BF54F51DDF527549EE5B3B6C7E8F623546B2C277A49051DF2CE1C4D95259'
BOTTLE_POSE_ABLATION_TERMINAL_SHA256 = '7D193B09D6D9DDA241E039FCE757B0A19D5C82E0FC6BC84B287D73E8915392AE'
BOTTLE_POSE_ABLATION_FIGURE_SHA256 = '96613B55D9D42144D17189A6597288CAD77DBC6552D9CF06DDAB2D2AAD7243F4'
BOTTLE_POSE_ABLATION_REPORT_BYTES = 872845
BOTTLE_POSE_ABLATION_TERMINAL_BYTES = 33103
BOTTLE_POSE_ABLATION_FIGURE_MAX_BYTES = 2*1024*1024
BOTTLE_POSE_ABLATION_WORKING_BUDGET = 128*1024*1024
BOTTLE_POSE_ABLATION_OUTPUT_BUDGET = 8*1024*1024
BOTTLE_POSE_ABLATION_CONDITION_IDS = (
    'syn-10-q8-rgb-t0-rgb', 'zero-10-full', 'zero-10-clipped', 'syn-10-q8-rgb-t180-rgb',
    'syn-50-q8-rgb-t0-rgb', 'zero-50-full', 'zero-50-clipped', 'syn-50-q8-rgb-t180-rgb',
    'syn-100-q8-rgb-t0-rgb', 'zero-100-full', 'zero-100-clipped', 'syn-100-q8-rgb-t180-rgb',
)
BOTTLE_POSE_ABLATION_SPEC_SHA256 = '4BB91EB3782EF689B97B14F0CAE68860768B39348526CED21AF25EC5CFB78340'
BOTTLE_POSE_ABLATION_SOURCE_SHA256 = '54BDD2F4CA4FB82551492E5397AB42CEA2E464C5DAD1F83360F7B23D33926542'
BOTTLE_POSE_ABLATION_TEST_SHA256 = '4FAF00469E154FE56E9352CCA375674A848D6E21104340D52CD51D7AC685CC91'

# Hash-bound R8 actual-source and terminal full-screen receipts. The page only
# reads these four fixed JSON files; it never exposes their cache paths.
BOTTLE_R8_ACTUAL_WORKER = CACHE/'diagnostics/bottle-r8-actual-preliminary-v1/worker.json'
BOTTLE_R8_ACTUAL_TERMINAL = CACHE/'diagnostics/bottle-r8-actual-preliminary-v1/terminal.json'
BOTTLE_R8_FULL28_CONTRACT = CACHE/'diagnostics/bottle-r8-full-capacity-v1/contract.json'
BOTTLE_R8_FULL28_TERMINAL = CACHE/'diagnostics/bottle-r8-full-capacity-v1/terminal.json'
BOTTLE_R8_ACTUAL_WORKER_SHA256 = '41431FEADA197015D30FAD3E91E3E1981139976FEFDE2D00DCB1220235EB212B'
BOTTLE_R8_ACTUAL_TERMINAL_SHA256 = '3812F82880470525692B706C01AAC0BCA2EDDB2C6D880F8952D5B04BC040F322'
BOTTLE_R8_FULL28_CONTRACT_SHA256 = '07150959FAEB28963FB1A8924065A71AF6B83CA3BB047E27BDCFEF210D373021'
BOTTLE_R8_FULL28_TERMINAL_SHA256 = '34F72068D97560B4FB152F41F48D8317D9DA272BC558E5F16706D457DA9549CB'
BOTTLE_R8_ACTUAL_WORKER_BYTES = 190419
BOTTLE_R8_ACTUAL_TERMINAL_BYTES = 41801
BOTTLE_R8_FULL28_CONTRACT_BYTES = 124587
BOTTLE_R8_FULL28_TERMINAL_BYTES = 4353
BOTTLE_R8_SOURCE_CONTEXTS = (
    'template-0010-000', 'template-0010-180',
    'template-0050-000', 'template-0050-180',
    'template-0100-000', 'template-0100-180',
)
BOTTLE_R8_WITNESS_COUNTS = (1, 0, 5, 1, 7, 1)
BOTTLE_R8_FULL28_CASE_COUNT = 28
BOTTLE_R8_ACTUAL_ROW_COUNT = 12
BOTTLE_R8_MESH_PARITY_SUMMARY = CACHE/'diagnostics/bottle-r8-split-mesh-visual-v3/summary.json'
BOTTLE_R8_MESH_PARITY_SVG = CACHE/'diagnostics/bottle-r8-split-mesh-visual-v3/mesh-parity.svg'
BOTTLE_R8_MESH_PARITY_SUMMARY_SHA256 = '84B21E80E369E739A8149518ED2939739259FD7E7D8AF8D7AAB3438F70D438E3'
BOTTLE_R8_MESH_PARITY_SVG_SHA256 = '263BA2ED486EB7BA1018881AD268DE3DFEC639A8D563B4E8E914187E765CB7CE'
BOTTLE_R8_MESH_PARITY_SUMMARY_BYTES = 7354
BOTTLE_R8_MESH_PARITY_SVG_BYTES = 56604
BOTTLE_R8_MESH_PARITY_SUMMARY_MAX_BYTES = 16 * 1024
BOTTLE_R8_MESH_PARITY_SVG_MAX_BYTES = 64 * 1024
BOTTLE_R8_MESH_PARITY_CONTEXTS = (
    'template-0010-000', 'template-0010-180',
    'template-0050-000', 'template-0050-180',
    'template-0100-000', 'template-0100-180',
)
BOTTLE_R8_MESH_PARITY_START = '<!-- r8-mesh-parity-v3 -->'
BOTTLE_R8_MESH_PARITY_END = '<!-- /r8-mesh-parity-v3 -->'
BOTTLE_R8_MESH_PARITY_ELEMENT_ID = 'r8MeshParityDiagnostic'
BOTTLE_R8_MESH_PARITY_PROJECT_ROOT = '{http://www.w3.org/2000/svg}'

# Hash-bound closed calibration evidence pins.
BOTTLE_PATCH_CALIBRATION_REPORT = CACHE/'diagnostics/bottle-patch-pose-calibration-v2/calibration.json'
BOTTLE_PATCH_CALIBRATION_TERMINAL = CACHE/'diagnostics/bottle-r6-parent-v2/terminal.json'
BOTTLE_PATCH_CALIBRATION_REPEAT = CACHE/'diagnostics/bottle-r6-closed-repeat-v1.json'
BOTTLE_PATCH_CALIBRATION_REPORT_SHA256 = '857A52DF0982ADEE17388E3DE5CB82836EA2BBD0249EF878C4741CB909C341AD'
BOTTLE_PATCH_CALIBRATION_TERMINAL_SHA256 = '214ED8C051D7865D46BB254821A662DCB85B1A0E776D88ECC727E5DA46C97CCF'
BOTTLE_PATCH_CALIBRATION_REPEAT_SHA256 = 'A5AFF9D8A0C0DD2C1DAA0D8C0F11D8F5628281403B2E61109C15CB6970CE1D52'
BOTTLE_PATCH_CALIBRATION_SOURCE_SHA256 = 'AB9DCA72EE92DA7657BBCC95A869967326C6A381017DE02CB75F711BC6D503D1'
BOTTLE_PATCH_CALIBRATION_TEST_SHA256 = '81614CB691648571A4E02C4A2E1D4959EEC2896D16479D5E793C95A5782B92EF'
BOTTLE_PATCH_CALIBRATION_SPEC_SHA256 = '761C9A4400548A4F9BE1E3129E7B932BFB513960B6D1341D918C13F2B6621AAA'
BOTTLE_PATCH_CALIBRATION_R5_REPORT_SHA256 = 'A394BF54F51DDF527549EE5B3B6C7E8F623546B2C277A49051DF2CE1C4D95259'
BOTTLE_PATCH_CALIBRATION_R5_TERMINAL_SHA256 = '7D193B09D6D9DDA241E039FCE757B0A19D5C82E0FC6BC84B287D73E8915392AE'
BOTTLE_PATCH_CALIBRATION_CONDITION_IDS = (
    'syn-10-q8-rgb-t0-rgb', 'zero-10-full', 'zero-10-clipped',
    'syn-10-q8-rgb-t180-rgb', 'syn-50-q8-rgb-t0-rgb', 'zero-50-full',
    'zero-50-clipped', 'syn-50-q8-rgb-t180-rgb', 'syn-100-q8-rgb-t0-rgb',
    'zero-100-full', 'zero-100-clipped', 'syn-100-q8-rgb-t180-rgb',
)
BOTTLE_PATCH_CALIBRATION_POSITIVE_RESULTS = {
    'syn-10-q8-rgb-t0-rgb': (10, 99.22, 42.43, False),
    'syn-50-q8-rgb-t0-rgb': (50, 98.41, 65.44, True),
    'syn-100-q8-rgb-t0-rgb': (100, 99.19, 69.73, True),
}
# End of closed calibration evidence pins.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256_text(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _finite_nonnegative(value):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0)


def _valid_pose_matrix(pose):
    return (isinstance(pose, list) and len(pose) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in pose)
            and all(type(value) in (int, float) and math.isfinite(value)
                    for row in pose for value in row))


def _valid_area_metrics(metrics, processed, accepted):
    _require(isinstance(metrics, dict), 'area metrics are missing')
    _require(metrics.get('processed') == processed and metrics.get('accepted') == accepted,
             'area metrics do not match branch frame accounting')
    _require(metrics.get('source_frames') == processed, 'area metrics omit processed source frames')
    median, p95 = metrics.get('median_720'), metrics.get('p95_720')
    if accepted:
        _require(_finite_nonnegative(median) and _finite_nonnegative(p95),
                 'accepted area metrics are not finite')
    else:
        _require(median is None and p95 is None, 'empty area metrics must not contain an error')


def _validate_mug_pnp_report(report, experiment_root):
    """Check the saved evaluator contract and every stage's output/trace binding."""
    _require(isinstance(report, dict) and report.get('schema_version') == 1 and report.get('object') == 'mug',
             'unexpected mug PnP report schema')
    _require(report.get('requested_source_frames') == len(MUG_PNP_FRAME_IDS)
             and report.get('source_frame_ids') == list(MUG_PNP_FRAME_IDS),
             'report does not cover the exact mug source frames')
    _require(report.get('full_window_results_complete') is True
             and report.get('full_comparison_complete') is True,
             'report is not a terminal complete full-window comparison')
    _require(report.get('independent_accuracy_verified') is False
             and report.get('overall_gate_passed') is False
             and report.get('reference_or_annotation_inputs') is False
             and report.get('annotations_loaded') is False,
             'report contains an invalid independent-accuracy claim')

    lifecycle = report.get('terminal_stage_evidence')
    _require(isinstance(lifecycle, dict) and lifecycle.get('passed') is True
             and lifecycle.get('issues') == [], 'six-stage terminal lifecycle did not pass')
    stages = lifecycle.get('stages')
    _require(isinstance(stages, list) and len(stages) == len(MUG_PNP_STAGE_IDS)
             and all(isinstance(stage, dict) for stage in stages)
             and [stage.get('stage_id') for stage in stages if isinstance(stage, dict)] == list(MUG_PNP_STAGE_IDS),
             'report does not contain all six stages in exact order')
    snapshot_path = Path(experiment_root)/'experiment.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    _require(snapshot.get('object') == 'mug'
             and snapshot.get('setup_frame_id') == 826
             and snapshot.get('expected_frame_ids') == list(MUG_PNP_FRAME_IDS)
             and snapshot.get('run_mode') == 'complete'
             and snapshot.get('prefix_lengths') == list(MUG_PNP_PREFIX_LENGTHS),
             'immutable experiment manifest does not match the mug window')
    source_files = snapshot.get('source_files')
    frozen_files = snapshot.get('frozen_files')
    provenance = snapshot.get('inference_provenance')
    masks = snapshot.get('mask_sha256_by_frame')
    _require(isinstance(source_files, dict) and source_files
             and isinstance(frozen_files, dict) and frozen_files
             and isinstance(provenance, dict) and provenance,
             'immutable experiment bindings are missing')
    _require(all(_sha256_text(value) for value in source_files.values())
             and all(_sha256_text(value) for value in frozen_files.values()),
             'immutable source or frozen-file hashes are malformed')
    _require(isinstance(masks, dict)
             and all(_sha256_text(masks.get(str(frame_id))) for frame_id in MUG_PNP_FRAME_IDS),
             'frozen mask hashes do not cover the complete mug window')
    report_masks = report.get('frozen_snapshot_expected_mask_sha256')
    _require(isinstance(report_masks, dict)
             and all(report_masks.get(str(frame_id)) == masks[str(frame_id)] for frame_id in MUG_PNP_FRAME_IDS),
             'report and immutable manifest disagree on frozen masks')

    runs = {}
    for stage in stages:
        stage_id = stage['stage_id']
        branch, length_text = stage_id.rsplit('-', 1)
        length = int(length_text)
        requested_ids = list(MUG_PNP_FRAME_IDS[:length])
        _require(stage.get('status') == 'succeeded' and type(stage.get('exit_code')) is int
                 and stage.get('exit_code') == 0 and stage.get('terminal_exit_recorded') is True,
                 f'{stage_id} lacks terminal exit-zero evidence')
        _require(stage.get('output') == f'{stage_id}.json'
                 and stage.get('requested_frame_ids') == requested_ids
                 and stage.get('result_frame_ids') == requested_ids
                 and stage.get('expected_complete') is (length == 240)
                 and stage.get('result_complete') is (length == 240)
                 and stage.get('pnp_use_extrinsic_guess') is (branch == 'control'),
                 f'{stage_id} stage bindings do not match the exact prefix')
        _require(stage.get('source_files') == source_files
                 and stage.get('frozen_files') == frozen_files
                 and stage.get('inference_provenance') == provenance,
                 f'{stage_id} immutable bindings differ from the experiment manifest')
        _require(_sha256_text(stage.get('output_sha256')),
                 f'{stage_id} has no valid terminal output hash')
        trace_name = f'{stage_id}.trace.jsonl'
        trace_hash = stage.get('trace_sha256')
        trace_count = stage.get('trace_record_count')
        _require(stage.get('trace') == trace_name and _sha256_text(trace_hash)
                 and type(trace_count) is int and trace_count > 0,
                 f'{stage_id} has incomplete terminal trace bindings')

        output_path = Path(experiment_root)/stage['output']
        output_bytes = output_path.read_bytes()
        _require(hashlib.sha256(output_bytes).hexdigest() == stage['output_sha256'],
                 f'{stage_id} output bytes differ from terminal evidence')
        run = json.loads(output_bytes.decode('utf-8'))
        _require(isinstance(run, dict) and run.get('object') == 'mug'
                 and run.get('mode') == 'complete' and run.get('complete') is (length == 240)
                 and ('status' not in run or run.get('status') == 'complete')
                 and run.get('frame_ids') == requested_ids,
                 f'{stage_id} saved result is incomplete or has unexpected frame IDs')
        frames = run.get('frames')
        _require(isinstance(frames, list)
                 and [frame.get('frameId') for frame in frames if isinstance(frame, dict)] == requested_ids,
                 f'{stage_id} saved frames do not match its exact prefix')
        settings = run.get('tracking_settings')
        _require(isinstance(settings, dict)
                 and settings.get('pnp_use_extrinsic_guess') is (branch == 'control')
                 and all(settings.get(key) is expected for key, expected in MUG_PNP_FIXED_SETTINGS.items())
                 and stage.get('tracking_settings') == settings
                 and stage.get('runtime') == run.get('runtime'),
                 f'{stage_id} saved settings/runtime differ from terminal evidence')
        for frame in frames:
            fid = frame['frameId']
            _require(frame.get('mask_sha256') == masks[str(fid)],
                     f'{stage_id} frame {fid} is not bound to the frozen observed mask')
        trace_path = Path(experiment_root)/stage['trace']
        trace_bytes = trace_path.read_bytes()
        _require(trace_bytes.endswith(b'\n') and hashlib.sha256(trace_bytes).hexdigest() == trace_hash,
                 f'{stage_id} trace bytes differ from terminal evidence')
        try:
            trace_records = [json.loads(line) for line in trace_bytes.decode('utf-8').splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'{stage_id} trace is malformed') from exc
        _require(len(trace_records) == trace_count and trace_records
                 and all(isinstance(row, dict) and isinstance(row.get('stage'), str)
                         and row.get('stage') and isinstance(row.get('arrays'), dict) and row['arrays']
                         for row in trace_records),
                 f'{stage_id} trace records do not match terminal evidence')
        runs[(branch, length)] = run

    prefixes = report.get('prefix_checks')
    _require(isinstance(prefixes, dict) and set(prefixes) == set(MUG_PNP_PREFIX_KEYS),
             'report does not contain all six exact prefix comparisons')
    for key, check in prefixes.items():
        _require(isinstance(check, dict)
                 and all(type(check.get(field)) is bool for field in
                         ('passed', 'runtime_equal', 'provenance_equal', 'settings_equal', 'prefix_equal'))
                 and isinstance(check.get('trace'), dict)
                 and check['trace'].get('supplied') is True
                 and type(check['trace'].get('passed')) is bool,
                 f'{key} prefix comparison is malformed')

    paired = report.get('paired_source_settings_checks')
    _require(isinstance(paired, dict) and type(paired.get('passed')) is bool,
             'paired source/settings result is missing')
    snapshot_intact = report.get('frozen_snapshot_intact')
    _require(type(snapshot_intact) is bool, 'frozen snapshot result is malformed')
    gates = report.get('continuation_gate_checks')
    expected_gate_names = {
        'full_six_stage_lifecycle_exit_zero', 'frozen_snapshot_intact',
        'paired_provenance_and_settings_match', 'candidate_accepted_at_least_control',
        'candidate_accepted_at_least_216', 'common_area_median_not_worse',
        'common_area_p95_strictly_better', 'own_area_median_not_worse',
        'own_area_p95_not_worse', 'no_new_90_degree_disagreement',
        *(f'prefix_{key}' for key in MUG_PNP_PREFIX_KEYS),
    }
    _require(isinstance(gates, dict) and set(gates) == expected_gate_names
             and all(type(value) is bool for value in gates.values())
             and type(report.get('continuation_gate_passed')) is bool
             and report['continuation_gate_passed'] is all(gates.values()),
             'continuation gate results are malformed or inconsistent')
    _require(gates['full_six_stage_lifecycle_exit_zero'] is True
             and gates['frozen_snapshot_intact'] is snapshot_intact
             and gates['paired_provenance_and_settings_match'] is paired['passed'],
             'continuation gate does not match terminal report checks')
    for key, check in prefixes.items():
        _require(gates[f'prefix_{key}'] is check['passed'],
                 f'{key} gate does not match its prefix result')
    return report, runs, masks


def _valid_full_frame(frame, expected_mask_sha256):
    state = frame.get('pose_state')
    _require(state in ('tracking', 'recovering', 'lost'), 'saved mug pose state is invalid')
    _require(frame.get('render_state') in ('visible', 'suppressed'), 'saved mug render state is invalid')
    _require(frame.get('mask_sha256') == expected_mask_sha256, 'saved mug mask hash is invalid')
    pose = frame.get('cameraFromObject')
    if state == 'tracking':
        _require(_valid_pose_matrix(pose) and frame.get('render_state') == 'visible'
                 and frame.get('mask_state') == 'available',
                 'tracking frame must have a current visible pose and available mask')
    else:
        _require(pose is None and frame.get('render_state') == 'suppressed',
                 'non-tracking frame must suppress rendering and clear the pose')


def _valid_accounting(accounting, accepted):
    _require(isinstance(accounting, dict)
             and accounting.get('requested_source_frames') == len(MUG_PNP_FRAME_IDS)
             and accounting.get('source_frames') == len(MUG_PNP_FRAME_IDS)
             and accounting.get('run_complete') is True
             and accounting.get('accepted_frames') == accepted
             and accounting.get('failures') == len(MUG_PNP_FRAME_IDS)-accepted,
             'branch availability accounting does not match its saved output')
    availability = accounting.get('accepted_pose_availability')
    _require(_finite_nonnegative(availability)
             and math.isclose(availability, accepted/len(MUG_PNP_FRAME_IDS), abs_tol=1e-12),
             'branch accepted availability is malformed')


def _validate_gate_values(report, control_accepted, candidate_accepted, common_metrics, own_metrics,
                          common_count):
    gates = report['continuation_gate_checks']
    checks = {
        'candidate_accepted_at_least_control': candidate_accepted >= control_accepted,
        'candidate_accepted_at_least_216': candidate_accepted >= 216,
        'common_area_median_not_worse': (
            common_metrics['control']['median_720'] is not None
            and common_metrics['candidate']['median_720'] is not None
            and common_metrics['candidate']['median_720'] <= common_metrics['control']['median_720']),
        'common_area_p95_strictly_better': (
            common_metrics['control']['p95_720'] is not None
            and common_metrics['candidate']['p95_720'] is not None
            and common_metrics['candidate']['p95_720'] < common_metrics['control']['p95_720']),
        'own_area_median_not_worse': (
            own_metrics['candidate']['median_720'] is not None
            and own_metrics['control']['median_720'] is not None
            and own_metrics['candidate']['median_720'] <= own_metrics['control']['median_720']),
        'own_area_p95_not_worse': (
            own_metrics['candidate']['p95_720'] is not None
            and own_metrics['control']['p95_720'] is not None
            and own_metrics['candidate']['p95_720'] <= own_metrics['control']['p95_720']),
    }
    orientation = report.get('new_candidate_large_orientation_disagreements')
    _require(isinstance(orientation, dict) and isinstance(orientation.get('new_candidate_frame_ids'), list),
             'orientation disagreement check is malformed')
    checks['no_new_90_degree_disagreement'] = not orientation['new_candidate_frame_ids']
    for key, value in checks.items():
        _require(gates[key] is value, f'{key} gate differs from the measured outputs')


def _load_mug_pnp_comparison(experiment_root=MUG_PNP_EXPERIMENT):
    """Return a verified completed comparison, or None when it is not safe to display."""
    experiment_root = Path(experiment_root)
    report_path = experiment_root/'report.json'
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report, runs, expected_masks = _validate_mug_pnp_report(report, experiment_root)
        branch_frames = {}
        accounting = report.get('branch_accounting')
        own_area = report.get('own_accepted_fixed128_area_metrics')
        common_area = report.get('common_accepted_fixed128_area_metrics')
        _require(isinstance(accounting, dict) and set(accounting) == set(MUG_PNP_BRANCHES)
                 and isinstance(own_area, dict) and set(own_area) == set(MUG_PNP_BRANCHES)
                 and isinstance(common_area, dict) and set(common_area) == set(MUG_PNP_BRANCHES),
                 'paired branch metrics are incomplete')
        accepted = {}
        own_metrics = {}
        common_metrics = {}
        for branch in MUG_PNP_BRANCHES:
            full = runs[(branch, 240)]
            frames = full['frames']
            for frame in frames:
                _valid_full_frame(frame, expected_masks[str(frame['frameId'])])
            accepted[branch] = sum(frame['pose_state'] == 'tracking' for frame in frames)
            _valid_accounting(accounting[branch], accepted[branch])
            _valid_area_metrics(own_area[branch], len(MUG_PNP_FRAME_IDS), accepted[branch])
            own_metrics[branch] = own_area[branch]
            branch_frames[branch] = {frame['frameId']: frame for frame in frames}
        for frame_id in MUG_PNP_FRAME_IDS:
            _require(branch_frames['control'][frame_id]['mask_sha256']
                     == branch_frames['candidate'][frame_id]['mask_sha256'],
                     f'paired branches use different observed masks at frame {frame_id}')
        common_ids = [frame_id for frame_id in MUG_PNP_FRAME_IDS
                      if (branch_frames['control'][frame_id]['pose_state'] == 'tracking'
                          and branch_frames['candidate'][frame_id]['pose_state'] == 'tracking')]
        _require(report.get('common_accepted_frame_ids') == common_ids
                 and report.get('common_accepted_frames') == len(common_ids),
                 'common accepted frame list does not match the saved branches')
        for branch in MUG_PNP_BRANCHES:
            _valid_area_metrics(common_area[branch], len(common_ids), len(common_ids))
            common_metrics[branch] = common_area[branch]
        _validate_gate_values(report, accepted['control'], accepted['candidate'], common_metrics,
                              own_metrics, len(common_ids))

        modes_by_frame = {}
        for frame_id in MUG_PNP_FRAME_IDS:
            modes_by_frame[frame_id] = {}
            for branch, mode in zip(MUG_PNP_BRANCHES, MUG_PNP_MODES):
                frame = branch_frames[branch][frame_id]
                modes_by_frame[frame_id][mode] = {
                    key: frame.get(key) for key in
                    ('frameId', 'cameraFromObject', 'mask_state', 'pose_state', 'render_state', 'failure_reason')
                }

        common_summary = {
            branch: {key: common_metrics[branch][key] for key in ('median_720', 'p95_720')}
            for branch in MUG_PNP_BRANCHES
        }
        own_summary = {
            branch: {key: own_metrics[branch][key] for key in
                     ('processed', 'accepted', 'median_720', 'p95_720')}
            for branch in MUG_PNP_BRANCHES
        }
        summary = {
            'continuation_gate_passed': report['continuation_gate_passed'],
            'continuation_gate_checks': report['continuation_gate_checks'],
            'branch_accounting': {
                branch: {key: accounting[branch][key] for key in
                         ('requested_source_frames', 'source_frames', 'accepted_frames', 'failures',
                          'accepted_pose_availability')}
                for branch in MUG_PNP_BRANCHES
            },
            'own_area_metrics': own_summary,
            'common_accepted_frames': len(common_ids),
            'common_area_metrics': common_summary,
        }
        return {'summary': summary, 'frames_by_id': modes_by_frame,
                'expected_mask_sha256_by_frame': expected_masks}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError,
            KeyError, IndexError, OverflowError):
        return None


def _mug_pnp_modes_for_player(player, masks, mask_root, comparison):
    """Require an exact viewer timeline and byte-identical observed cached masks."""
    frames = player.get('frames') if isinstance(player, dict) else None
    if (not isinstance(frames, list) or len(frames) != len(MUG_PNP_FRAME_IDS)
            or any(not isinstance(frame, dict) for frame in frames)
            or [frame.get('frameId') for frame in frames] != list(MUG_PNP_FRAME_IDS)):
        return None
    by_id = comparison.get('frames_by_id') if isinstance(comparison, dict) else None
    expected_masks = comparison.get('expected_mask_sha256_by_frame') if isinstance(comparison, dict) else None
    if not isinstance(by_id, dict) or set(by_id) != set(MUG_PNP_FRAME_IDS) or not isinstance(expected_masks, dict):
        return None
    root = Path(mask_root).resolve()
    for frame_id in MUG_PNP_FRAME_IDS:
        record = masks.get(frame_id)
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            return None
        mask_path = (root/record['path']).resolve()
        try:
            mask_path.relative_to(root)
            actual_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            return None
        if actual_hash != expected_masks.get(str(frame_id)):
            return None
        current = by_id[frame_id]
        if set(current) != set(MUG_PNP_MODES):
            return None
        if any(current[mode].get('frameId') != frame_id for mode in MUG_PNP_MODES):
            return None
    return by_id


def _format_px(value):
    return '—' if value is None else f'{value:.2f}'


def _gate_label(key):
    labels = {
        'full_six_stage_lifecycle_exit_zero': 'six-stage terminal lifecycle check failed',
        'frozen_snapshot_intact': 'frozen-input integrity check failed',
        'paired_provenance_and_settings_match': 'paired provenance/settings check failed',
        'candidate_accepted_at_least_control': 'candidate accepted fewer frames than control',
        'candidate_accepted_at_least_216': 'candidate accepted fewer than 216 frames',
        'common_area_median_not_worse': 'common-frame area median worsened',
        'common_area_p95_strictly_better': 'common-frame area P95 was not strictly lower',
        'own_area_median_not_worse': 'own-frame area median worsened',
        'own_area_p95_not_worse': 'own-frame area P95 worsened',
        'no_new_90_degree_disagreement': 'candidate introduced a new 90-degree reference disagreement',
    }
    if key in labels:
        return labels[key]
    if key.startswith('prefix_'):
        parts = key.removeprefix('prefix_').split('_')
        if len(parts) >= 4:
            return f'{parts[0]} {parts[1]}-to-{parts[3]} prefix equivalence failed'
    return 'a continuation criterion failed'


def _mug_pnp_summary_html(comparison):
    summary = comparison['summary']
    gate_passed = summary['continuation_gate_passed']
    failed = [_gate_label(key) for key, passed in summary['continuation_gate_checks'].items() if not passed]
    gate_text = ('The continuation criteria passed for this comparison.' if gate_passed
                 else 'Continuation gate rejected: '+', '.join(failed)+'.')
    rows = []
    labels = {'control': 'Full-window PnP control', 'candidate': 'No-extrinsic-guess ablation'}
    for branch in MUG_PNP_BRANCHES:
        available = summary['branch_accounting'][branch]
        own = summary['own_area_metrics'][branch]
        common = summary['common_area_metrics'][branch]
        rows.append(
            '<tr><td>'+labels[branch]+'</td>'
            f'<td>{available["accepted_frames"]} / 240 ({available["accepted_pose_availability"]:.1%})</td>'
            f'<td>{_format_px(own["median_720"])} / {_format_px(own["p95_720"])}</td>'
            f'<td>{summary["common_accepted_frames"]} / 240</td>'
            f'<td>{_format_px(common["median_720"])} / {_format_px(common["p95_720"])}</td></tr>')
    return (
        '<h2>Fresh full-window mug PnP comparison</h2>'
        f'<p>{gate_text} Overall and independent-accuracy gates remain false. '
        'Both modes use the same mug footage, timeline and observed cached masks. Area-sampled median/P95 '
        'errors are in 720p-equivalent pixels against estimated reference poses from the same model family under study; '
        'they are not independent physical attachment errors.</p>'
        '<table><tr><th>Mode</th><th>Own accepted availability</th>'
        '<th>Own accepted area median / P95 (px)</th><th>Common accepted frames</th>'
        '<th>Common accepted area median / P95 (px)</th></tr>'
        + ''.join(rows) + '</table>')


def _valid_digest_any_case(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdefABCDEF' for character in value))


def _validate_bottle_pose_evidence(report, terminal):
    """Validate the small set of closed R5 result and terminal invariants shown to readers."""
    _require(isinstance(report, dict) and report.get('schema_version') == 1
             and report.get('scope') == 'Frozen R5 CPU-only source-observability pose ablation; no forwards or renders'
             and report.get('status') == 'complete' and report.get('complete') is True,
             'closed R5 report is not terminal and complete')
    _require(report.get('spec_sha256', '').lower() == BOTTLE_POSE_ABLATION_SPEC_SHA256.lower(),
             'R5 report does not bind the frozen specification')
    _require(report.get('planned_conditions') == 12 and report.get('planned_fit_arms') == 24
             and report.get('fit_arm_outcome_count') == 24
             and report.get('all_twelve_conditions_accounted') is True
             and report.get('all_24_arms_accounted') is True,
             'R5 report does not account for all twelve conditions and twenty-four arms')
    _require(report.get('forward_calls') == 0 and report.get('render_calls') == 0
             and report.get('neural_or_model_calls') == 0
             and report.get('reference_or_annotations_loaded') is False,
             'R5 report contains a model, rendering, reference, or annotation call')
    _require(report.get('source_selection_frozen_before_flow_and_evaluator_truth') is True
             and report.get('all_twelve_closed_conditions_loaded_before_cpu_fitting') is True,
             'R5 report does not establish source-only selection and full preflight loading')
    _require(report.get('working_buffer_budget_bytes') == BOTTLE_POSE_ABLATION_WORKING_BUDGET
             and report.get('output_budget_bytes') == BOTTLE_POSE_ABLATION_OUTPUT_BUDGET
             and type(report.get('preflight_max_estimated_working_bytes')) is int
             and 0 < report['preflight_max_estimated_working_bytes'] <= BOTTLE_POSE_ABLATION_WORKING_BUDGET,
             'R5 report violates its frozen working or output budget')
    errors = report.get('errors')
    _require(errors in (None, [], {}), 'R5 report contains errors')
    _require(report.get('independent_real_accuracy_verified') is False
             and report.get('real_pose_accuracy_claimed') is False
             and report.get('default_or_runtime_promotion') is False,
             'R5 report claims real accuracy or promotes a runtime/default')

    expected_pins = report.get('source_pins')
    _require(isinstance(expected_pins, dict)
             and isinstance(expected_pins.get('capture_snapshot'), dict)
             and isinstance(expected_pins.get('zero_view'), dict)
             and isinstance(expected_pins.get('zero_view', {}).get('capture_source_snapshot'), dict)
             and isinstance(expected_pins.get('zero_view', {}).get('current_sources'), dict)
             and isinstance(expected_pins.get('zero_view', {}).get('checkpoint'), dict)
             and expected_pins.get('capture_snapshot')
             == expected_pins.get('zero_view', {}).get('capture_source_snapshot')
             and expected_pins.get('checkpoint_sha256', '').lower()
             == expected_pins.get('zero_view', {}).get('checkpoint', {}).get('sha256', '').lower()
             and expected_pins.get('current_audit_sha256', '').lower()
             == expected_pins.get('zero_view', {}).get('current_sources', {}).get(
                 'bench/quality_bottle_identity_audit.py', '').lower(),
             'R5 report source pins are missing or inconsistent')
    immutable = report.get('immutable_input_revalidation')
    _require(isinstance(immutable, dict) and immutable.get('state') == 'matched'
             and immutable.get('changed_or_missing') == 0
             and type(immutable.get('verified_files')) is int and immutable['verified_files'] > 0
             and isinstance(immutable.get('files'), list)
             and len(immutable['files']) == immutable['verified_files'],
             'R5 immutable inputs did not revalidate')
    for item in immutable['files']:
        _require(isinstance(item, dict) and item.get('state') == 'matched'
                 and item.get('passed') is True
                 and item.get('actual_bytes') == item.get('expected_bytes')
                 and _valid_digest_any_case(item.get('actual_sha256'))
                 and item.get('actual_sha256', '').lower() == item.get('expected_sha256', '').lower(),
                 'R5 report contains an unmatched frozen input')
    preflight = report.get('preflight')
    _require(isinstance(preflight, list) and len(preflight) == 12
             and all(isinstance(row, dict) and row.get('state') == 'loaded'
                     and type(row.get('estimated_working_bytes')) is int
                     and row['estimated_working_bytes'] <= BOTTLE_POSE_ABLATION_WORKING_BUDGET
                     for row in preflight),
             'R5 report does not show all twelve inputs loaded within budget')

    _require(isinstance(terminal, dict) and terminal.get('schema_version') == 1
             and terminal.get('state') == 'terminal' and terminal.get('reviewed_by') == 'Sol'
             and terminal.get('review_scope') == 'bounded CPU-only exploratory twelve-condition/twenty-four-arm ablation'
             and terminal.get('exit_code') == 0 and terminal.get('source_freeze_matched') is True
             and terminal.get('neural_execution') is False and terminal.get('rendering') is False,
             'parent terminal receipt is not a successful frozen CPU-only run')
    terminal_errors = terminal.get('errors')
    _require(terminal_errors in (None, [], {}), 'parent terminal receipt contains errors')
    _require(terminal.get('output_root') == report.get('output_root'),
             'R5 report and terminal receipt name different output roots')
    source_pins = terminal.get('source_pins')
    source_pins_after = terminal.get('source_pins_after')
    _require(isinstance(source_pins, dict) and source_pins
             and source_pins == source_pins_after
             and all(_valid_digest_any_case(value) for value in source_pins.values()),
             'parent terminal receipt does not preserve its source freeze')
    _require(source_pins.get('bench/quality_bottle_pose_ablation.py', '').upper()
             == BOTTLE_POSE_ABLATION_SOURCE_SHA256
             and source_pins.get('bench/test_quality_bottle_pose_ablation.py', '').upper()
             == BOTTLE_POSE_ABLATION_TEST_SHA256
             and source_pins.get('docs/bottle-identity-next-experiment.md', '').upper()
             == BOTTLE_POSE_ABLATION_SPEC_SHA256,
             'parent terminal receipt does not bind the reviewed R5 source and test')

    prerequisite = report.get('exploratory_pose_prerequisite')
    _require(isinstance(prerequisite, dict)
             and prerequisite.get('name') == 'exploratory_source_observable_pose_prerequisite'
             and prerequisite.get('passed') is False
             and prerequisite.get('runtime_integration_authorized') is False
             and prerequisite.get('original_dense_identity_and_R4_eligibility_gates_remain_failed') is True
             and prerequisite.get('positive_and_self_truth_sets_valid') is True
             and prerequisite.get('negative_truth_visible_sets_report_only') is True
             and prerequisite.get('all_three_positive_source_observable_passed') is False
             and prerequisite.get('all_six_self_controls_passed') is False
             and prerequisite.get('at_least_one_positive_improves') is True
             and prerequisite.get('no_new_negative_false_acceptance') is True
             and prerequisite.get('deterministic_repeats_match') is True,
             'R5 exploratory pose prerequisite is not the reviewed failed result')
    dense = report.get('original_dense_gates')
    _require(isinstance(dense, dict)
             and dense.get('r4_all_three_dense_identity_gate_remains_failed') is True
             and dense.get('r1_original_dense_and_patch_scores_preserved') is True
             and dense.get('no_historical_pnp_fit_replayed') is True,
             'R5 report changes or omits its original dense-gate evidence')

    conditions = report.get('conditions')
    _require(isinstance(conditions, list)
             and [row.get('condition_id') for row in conditions if isinstance(row, dict)]
             == list(BOTTLE_POSE_ABLATION_CONDITION_IDS)
             and len(conditions) == len(BOTTLE_POSE_ABLATION_CONDITION_IDS),
             'R5 conditions differ from the frozen twelve-row selection')
    conditions_by_id = {}
    arm_count = 0
    for condition in conditions:
        condition_id = condition['condition_id']
        expected_kind = ('self_control' if condition_id.startswith('zero-') else
                         'synthetic_positive' if '-t0-' in condition_id else 'synthetic_negative')
        _require(condition.get('kind') == expected_kind and condition.get('state') == 'complete'
                 and type(condition.get('frozen_truth_set_count')) is int
                 and _valid_digest_any_case(condition.get('frozen_truth_set_sha256')),
                 f'{condition_id} is not a complete frozen R5 condition')
        arms = condition.get('arms')
        _require(isinstance(arms, list) and len(arms) == 2
                 and [arm.get('arm') for arm in arms if isinstance(arm, dict)]
                 == ['all', 'source_observable'],
                 f'{condition_id} is missing a paired fit arm')
        for arm in arms:
            repeat = arm.get('deterministic_repeat')
            _require(isinstance(repeat, dict) and repeat.get('matched') is True
                     and _valid_digest_any_case(repeat.get('primary_sha256'))
                     and repeat.get('primary_sha256', '').lower()
                     == repeat.get('repeated_sha256', '').lower(),
                     f'{condition_id}/{arm["arm"]} deterministic repeat did not match')
        arm_count += len(arms)
        conditions_by_id[condition_id] = condition
    _require(arm_count == 24, 'R5 arm count is incomplete')

    positive_ids = [condition_id for condition_id in BOTTLE_POSE_ABLATION_CONDITION_IDS
                    if condition_id.startswith('syn-') and '-t0-' in condition_id]
    positive_checks = prerequisite.get('positive_rows')
    _require(isinstance(positive_checks, list)
             and [row.get('condition_id') for row in positive_checks if isinstance(row, dict)] == positive_ids
             and len(positive_checks) == 3
             and all(row.get('truth_set_valid') is True
                     and row.get('source_observable_accepted') is True
                     and row.get('passed') is False for row in positive_checks),
             'R5 positive prerequisite rows do not show the reviewed failed gate')
    positive_results = []
    for condition_id in positive_ids:
        condition = conditions_by_id[condition_id]
        all_arm, selected_arm = condition['arms']
        _require(all_arm.get('accepted_pose_state') == 'accepted'
                 and selected_arm.get('accepted_pose_state') == 'accepted',
                 f'{condition_id} positive poses are unavailable')
        all_error = all_arm.get('proposal_rotation_error')
        selected_error = selected_arm.get('proposal_rotation_error')
        _require(isinstance(all_error, dict) and isinstance(selected_error, dict)
                 and _finite_nonnegative(all_error.get('rotation_error_degrees'))
                 and _finite_nonnegative(selected_error.get('rotation_error_degrees'))
                 and _finite_nonnegative(all_error.get('translation_error_fraction_D'))
                 and _finite_nonnegative(selected_error.get('translation_error_fraction_D')),
                 f'{condition_id} positive truth-pose metrics are incomplete')
        all_projection = all_arm.get('proposal_surface_projection_720')
        selected_projection = selected_arm.get('proposal_surface_projection_720')
        _require(isinstance(all_projection, dict) and isinstance(selected_projection, dict)
                 and all_projection.get('state') == 'scored'
                 and selected_projection.get('state') == 'scored'
                 and all_projection.get('invalid_candidate_projection_count') == 0
                 and selected_projection.get('invalid_candidate_projection_count') == 0,
                 f'{condition_id} report-only full-surface projection metrics are unavailable')
        all_denominator = all_projection.get('denominator')
        selected_denominator = selected_projection.get('denominator')
        _require(isinstance(all_denominator, dict) and isinstance(selected_denominator, dict)
                 and all_denominator.get('count') == condition['frozen_truth_set_count']
                 and selected_denominator.get('count') == all_denominator.get('count')
                 and all_denominator.get('ordered_source_ids_sha256', '').lower()
                 == condition['frozen_truth_set_sha256'].lower()
                 and selected_denominator.get('ordered_source_ids_sha256', '').lower()
                 == all_denominator.get('ordered_source_ids_sha256', '').lower()
                 and selected_denominator.get('ordered_source_rows_sha256', '').lower()
                 == all_denominator.get('ordered_source_rows_sha256', '').lower(),
                 f'{condition_id} full-surface projection denominator shrank by arm')
        _require(_finite_nonnegative(all_projection.get('p95_720'))
                 and _finite_nonnegative(selected_projection.get('p95_720')),
                 f'{condition_id} full-surface projection scores are malformed')
        positive_results.append({
            'frame_id': condition['frame_id'],
            'all_rotation_degrees': all_error['rotation_error_degrees'],
            'selected_rotation_degrees': selected_error['rotation_error_degrees'],
            'all_translation_fraction_D': all_error['translation_error_fraction_D'],
            'selected_translation_fraction_D': selected_error['translation_error_fraction_D'],
            'all_projection_p95_720': all_projection['p95_720'],
            'selected_projection_p95_720': selected_projection['p95_720'],
        })

    self_ids = [condition_id for condition_id in BOTTLE_POSE_ABLATION_CONDITION_IDS
                if condition_id.startswith('zero-')]
    self_checks = prerequisite.get('self_rows')
    _require(isinstance(self_checks, list)
             and [row.get('condition_id') for row in self_checks if isinstance(row, dict)] == self_ids
             and len(self_checks) == 6
             and all(row.get('truth_set_valid') is True and row.get('both_arms_accepted') is True
                     and row.get('errors_no_worse_than_all') is False
                     and row.get('passed') is False
                     and type(row.get('both_arms_within_1deg_001D')) is bool for row in self_checks),
             'R5 self-control prerequisite rows do not match the reviewed failed gate')
    self_controls_within_bound = sum(row['both_arms_within_1deg_001D'] for row in self_checks)
    _require(self_controls_within_bound == 3,
             'R5 self-control accuracy-bound result differs from the reviewed report')
    for condition_id in self_ids:
        arms = conditions_by_id[condition_id]['arms']
        _require(all(arm.get('accepted_pose_state') == 'accepted' for arm in arms),
                 f'{condition_id} self-control fit availability changed')

    negative_ids = [condition_id for condition_id in BOTTLE_POSE_ABLATION_CONDITION_IDS
                    if condition_id.startswith('syn-') and '-t180-' in condition_id]
    negative_rows = prerequisite.get('negative_rows')
    _require(isinstance(negative_rows, list)
             and [row.get('condition_id') for row in negative_rows if isinstance(row, dict)] == negative_ids
             and len(negative_rows) == 3,
             'R5 negative prerequisite rows do not match the three frozen opposite-view conditions')
    negative_unavailable = 0
    negative_support_rejected = 0
    for condition_id, negative in zip(negative_ids, negative_rows):
        _require(negative.get('full_original_wrong_surface_evidence') is True
                 and negative.get('all_arm_false_acceptance') is True
                 and negative.get('source_observable_false_acceptance') is False
                 and negative.get('new_negative_false_acceptance') is False
                 and negative.get('projection_role') == 'report_only',
                 f'{condition_id} negative result omits original wrong-surface evidence')
        arms = conditions_by_id[condition_id]['arms']
        all_arm, selected_arm = arms
        _require(all_arm.get('accepted_pose_state') == 'accepted',
                 f'{condition_id} all-match false acceptance changed')
        if selected_arm.get('fit_state') == 'unavailable':
            _require(condition_id == negative_ids[0]
                     and selected_arm.get('fit_reason') == 'fewer_than_24_confident_masked_correspondences'
                     and selected_arm.get('accepted_pose_state') == 'unavailable',
                     f'{condition_id} unavailable selected arm has an unexpected reason')
            negative_unavailable += 1
        else:
            _require(condition_id in negative_ids[1:]
                     and selected_arm.get('fit_state') == 'refined'
                     and selected_arm.get('accepted_pose_state') == 'rejected'
                     and selected_arm.get('validation_reason') == 'localized_or_ambiguous_support',
                     f'{condition_id} selected arm was not rejected for localized spatial support')
            negative_support_rejected += 1
    _require(negative_unavailable == 1 and negative_support_rejected == 2,
             'R5 negative outcomes differ from the reviewed availability/support result')

    return {
        'positive_results': positive_results,
        'self_control_count': len(self_checks),
        'self_controls_within_bound': self_controls_within_bound,
        'self_controls_no_worse': sum(row['errors_no_worse_than_all'] is True for row in self_checks),
        'negative_all_match_false_acceptances': len(negative_rows),
        'negative_unavailable': negative_unavailable,
        'negative_support_rejected': negative_support_rejected,
        'verified_input_count': immutable['verified_files'],
        'figure_data_uri': None,
    }


def _bottle_pose_summary_from_records(report, terminal, figure_data_uri=None):
    """Return only the reviewed compact facts; invalid records are never displayable."""
    try:
        summary = _validate_bottle_pose_evidence(report, terminal)
        summary['figure_data_uri'] = figure_data_uri
        return summary
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError):
        return None


def _optional_bottle_pose_figure_data_uri(figure_path):
    if figure_path is None:
        return None
    path = Path(figure_path)
    try:
        size = path.stat().st_size
        if size <= 0 or size > BOTTLE_POSE_ABLATION_FIGURE_MAX_BYTES:
            return None
        content = path.read_bytes()
    except OSError:
        return None
    if (len(content) != size or not content.startswith(b'\x89PNG\r\n\x1a\n')
            or hashlib.sha256(content).hexdigest().upper() != BOTTLE_POSE_ABLATION_FIGURE_SHA256):
        return None
    return 'data:image/png;base64,'+base64.b64encode(content).decode('ascii')


def _load_bottle_pose_summary(report_path=None, terminal_path=None, figure_path=None):
    """Load the exact closed R5 report and its Sol terminal receipt, or return None."""
    report_path = BOTTLE_POSE_ABLATION_REPORT if report_path is None else Path(report_path)
    terminal_path = BOTTLE_POSE_ABLATION_TERMINAL if terminal_path is None else Path(terminal_path)
    figure_path = BOTTLE_POSE_ABLATION_FIGURE if figure_path is None else figure_path
    try:
        if (report_path.stat().st_size != BOTTLE_POSE_ABLATION_REPORT_BYTES
                or terminal_path.stat().st_size != BOTTLE_POSE_ABLATION_TERMINAL_BYTES):
            return None
        report_bytes = report_path.read_bytes()
        terminal_bytes = terminal_path.read_bytes()
        if (hashlib.sha256(report_bytes).hexdigest().upper() != BOTTLE_POSE_ABLATION_REPORT_SHA256
                or hashlib.sha256(terminal_bytes).hexdigest().upper() != BOTTLE_POSE_ABLATION_TERMINAL_SHA256):
            return None
        report = json.loads(report_bytes.decode('utf-8'))
        terminal = json.loads(terminal_bytes.decode('utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    figure_data_uri = _optional_bottle_pose_figure_data_uri(figure_path)
    return _bottle_pose_summary_from_records(report, terminal, figure_data_uri)


def _bottle_pose_summary_html(summary):
    if not isinstance(summary, dict):
        return ''
    positive_rows = []
    for result in summary['positive_results']:
        positive_rows.append(
            f'<p><strong>Positive query {result["frame_id"]}:</strong> all → source-observable '
            f'rotation error {result["all_rotation_degrees"]:.3f} → '
            f'{result["selected_rotation_degrees"]:.3f}°; translation error '
            f'{100*result["all_translation_fraction_D"]:.3f} → '
            f'{100*result["selected_translation_fraction_D"]:.3f}% of model diagonal; '
            f'all-original-surface projection P95 '
            f'{result["all_projection_p95_720"]:.3f} → '
            f'{result["selected_projection_p95_720"]:.3f} px at 720p, report-only.</p>')
    figure = ''
    data_uri = summary.get('figure_data_uri')
    if isinstance(data_uri, str) and data_uri.startswith('data:image/png;base64,'):
        figure = (f'<figure style="margin:8px 0"><img alt="Reviewed closed R5 bottle source-observable '
                  f'pose diagnostic figure" style="display:block;max-width:100%;height:auto" src="{data_uri}"></figure>')
    return (
        '<details class="bottlePoseSummary" style="border:1px solid #31505a;border-radius:6px;margin:10px 0;padding:8px 10px">'
        '<summary>Rejected bottle source-observable pose diagnostic · cached synthetic/self views</summary>'
        f'<p>Closed CPU-only R5 evidence: 12/12 conditions and 24/24 fit arms; '
        f'{summary["verified_input_count"]}/{summary["verified_input_count"]} immutable inputs and deterministic repeats matched. '
        'There were zero model, forward or rendering calls. This compares cached synthetic query views and self-view controls, '
        'not real-video attachment or independent accuracy.</p>'
        '<p>All-match arms accepted all three opposite-view poses despite frozen full-field wrong-surface evidence. '
        'Source-observable selection made one proposal unavailable below 24 correspondences and rejected two for '
        'localized/ambiguous spatial support. Those are availability/support rejections; they do not demonstrate corrected identity.</p>'
        + ''.join(positive_rows)
        + f'<p>Both arms were accepted in {summary["self_control_count"]}/{summary["self_control_count"]} self controls, '
        f'but the selected result failed the no-worse comparison in '
        f'{summary["self_control_count"]}/{summary["self_control_count"]}; '
        f'{summary["self_control_count"]-summary["self_controls_within_bound"]}/'
        f'{summary["self_control_count"]} controls failed the frozen 1° / 0.01D self bound in one or both arms.</p>'
        '<p>The exploratory pose prerequisite failed. Runtime/default integration remains off, and the original dense identity '
        'and R4 eligibility gates remain failed. Improved report-only projection scores do not change that decision.</p>'
        + figure + '</details>')


def _bottle_r8_capacity_summary_from_records(actual_worker, actual_terminal,
                                             full_contract, full_terminal):
    """Validate the four fixed closed R8 receipts before presenting their status."""
    try:
        if not all(isinstance(value, dict) for value in
                   (actual_worker, actual_terminal, full_contract, full_terminal)):
            return None
        actual = actual_worker.get('actual')
        if not isinstance(actual, dict):
            return None
        counts = actual.get('counts')
        source_views = actual.get('source_views')
        rows = actual.get('rows')
        actual_claims = actual.get('claims')
        if (actual_worker.get('state') != 'complete_actual_only'
                or actual.get('state') != 'complete'
                or actual.get('complete') is not True
                or actual.get('capacity_screen_complete') is not True
                or actual.get('capacity_only') is not True
                or actual.get('capacity_available') is not False
                or actual.get('full_28_case_screen_complete') is not False
                or actual.get('matching_started') is not False
                or actual.get('fitting_started') is not False
                or not isinstance(counts, dict)
                or counts.get('source_views_expected') != 6
                or counts.get('source_views_evaluated_once') != 6
                or counts.get('rows_expected') != BOTTLE_R8_ACTUAL_ROW_COUNT
                or counts.get('rows_decided') != BOTTLE_R8_ACTUAL_ROW_COUNT
                or counts.get('rows_capacity_possible') != 0
                or counts.get('rows_capacity_unavailable') != BOTTLE_R8_ACTUAL_ROW_COUNT
                or not isinstance(source_views, list)
                or len(source_views) != len(BOTTLE_R8_SOURCE_CONTEXTS)
                or not isinstance(rows, list)
                or len(rows) != BOTTLE_R8_ACTUAL_ROW_COUNT
                or not isinstance(actual_claims, dict)
                or any(actual_claims.get(key) is not False for key in
                       ('accuracy', 'pose', 'full_28_case_screen_complete', 'evaluator_truth_read'))
                or any(actual_claims.get(key) != 0 for key in
                       ('model_calls', 'ncc_calls', 'optimizer_calls', 'render_calls'))):
            return None

        view_summaries = []
        for index, (view, context_id, witness_count) in enumerate(
                zip(source_views, BOTTLE_R8_SOURCE_CONTEXTS, BOTTLE_R8_WITNESS_COUNTS)):
            if not isinstance(view, dict) or view.get('context_id') != context_id:
                return None
            capacity = view.get('capacity')
            if (not isinstance(capacity, dict)
                    or capacity.get('source_context_id') != context_id
                    or capacity.get('capacity_only') is not True
                    or capacity.get('capacity_available') is not False
                    or capacity.get('witness_count') != witness_count
                    or capacity.get('fit_pool_count') != 0
                    or capacity.get('fit_count') != 0
                    or capacity.get('bank_anchor_count') != 0
                    or capacity.get('matching_started') is not False
                    or capacity.get('fitting_started') is not False):
                return None
            view_summaries.append({
                'context_id': context_id,
                'witness_count': witness_count,
                'fit_pool_count': 0,
                'fit_count': 0,
                'bank_anchor_count': 0,
            })

        row_ids = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('condition_id'), str)
                    or row.get('capacity_possible') is not False
                    or row.get('matching_started') is not False
                    or row.get('fitting_started') is not False
                    or row.get('accuracy') is not False):
                return None
            row_ids.append(row['condition_id'])
        if len(set(row_ids)) != BOTTLE_R8_ACTUAL_ROW_COUNT:
            return None

        worker_sha = BOTTLE_R8_ACTUAL_WORKER_SHA256.lower()
        if (actual_terminal.get('state') != 'complete_actual_only'
                or actual_terminal.get('return_code') != 0
                or actual_terminal.get('worker_sha256') != worker_sha
                or actual_terminal.get('final_source_runtime_seal') is not True
                or actual_terminal.get('full_28_case_screen_complete') is not False
                or actual_terminal.get('controls_evaluated') is not False):
            return None

        limits = full_contract.get('limits')
        if (full_contract.get('schema') != 'r8-capacity-root-v1'
                or not isinstance(limits, dict)
                or limits.get('wall_seconds') != 900
                or limits.get('working_bytes') != 134217728
                or limits.get('cache_bytes') != 8589934592
                or limits.get('artifact_bytes') != 8388608):
            return None
        failed_cases = full_terminal.get('all_cases')
        if (full_terminal.get('state') != 'terminal_failure'
                or full_terminal.get('return_code') != 1
                or full_terminal.get('timed_out') is not True
                or full_terminal.get('deadline_exceeded') is not True
                or full_terminal.get('final_source_runtime_seal') is not True
                or full_terminal.get('full_28_case_screen_complete') is not False
                or not isinstance(failed_cases, list)
                or len(failed_cases) != BOTTLE_R8_FULL28_CASE_COUNT
                or any(not isinstance(row, dict)
                       or not isinstance(row.get('case_id'), str)
                       or row.get('state') != 'not_evaluated_due_terminal_failure'
                       for row in failed_cases)):
            return None
        failed_ids = [row['case_id'] for row in failed_cases]
        if len(set(failed_ids)) != BOTTLE_R8_FULL28_CASE_COUNT:
            return None

        return {
            'source_views': view_summaries,
            'actual_rows': BOTTLE_R8_ACTUAL_ROW_COUNT,
            'actual_capacity_possible': 0,
            'full28_rows': BOTTLE_R8_FULL28_CASE_COUNT,
            'full28_unmeasured': len(failed_cases),
            'wall_limit_seconds': limits['wall_seconds'],
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _load_bottle_r8_capacity_summary():
    receipt_specs = (
        (BOTTLE_R8_ACTUAL_WORKER, BOTTLE_R8_ACTUAL_WORKER_BYTES,
         BOTTLE_R8_ACTUAL_WORKER_SHA256),
        (BOTTLE_R8_ACTUAL_TERMINAL, BOTTLE_R8_ACTUAL_TERMINAL_BYTES,
         BOTTLE_R8_ACTUAL_TERMINAL_SHA256),
        (BOTTLE_R8_FULL28_CONTRACT, BOTTLE_R8_FULL28_CONTRACT_BYTES,
         BOTTLE_R8_FULL28_CONTRACT_SHA256),
        (BOTTLE_R8_FULL28_TERMINAL, BOTTLE_R8_FULL28_TERMINAL_BYTES,
         BOTTLE_R8_FULL28_TERMINAL_SHA256),
    )
    records = []
    try:
        for path, expected_bytes, expected_sha256 in receipt_specs:
            if path.stat().st_size != expected_bytes:
                return None
            raw = path.read_bytes()
            if (len(raw) != expected_bytes
                    or hashlib.sha256(raw).hexdigest().upper() != expected_sha256):
                return None
            value = json.loads(raw.decode('utf-8'))
            if not isinstance(value, dict):
                return None
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return _bottle_r8_capacity_summary_from_records(*records)


def _read_bounded_pinned_bytes(path, expected_bytes, byte_cap, expected_sha256):
    try:
        before = path.stat()
        if (path.is_symlink() or not path.is_file() or before.st_size != expected_bytes
                or before.st_size > byte_cap):
            return None
        with path.open('rb') as stream:
            raw = stream.read(byte_cap + 1)
        after = path.stat()
        if (len(raw) != expected_bytes or len(raw) > byte_cap
                or after.st_size != expected_bytes
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ino != after.st_ino
                or hashlib.sha256(raw).hexdigest().upper() != expected_sha256):
            return None
        return raw
    except (OSError, TypeError, ValueError):
        return None


def _r8_mesh_parity_evidence_from_records(summary, svg_bytes):
    """Return display-only facts for the pinned offline CPU geometry receipt."""
    try:
        if (not isinstance(summary, dict) or not isinstance(svg_bytes, bytes)
                or len(svg_bytes) > BOTTLE_R8_MESH_PARITY_SVG_MAX_BYTES
                or summary.get('scope') !=
                'offline CPU mesh/ray diagnostic; synthetic rendered-model depth, not camera depth'
                or summary.get('tracking_accuracy_measured') is not False
                or summary.get('live_fps_measured') is not False
                or summary.get('process_rss_measured') is not False
                or summary.get('exact_oriented_triangles_equal') is not True
                or type(summary.get('triangle_count')) is not int
                or summary.get('triangle_count') != 8316
                or type(summary.get('vertex_count')) is not int
                or summary.get('vertex_count') != 5003
                or type(summary.get('expected_samples')) is not int
                or summary.get('expected_samples') != 96
                or type(summary.get('measured_samples')) is not int
                or summary.get('measured_samples') != 96):
            return None

        elapsed = summary.get('elapsed_seconds')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or not math.isclose(elapsed, 99.078, rel_tol=0.0, abs_tol=0.001)):
            return None
        receipts = summary.get('receipt_sha256')
        receipt_names = {'contract.json', 'loader-worker.json', 'numeric-worker.json', 'terminal.json'}
        if (not isinstance(receipts, dict) or set(receipts) != receipt_names
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in '0123456789abcdef' for char in value.lower())
                       for value in receipts.values())):
            return None

        phases = summary.get('phase_estimates_bytes')
        if (not isinstance(phases, dict)
                or set(phases) != {'loader', 'numeric', 'maximum_sequential'}
                or any(type(value) is not int or value < 0 for value in phases.values())
                or phases['loader'] > 384 * 1024**2
                or phases['numeric'] > 128 * 1024**2
                or phases['maximum_sequential'] != max(phases['loader'], phases['numeric'])):
            return None

        views = summary.get('views')
        if (not isinstance(views, list) or len(views) != len(BOTTLE_R8_MESH_PARITY_CONTEXTS)
                or [view.get('context_id') if isinstance(view, dict) else None for view in views]
                != list(BOTTLE_R8_MESH_PARITY_CONTEXTS)):
            return None
        hypothesis_names = ('native_two_sided', 'renderer_two_sided', 'renderer_ccw_culling')
        p95_values = []
        maximum_residual = 0.0
        max_ccw_stat_difference_m = 0.0
        for view in views:
            hypotheses = view.get('hypotheses')
            if (type(view.get('denominator')) is not int or view['denominator'] != 16
                    or type(view.get('paired_native_renderer_hits')) is not int
                    or view['paired_native_renderer_hits'] != 16
                    or type(view.get('max_native_renderer_z_difference_m')) not in (int, float)
                    or view['max_native_renderer_z_difference_m'] != 0.0
                    or not isinstance(hypotheses, dict) or set(hypotheses) != set(hypothesis_names)):
                return None
            native = hypotheses['native_two_sided']
            renderer = hypotheses['renderer_two_sided']
            culling = hypotheses['renderer_ccw_culling']
            if not (isinstance(native, dict) and isinstance(renderer, dict) and isinstance(culling, dict)):
                return None
            if renderer != native:
                return None
            for row in (native, renderer, culling):
                if (type(row.get('hits')) is not int or row['hits'] != 16
                        or type(row.get('misses')) is not int or row['misses'] != 0
                        or type(row.get('invalid')) is not int or row['invalid'] != 0):
                    return None
                for key in ('median_abs_z_residual_mm', 'p95_abs_z_residual_mm',
                            'max_abs_z_residual_mm'):
                    value = row.get(key)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        return None
                maximum_residual = max(maximum_residual, row['max_abs_z_residual_mm'])
            p95_values.append(native['p95_abs_z_residual_mm'])
            for key in ('median_abs_z_residual_mm', 'p95_abs_z_residual_mm',
                        'max_abs_z_residual_mm'):
                max_ccw_stat_difference_m = max(
                    max_ccw_stat_difference_m,
                    abs(native[key] - culling[key]) * 1e-3,
                )
        if (not 0.996 < min(p95_values) < 1.0
                or not 2.393 < max(p95_values) < 2.395
                or not 4.30 < maximum_residual < 4.32
                or max_ccw_stat_difference_m > 5e-16):
            return None

        svg_text = svg_bytes.decode('utf-8')
        if b'<!doctype' in svg_bytes.lower() or b'<!entity' in svg_bytes.lower():
            return None
        svg_root = ET.fromstring(svg_bytes)
        if (svg_root.tag != BOTTLE_R8_MESH_PARITY_PROJECT_ROOT + 'svg'
                or svg_root.get('viewBox') != '0 0 1440 900'
                or svg_root.get('role') != 'img'):
            return None
        allowed_svg_tags = {'svg', 'title', 'desc', 'style', 'rect', 'text', 'line', 'g', 'circle'}
        for element in svg_root.iter():
            tag = element.tag.rsplit('}', 1)[-1] if isinstance(element.tag, str) else ''
            if tag not in allowed_svg_tags:
                return None
            if tag == 'style':
                stylesheet = (element.text or '').casefold()
                if any(token in stylesheet for token in ('@import', 'url(', '@font-face')):
                    return None
            for key, value in element.attrib.items():
                local_key = key.rsplit('}', 1)[-1].lower()
                if (local_key.startswith('on') or local_key in {'href', 'src'}
                        or 'url(' in value.casefold()):
                    return None
        ns = BOTTLE_R8_MESH_PARITY_PROJECT_ROOT
        title = svg_root.find(ns + 'title')
        description = svg_root.find(ns + 'desc')
        if (title is None or title.text != 'R8 mesh parity SOURCE-ray residual diagnostic'
                or description is None or 'Six fixed SOURCE views' not in (description.text or '')):
            return None

        hypothesis_labels = {
            'Native two-sided': 'native_two_sided',
            'Renderer two-sided': 'renderer_two_sided',
            'Renderer CCW-culling hypothesis': 'renderer_ccw_culling',
        }
        groups = {}
        for group in svg_root.findall('.//' + ns + 'g'):
            label = group.get('aria-label')
            if not isinstance(label, str) or ', ' not in label:
                continue
            context_id, hypothesis_label = label.split(', ', 1)
            hypothesis = hypothesis_labels.get(hypothesis_label)
            if context_id not in BOTTLE_R8_MESH_PARITY_CONTEXTS or hypothesis is None:
                continue
            points = {}
            for circle in group.findall(ns + 'circle'):
                source_id = circle.get('data-source-id')
                residual = circle.get('data-value-mm')
                try:
                    source_id = int(source_id)
                    residual = float(residual)
                except (TypeError, ValueError, OverflowError):
                    return None
                if source_id < 0 or not math.isfinite(residual) or residual < 0 or source_id in points:
                    return None
                points[source_id] = residual
            groups[(context_id, hypothesis)] = points
        if len(groups) != 18:
            return None
        for context_id in BOTTLE_R8_MESH_PARITY_CONTEXTS:
            native = groups.get((context_id, 'native_two_sided'))
            renderer = groups.get((context_id, 'renderer_two_sided'))
            culling = groups.get((context_id, 'renderer_ccw_culling'))
            if (not isinstance(native, dict) or len(native) != 16
                    or renderer != native or not isinstance(culling, dict)
                    or set(culling) != set(native)):
                return None

        return {
            'elapsed_seconds': elapsed,
            'vertex_count': 5003,
            'triangle_count': 8316,
            'expected_samples': 96,
            'measured_samples': 96,
            'p95_min_mm': min(p95_values),
            'p95_max_mm': max(p95_values),
            'max_residual_mm': maximum_residual,
            'max_ccw_stat_difference_m': max_ccw_stat_difference_m,
            'svg_markup': svg_text,
        }
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ET.ParseError):
        return None


def load_r8_mesh_parity_evidence():
    """Load the one pinned compact R8 mesh summary and inline SVG, fail closed."""
    summary_bytes = _read_bounded_pinned_bytes(
        BOTTLE_R8_MESH_PARITY_SUMMARY, BOTTLE_R8_MESH_PARITY_SUMMARY_BYTES,
        BOTTLE_R8_MESH_PARITY_SUMMARY_MAX_BYTES, BOTTLE_R8_MESH_PARITY_SUMMARY_SHA256,
    )
    svg_bytes = _read_bounded_pinned_bytes(
        BOTTLE_R8_MESH_PARITY_SVG, BOTTLE_R8_MESH_PARITY_SVG_BYTES,
        BOTTLE_R8_MESH_PARITY_SVG_MAX_BYTES, BOTTLE_R8_MESH_PARITY_SVG_SHA256,
    )
    if summary_bytes is None or svg_bytes is None:
        return None
    try:
        summary = json.loads(summary_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return _r8_mesh_parity_evidence_from_records(summary, svg_bytes)


def _bottle_r8_mesh_parity_panel_html(evidence):
    if not isinstance(evidence, dict) or not isinstance(evidence.get('svg_markup'), str):
        return ''
    return (
        f'<details id="{BOTTLE_R8_MESH_PARITY_ELEMENT_ID}" class="bottleR8MeshParity" '
        'style="box-sizing:border-box;max-width:100%;overflow:hidden;border:1px solid #31505a;'
        'border-radius:6px;margin:10px 0;padding:8px 10px">'
        '<summary style="display:flex;align-items:center;min-height:44px;cursor:pointer;'
        'line-height:1.4;overflow-wrap:anywhere">Completed CPU geometry check: native and CPU renderer matched '
        'on all 96/96 fixed SOURCE rays</summary>'
        f'<p>Both meshes had {evidence["vertex_count"]:,} vertices and '
        f'{evidence["triangle_count"]:,} oriented triangles. Across the fixed samples, '
        'native and CPU-renderer camera-z values agreed exactly (maximum paired difference 0.0 m). '
        'The reported residual statistics for the modeled CCW-culling hypothesis differed by at most '
        '5e-16 m.</p>'
        f'<p>Residuals against cached synthetic GPU-rendered model camera-z depth had per-view P95s '
        f'from {evidence["p95_min_mm"]:.3f} to {evidence["p95_max_mm"]:.3f} mm; the largest sample was '
        f'{evidence["max_residual_mm"]:.2f} mm. The {evidence["elapsed_seconds"]:.3f}-second total '
        'covers two confirmed sequential CPU processes. This is not camera-sensor depth, tracking '
        'accuracy, footage attachment, hardware-GPU depth equivalence or phone performance. It is not live FPS. '
        'The earlier source-capacity result remains 0/12 rows possible, with all 28 full-screen cases '
        'unmeasured after timeout.</p>'
        '<figure style="max-width:100%;margin:8px 0">'
        '<figcaption class="note">Cached synthetic model-depth SOURCE-ray residuals. Swipe horizontally '
        'inside the chart to inspect all six fixed views.</figcaption>'
        '<div role="region" tabindex="0" aria-label="Scrollable R8 mesh parity diagnostic chart" '
        'style="box-sizing:border-box;max-width:100%;overflow-x:auto;overflow-y:hidden;'
        '-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain">'
        f'<div style="min-width:720px;width:100%">{evidence["svg_markup"]}</div>'
        '</div></figure></details>'
    )


def render_r8_mesh_parity_panel():
    """Render the optional panel from the exact pinned evidence files."""
    return _bottle_r8_mesh_parity_panel_html(load_r8_mesh_parity_evidence())


def insert_r8_mesh_parity_panel(existing_html):
    """Replace/insert the bounded panel before Evidence status in an existing page."""
    if not isinstance(existing_html, str):
        raise TypeError('Existing player HTML must be text')
    panel = render_r8_mesh_parity_panel()
    block = (BOTTLE_R8_MESH_PARITY_START + panel + BOTTLE_R8_MESH_PARITY_END) if panel else ''
    start_count = existing_html.count(BOTTLE_R8_MESH_PARITY_START)
    end_count = existing_html.count(BOTTLE_R8_MESH_PARITY_END)
    if start_count or end_count:
        if start_count == end_count == 1:
            begin = existing_html.index(BOTTLE_R8_MESH_PARITY_START)
            finish = existing_html.index(BOTTLE_R8_MESH_PARITY_END)
            if finish < begin:
                return existing_html
            finish += len(BOTTLE_R8_MESH_PARITY_END)
            return existing_html[:begin] + block + existing_html[finish:]
        return existing_html
    if not panel:
        return existing_html
    anchor = '<h2>Evidence status</h2>'
    if existing_html.count(anchor) != 1:
        return existing_html
    offset = existing_html.index(anchor)
    return existing_html[:offset] + block + existing_html[offset:]


def _bottle_r8_capacity_summary_html(summary):
    if not isinstance(summary, dict):
        return ''
    witnesses = ', '.join(str(row['witness_count']) for row in summary['source_views'])
    return (
        '<details class="bottleR8CapacitySummary" style="border:1px solid #31505a;'
        'border-radius:6px;margin:10px 0;padding:8px 10px;max-width:100%">'
        '<summary>R8 source capacity: 0/12 rows possible; the 28-case screen timed out</summary>'
        f'<p>Closed actual-source receipt: six template views were screened once. '
        f'Witness counts (frames 10, 50, 100; 0° then 180°) were {witnesses}. '
        'Each view had zero disjoint fit-pool IDs, zero selected fit IDs and zero bank anchors. '
        'The frozen reservation structure is saturated by witness overlap in this run; that does not show that source texture is absent.</p>'
        f'<p>The full screen reached its shared {summary["wall_limit_seconds"]}-second limit. '
        f'All {summary["full28_unmeasured"]}/{summary["full28_rows"]} cases remain unmeasured. '
        'No matching, fitting or new poses were run. The source first-hit/camera-z epsilon remains at its frozen value; '
        'renderer-to-camera depth parity is pending. This diagnostic reports capacity and source geometry only, '
        'not attachment accuracy or a tracking improvement.</p></details>')


# Fail-closed presentation helpers for closed calibration evidence.
def _validate_bottle_patch_calibration_evidence(report, terminal, repeat):
    """Return compact facts only for the exact rejected R6 calibration record."""
    _require(isinstance(report, dict) and report.get('schema_version') == 1
             and report.get('scope') == 'R6 CPU-only current-image patch calibration followed by a conditional pose candidate'
             and report.get('status') == 'calibration_failed'
             and report.get('complete') is True
             and report.get('calibration_passed') is False,
             'R6 report is not the reviewed rejected calibration')
    _require(report.get('spec_sha256', '').lower()
             == BOTTLE_PATCH_CALIBRATION_SPEC_SHA256.lower()
             and report.get('r5_report_sha256', '').lower()
             == BOTTLE_PATCH_CALIBRATION_R5_REPORT_SHA256.lower()
             and report.get('r5_terminal_sha256', '').lower()
             == BOTTLE_PATCH_CALIBRATION_R5_TERMINAL_SHA256.lower(),
             'R6 report does not bind the reviewed experiment inputs')
    expected_pins = {
        'bench/quality_bottle_patch_pose_calibration.py': BOTTLE_PATCH_CALIBRATION_SOURCE_SHA256,
        'bench/test_quality_bottle_patch_pose_calibration.py': BOTTLE_PATCH_CALIBRATION_TEST_SHA256,
        'docs/bottle-patch-pose-calibration-spec.md': BOTTLE_PATCH_CALIBRATION_SPEC_SHA256,
    }
    report_pins = report.get('source_pins')
    _require(isinstance(report_pins, dict)
             and all(report_pins.get(path, '').lower() == digest.lower()
                     for path, digest in expected_pins.items()),
             'R6 report source pins differ from the reviewed calibration')
    _require(report.get('planned_conditions') == 12
             and report.get('all_twelve_conditions_accounted') is True
             and report.get('all_twelve_candidates_accounted') is True
             and report.get('candidate_fit_calls') == 0
             and report.get('forward_calls') == 0
             and report.get('model_calls') == 0
             and report.get('render_calls') == 0
             and report.get('reference_or_annotations_loaded') is False
             and report.get('source_selection_frozen_before_flow_or_evaluator_truth') is True,
             'R6 report does not show the frozen no-fit calibration run')
    for key in ('input_revalidation_before_measurement',
                'input_revalidation_after_measurement'):
        validation = report.get(key)
        _require(isinstance(validation, dict) and validation.get('state') == 'matched'
                 and validation.get('changed_or_missing') == 0
                 and type(validation.get('verified_files')) is int
                 and validation['verified_files'] > 0,
                 'R6 report input revalidation did not match')
    preserved = report.get('r4_r5_gates_preserved')
    _require(isinstance(preserved, dict)
             and preserved.get('r4_dense_gate_remains_failed') is True
             and preserved.get('r5_pose_gate_remains_failed') is True,
             'R6 report does not preserve the rejected earlier gates')

    _require(isinstance(terminal, dict) and terminal.get('schema_version') == 1
             and terminal.get('state') == 'terminal'
             and terminal.get('exit_code') == 0
             and terminal.get('reviewed_by') == 'Sol'
             and terminal.get('source_freeze_matched') is True
             and terminal.get('calibration_passed') is False
             and terminal.get('candidate_fit_calls') == 0
             and terminal.get('report_sha256', '').upper()
             == BOTTLE_PATCH_CALIBRATION_REPORT_SHA256,
             'R6 parent receipt is not the successful frozen rejected run')
    terminal_pins = terminal.get('source_pins')
    reviewed_pins = terminal.get('reviewed_sources')
    _require(isinstance(terminal_pins, dict) and terminal_pins
             and terminal.get('source_pins_after') == terminal_pins
             and isinstance(reviewed_pins, dict)
             and all(terminal_pins.get(path, '').upper() == digest.upper()
                     and reviewed_pins.get(path, '').upper() == digest.upper()
                     for path, digest in expected_pins.items()),
             'R6 parent receipt source freeze does not match')

    conditions = report.get('conditions')
    _require(isinstance(conditions, list) and len(conditions) == 12
             and [row.get('condition_id') for row in conditions if isinstance(row, dict)]
             == list(BOTTLE_PATCH_CALIBRATION_CONDITION_IDS),
             'R6 report conditions differ from the frozen twelve-row cohort')
    by_id = {}
    candidate_not_run = 0
    for row in conditions:
        candidate = row.get('candidate')
        _require(row.get('state') == 'measured' and isinstance(candidate, dict)
                 and candidate.get('state') == 'candidate_not_run',
                 'R6 report contains an unmeasured row or fitted candidate')
        by_id[row['condition_id']] = row
        candidate_not_run += 1

    self_ids = ('zero-10-full', 'zero-10-clipped', 'zero-50-full',
                'zero-50-clipped', 'zero-100-full', 'zero-100-clipped')
    _require(all(by_id[condition_id].get('kind') == 'self_control'
                 and isinstance(by_id[condition_id].get('gates'), dict)
                 and by_id[condition_id]['gates'].get('passed') is True
                 for condition_id in self_ids),
             'R6 self-control calibration rows differ from the reviewed pass')

    gate = report.get('calibration_gate')
    required_ids = (*self_ids, 'syn-10-q8-rgb-t0-rgb',
                    'syn-50-q8-rgb-t0-rgb', 'syn-100-q8-rgb-t0-rgb')
    required_rows = gate.get('required_rows') if isinstance(gate, dict) else None
    _require(isinstance(gate, dict) and gate.get('passed') is False
             and gate.get('required_condition_count') == 9
             and isinstance(required_rows, list) and len(required_rows) == 9
             and [row.get('condition_id') for row in required_rows]
             == list(required_ids)
             and [row.get('passed') for row in required_rows]
             == [True] * 6 + [False, True, True],
             'R6 calibration gate differs from the reviewed rejected result')

    positive_results = []
    for condition_id, expected in BOTTLE_PATCH_CALIBRATION_POSITIVE_RESULTS.items():
        frame_id, precision_percent, coverage_percent, passed = expected
        row = by_id[condition_id]
        measurement = row.get('conditional_observable_measurement')
        spatial = row.get('fixed_anchor_spatial_calibration')
        spatial_summary = spatial.get('summary') if isinstance(spatial, dict) else None
        row_gates = row.get('gates')
        _require(row.get('kind') == 'synthetic_positive'
                 and row.get('frame_id') == frame_id
                 and isinstance(measurement, dict)
                 and isinstance(spatial, dict)
                 and isinstance(spatial_summary, dict)
                 and isinstance(row_gates, dict),
                 f'{condition_id} calibration measurements are incomplete')
        precision = measurement.get('correct_fraction_among_verified_confident_true_visible')
        coverage = measurement.get('verified_confident_coverage')
        hull_fraction = spatial_summary.get('supported_hull_fraction')
        _require(_finite_nonnegative(precision) and _finite_nonnegative(coverage)
                 and _finite_nonnegative(hull_fraction)
                 and round(100 * precision, 2) == precision_percent
                 and round(100 * coverage, 2) == coverage_percent
                 and row_gates.get('passed') is passed,
                 f'{condition_id} calibration scores or outcome changed')
        if frame_id == 10:
            _require(measurement.get('coverage_passed') is False
                     and measurement.get('correctness_passed') is True
                     and coverage < 0.50
                     and spatial.get('qualified') is False
                     and hull_fraction < 0.12
                     and row_gates.get('fixed_anchor_spatial_calibration_passed') is False,
                     'synthetic query 10 no longer fails the frozen coverage/spatial checks')
        else:
            _require(measurement.get('coverage_passed') is True
                     and measurement.get('correctness_passed') is True
                     and spatial.get('qualified') is True
                     and hull_fraction >= 0.12
                     and row_gates.get('fixed_anchor_spatial_calibration_passed') is True,
                     f'{condition_id} no longer passes its frozen calibration row')
        positive_results.append({
            'frame_id': frame_id,
            'precision_percent': precision_percent,
            'coverage_percent': coverage_percent,
            'hull_percent': round(100 * hull_fraction, 2),
            'passed': passed,
        })

    negative_ids = ('syn-10-q8-rgb-t180-rgb', 'syn-50-q8-rgb-t180-rgb',
                    'syn-100-q8-rgb-t180-rgb')
    negative_original_visible_counts = []
    for condition_id, expected_visible_count in zip(negative_ids, (8, 0, 0)):
        row = by_id[condition_id]
        measurement = row.get('conditional_observable_measurement')
        original_dense = row.get('original_dense_measurements')
        visible_denominator = (original_dense.get('visible_denominator')
                               if isinstance(original_dense, dict) else None)
        evaluator_evidence = row.get('evaluator_only_evidence')
        _require(row.get('kind') == 'synthetic_negative'
                 and isinstance(measurement, dict)
                 and measurement.get('eligible_true_visible_denominator') == 0
                 and measurement.get('verified_confident_true_visible_count') == 0
                 and measurement.get('empty_eligible_V_unavailable') is True
                 and isinstance(visible_denominator, dict)
                 and visible_denominator.get('count') == expected_visible_count
                 and isinstance(evaluator_evidence, dict)
                 and evaluator_evidence.get('visible_count') == expected_visible_count
                 and evaluator_evidence.get('expected_negative_V_count')
                 == expected_visible_count
                 and evaluator_evidence.get('negative_V_count_matches_frozen_cohort') is True,
                 f'{condition_id} is no longer an unobservable negative condition')
        negative_original_visible_counts.append(expected_visible_count)

    _require(isinstance(repeat, dict) and repeat.get('scope')
             == 'Closed CPU calibration repeat only; no tracking, independent accuracy, pose fitting or runtime promotion'
             and repeat.get('prior_parent_exit') == 1
             and repeat.get('repeat_parent_exit') == 0
             and repeat.get('repeat_parent_sha256', '').upper()
             == BOTTLE_PATCH_CALIBRATION_TERMINAL_SHA256
             and repeat.get('repeat_report_sha256', '').upper()
             == BOTTLE_PATCH_CALIBRATION_REPORT_SHA256
             and repeat.get('repeat_source_freeze_matched') is True
             and repeat.get('calibration_gate_passed') is False
             and repeat.get('candidate_fit_calls') == 0
             and repeat.get('independent_accuracy_verified') is False
             and repeat.get('runtime_or_phone_promotion') is False
             and repeat.get('prior_source_binding_failure_preserved') is True
             and repeat.get('all_twelve_condition_objects_identical') is True,
             'R6 closed repeat does not preserve the failed source-binding trial')
    repeat_rows = repeat.get('rows')
    _require(isinstance(repeat_rows, list) and len(repeat_rows) == 12
             and [row.get('condition_id') for row in repeat_rows]
             == list(BOTTLE_PATCH_CALIBRATION_CONDITION_IDS)
             and all(row.get('complete_condition_identical') is True
                     for row in repeat_rows),
             'R6 repeat did not reproduce all twelve condition records')

    return {
        'positive_results': positive_results,
        'condition_count': len(conditions),
        'measured_count': len(conditions),
        'candidate_not_run_count': candidate_not_run,
        'candidate_fit_calls': 0,
        'self_control_pass_count': len(self_ids),
        'positive_pass_count': 2,
        'negative_unobservable_count': len(negative_ids),
        'negative_original_visible_counts': negative_original_visible_counts,
        'prior_source_binding_failure_preserved': True,
    }


def _bottle_patch_calibration_summary_from_records(report, terminal, repeat):
    try:
        return _validate_bottle_patch_calibration_evidence(report, terminal, repeat)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError):
        return None


def _load_bottle_patch_calibration_summary(report_path=None, terminal_path=None,
                                           repeat_path=None):
    report_path = (BOTTLE_PATCH_CALIBRATION_REPORT if report_path is None
                   else Path(report_path))
    terminal_path = (BOTTLE_PATCH_CALIBRATION_TERMINAL if terminal_path is None
                     else Path(terminal_path))
    repeat_path = (BOTTLE_PATCH_CALIBRATION_REPEAT if repeat_path is None
                   else Path(repeat_path))
    pinned_files = (
        (report_path, BOTTLE_PATCH_CALIBRATION_REPORT_SHA256, 2 * 1024 * 1024),
        (terminal_path, BOTTLE_PATCH_CALIBRATION_TERMINAL_SHA256, 1024 * 1024),
        (repeat_path, BOTTLE_PATCH_CALIBRATION_REPEAT_SHA256, 64 * 1024),
    )
    records = []
    try:
        for path, expected_sha256, max_bytes in pinned_files:
            size = path.stat().st_size
            if size <= 0 or size > max_bytes:
                return None
            raw = path.read_bytes()
            if (len(raw) != size
                    or hashlib.sha256(raw).hexdigest().upper() != expected_sha256):
                return None
            records.append(json.loads(raw.decode('utf-8')))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return _bottle_patch_calibration_summary_from_records(*records)


def _bottle_patch_calibration_summary_html(summary):
    if not isinstance(summary, dict):
        return ''
    rows = []
    for result in summary['positive_results']:
        outcome = 'passed' if result['passed'] else 'failed calibration'
        rows.append(
            f'<li><strong>Synthetic query {result["frame_id"]}:</strong> '
            f'{result["precision_percent"]:.2f}% correct among verified confident true-visible points; '
            f'{result["coverage_percent"]:.2f}% verified coverage among source-observable true-visible samples; '
            f'{outcome}.</li>')
    return (
        '<details class="bottlePatchCalibrationSummary" style="border:1px solid #31505a;'
        'border-radius:6px;margin:10px 0;padding:8px 10px">'
        '<summary>Rejected calibration — no new video poses</summary>'
        '<p>The cached synthetic/self-view calibration failed coverage and spatial spread: query 10 had '
        '42.43% verified coverage among source-observable true-visible samples against 50% and a '
        '9.60% support hull against 12%. '
        f'No pose candidate ran ({summary["candidate_not_run_count"]}/12 remained not run; '
        f'{summary["candidate_fit_calls"]} fit calls); '
        f'all {summary["measured_count"]}/{summary["condition_count"]} frozen conditions were measured.</p>'
        '<ul style="margin:6px 0;padding-left:20px">' + ''.join(rows) + '</ul>'
        f'<p>{summary["self_control_pass_count"]}/6 self controls and synthetic queries 50 and 100 passed. '
        'Query 10 failed: verified coverage among source-observable true-visible samples was 42.43% against a '
        '50% minimum, and the fixed-anchor support hull '
        'was 9.60% against a 12% minimum.</p>'
        f'<p>The three opposite-view rows retain their original visible-surface counts V='
        f'{"/".join(str(value) for value in summary["negative_original_visible_counts"])}. '
        'They had zero verified true-visible matches and no candidate pose fits, so these rows are unobservable '
        'and do not show false-pose rejection. '
        'This synthetic/self-view calibration does not establish real-video mapping.</p>'
        f'<p>The earlier source-binding failure remains preserved; the closed repeat matched all 12 records. '
        'The calibration is rejected; earlier dense-correspondence and pose prerequisites remain failed. '
        'At calibration time, independent labels were 0/120 reviewed. '
        'Tracking defaults and phone predictions were not changed.</p>'
        '</details>')
# End of closed calibration summary helpers.


def contours(mask, simplify=0):
    paths, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None: return []
    if simplify:
        paths = [cv2.approxPolyDP(p, simplify, True) for p in paths]
    return [dict(points=p[:, 0].tolist(), hole=int(hierarchy[0, i, 3]) >= 0) for i, p in enumerate(paths) if len(p) >= 3]


_OFFLINE_REPLAY_NOTICE = ('<p id="replayNotice" role="note"><strong>Offline replay.</strong> '
                          'Masks and poses are precomputed. FPS below measures browser playback, not live tracking.</p>')


def _object_selection_help_html():
    return (
        '<details id="objectSelectionHelp" style="border:1px solid #31505a;'
        'border-radius:6px;margin:10px 0;padding:8px 10px">'
        '<summary style="min-height:44px;display:flex;align-items:center;cursor:pointer">'
        'How object selection works</summary>'
        '<div>'
        '<p>The object buttons switch among saved comparisons for the keyboard, white mug, and dressing bottle. '
        'Each comparison uses its fixed textured 3D model; the buttons do not identify or scan a new object.</p>'
        '<p>Initial setup uses six prompts once: three points on the object and three on the hand or background. '
        'SAM 2 propagates the mask forward. If tracking loses the object, automatic CNOS recovery can search the '
        'image again. No scored correction clicks are used.</p>'
        '<p>Controlled mode gets one onboarding pose. Complete mode estimates its initialization with FoundPose '
        'and GoTrack.</p>'
        '<p>Mask pixels mark image regions; they do not confirm that the 3D model has the correct pose or is '
        'attached correctly.</p>'
        '<p>This page replays precomputed results. Tapping or scanning an arbitrary object on a phone is not '
        'implemented yet.</p>'
        '</div></details>'
    )


def _insert_object_selection_help(html):
    return replace_once(
        html,
        _OFFLINE_REPLAY_NOTICE,
        _OFFLINE_REPLAY_NOTICE + _object_selection_help_html(),
    )


def build(include_r8_mesh_parity=False):
    output = ROOT/'artifacts/model-quality'; output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'artifacts/video-60'
    mug_pnp_comparison = _load_mug_pnp_comparison(MUG_PNP_EXPERIMENT)
    bottle_pose_summary = _load_bottle_pose_summary()
    bottle_r8_capacity_summary = _load_bottle_r8_capacity_summary()
    mug_pnp_available = False
    review_path = output/'failure-review.json'
    reviews = json.loads(review_path.read_text())['objects'] if review_path.exists() else {}
    cases = []; rows = []; model_outputs_present = False
    for alias, label in (('keyboard', 'Keyboard'), ('mug', 'White mug'), ('ranch', 'Dressing bottle')):
        original = json.loads((source/alias/'player.json').read_text())
        original['review_bookmarks'] = reviews.get(alias, {}).get('bookmarks', [])
        directory = output/alias; directory.mkdir(exist_ok=True)
        mask_root = CACHE/'results'/alias/'segmentation'
        mask_file = mask_root/'results.json'
        masks = {f['frameId']: f for f in json.loads(mask_file.read_text())['frames']} if mask_file.exists() else {}
        mug_pnp_modes = None
        if alias == 'mug' and mug_pnp_comparison is not None:
            mug_pnp_modes = _mug_pnp_modes_for_player(original, masks, mask_root, mug_pnp_comparison)
            if mug_pnp_modes is not None:
                original['mug_pnp_comparison'] = mug_pnp_comparison['summary']
                mug_pnp_available = True
        model_outputs_present |= bool(masks)
        predictions = {}
        for mode in ('complete', 'controlled', 'memory', 'appearance', 'texture_control', 'single_sample', 'stable_appearance', 'recovery', 'recovery_association'):
            path = CACHE/'results'/alias/f'{mode}.json' if mode != 'memory' else CACHE/'results'/alias/'model-memory/complete.json'
            if mode == 'memory' and not path.exists():
                path = CACHE/'results'/alias/'model-memory/prefix-30/complete.json'
            if mode == 'appearance':
                path = CACHE/'results'/alias/'appearance/complete.json'
                if not path.exists(): path = CACHE/'results'/alias/'appearance/prefix-30/complete.json'
            if mode == 'texture_control':
                path = CACHE/'results'/alias/'appearance/without-appearance.json'
                if not path.exists(): path = CACHE/'results'/alias/'appearance/prefix-30/without-appearance.json'
            if mode == 'single_sample':
                path = CACHE/'results'/alias/'render-stability/complete.json'
                if not path.exists(): path = CACHE/'results'/alias/'render-stability/prefix-30.json'
            if mode == 'stable_appearance':
                path = CACHE/'results'/alias/'render-stability-appearance/complete.json'
                if not path.exists(): path = CACHE/'results'/alias/'render-stability-appearance/prefix-30.json'
            if mode == 'recovery':
                path = CACHE/'results'/alias/'recovery-1239/complete.json'
            if mode == 'recovery_association':
                path = CACHE/'results'/alias/'recovery-association-1239/complete.json'
            predictions[mode] = {f['frameId']: f for f in json.loads(path.read_text())['frames']} if path.exists() else {}
        original['experiment_status'] = {m: dict(processed=len(predictions[m]),
            accepted=sum(f.get('pose_state') == 'tracking' for f in predictions[m].values())) for m in ('memory', 'appearance', 'texture_control', 'single_sample', 'stable_appearance', 'recovery', 'recovery_association')}
        stress_sets={};original['bookmarks_by_mode']={}
        # State-transition bookmarks require no labels or reference poses.
        appearance_frames=list(predictions['stable_appearance'].values())
        first_veto=next((f for f in appearance_frames if f.get('failure_reason')=='contradicts_rendered_texture'),None)
        if first_veto:
            original['bookmarks_by_mode']['stable_appearance']=[dict(frame_id=first_veto['frameId'],label='Texture contradiction')]
            resumed=next((f for f in appearance_frames if f['frameId']>first_veto['frameId'] and f['pose_state']=='tracking'),None)
            if resumed:
                original['bookmarks_by_mode']['stable_appearance'].append(dict(frame_id=resumed['frameId'],label='Validated rendering resumes'))
            later_resumptions=[newer for older,newer in zip(appearance_frames,appearance_frames[1:])
                if older['pose_state']!='tracking' and newer['pose_state']=='tracking'
                and resumed and newer['frameId']!=resumed['frameId']]
            for newer in later_resumptions[-3:]:
                original['bookmarks_by_mode']['stable_appearance'].append(dict(frame_id=newer['frameId'],label=f'Recovery review: frame {newer["frameId"]}'))
        for stress_mode,stress_folder in [('recovery','recovery-1239'),('recovery_association','recovery-association-1239')]:
            stress_root=CACHE/'results'/alias/stress_folder/'segmentation';stress_path=stress_root/'results.json'
            stress_data=json.loads(stress_path.read_text()) if stress_path.exists() else {}
            stress=stress_data.get('stress_test')
            if not stress:continue
            start=stress['occlusion_start'];end=start+stress['source_frames']
            stress_sets[stress_mode]=(stress_root,{f['frameId']:f for f in stress_data['frames']},start,end)
            original['bookmarks_by_mode'][stress_mode]=[dict(frame_id=fid,label=title) for fid,title in
                [(start,'Interruption begins'),(end,'Visibility returns'),(end+30,'Recovery deadline')]]
            resumed=next((f['frameId'] for f in predictions[stress_mode].values()
                if f['frameId']>=end and f['pose_state']=='tracking'),None)
            if resumed is not None:
                original['bookmarks_by_mode'][stress_mode].insert(2,dict(frame_id=resumed,label='Rendering resumes'))
        annotation_path = output/'annotations'/alias/'annotations.json'
        annotations = json.loads(annotation_path.read_text())
        reviewed = [a for a in annotations['frames'] if a['status'] == 'reviewed']
        landmarks = {p['landmark_id']: p['object_point_m'] for a in reviewed for p in a['landmarks'] if p.get('reviewed')}
        original['landmarks'] = landmarks
        for f in original['frames']:
            fid = f['frameId']; record = masks.get(fid)
            f['mask'] = None
            if record and record['mask_state'] == 'available':
                predicted = cv2.imread(str(mask_root/record['path']), cv2.IMREAD_GRAYSCALE)
                if predicted is not None: f['mask'] = contours(predicted)
            f['mask_state'] = record['mask_state'] if record else 'not_run'
            f['mode_masks']={}
            for stress_mode,(stress_root,stress_masks,start,end) in stress_sets.items():
                stress_record = stress_masks.get(fid); paths = None
                if stress_record and stress_record['mask_state']=='available':
                    stress_bitmap = cv2.imread(str(stress_root/stress_record['path']),cv2.IMREAD_GRAYSCALE)
                    if stress_bitmap is not None: paths = contours(stress_bitmap,simplify=2.)
                f['mode_masks'][stress_mode]=dict(mask=paths,
                    mask_state=stress_record['mask_state'] if stress_record else 'not_run',
                    mask_source='CNOS' if stress_record and 'detection' in stress_record.get('timings_ms',{}) else 'SAM 2',
                    blank_input=start<=fid<end)
            # The preview needs poses/states, not duplicated native-mask hashes,
            # dense-validation diagnostics or per-stage timings. Full records
            # remain preserved in the benchmark cache and measured reports.
            view_keys=('frameId','cameraFromObject','mask_state','pose_state','render_state','failure_reason')
            f['new'] = {m: None if fid not in predictions[m] else
                {key:predictions[m][fid].get(key) for key in view_keys} for m in predictions}
            if mug_pnp_modes is not None:
                f['new'].update(mug_pnp_modes[fid])
            f['manual'] = next((a['landmarks'] for a in reviewed if a['frame_id'] == fid), [])
        write_artifact(directory/'player.json', json.dumps(original, separators=(',', ':')))
        cases.append(dict(name=alias, label=label))
        available = sum(f.get('pose_state') == 'tracking' for f in predictions['complete'].values())
        availability_label = f'{available} accepted' if predictions['complete'] else 'Not measured'
        scored_mask_count = sum(f['frameId'] in masks for f in original['frames'])
        controlled_count = sum(f.get('pose_state') == 'tracking' for f in predictions['controlled'].values())
        controlled_label = f'{controlled_count} / {len(predictions["controlled"])}' if predictions['controlled'] else 'Not measured'
        complete_label = f'{available} / {len(predictions["complete"])}' if predictions['complete'] else availability_label
        experiment_labels = {m: f'{sum(f.get("pose_state") == "tracking" for f in predictions[m].values())} / {len(predictions[m])}' if predictions[m] else 'Not measured' for m in ('texture_control','appearance')}
        rows.append(f'<tr data-object="{alias}"><td>{label}</td><td>240</td><td>{len(reviewed)} / 40</td><td>{scored_mask_count} / 240</td><td>{controlled_label}</td><td>{complete_label}</td><td>{experiment_labels["texture_control"]}</td><td>{experiment_labels["appearance"]}</td><td>Unverified</td></tr>')
    runtime_status = ('Saved pretrained outputs come from an isolated Windows environment. Docker remains untouched. Partial runs are diagnostics; all original frames are required for a benchmark claim.' if model_outputs_present else 'Model inference has not run. Docker remains untouched.')
    appearance_note = ''
    appearance_report = output/'appearance-results.json'
    if appearance_report.exists():
        experiment = json.loads(appearance_report.read_text())['with_appearance']
        agreement = experiment['secondary_pose_agreement']
        if experiment['full_window']:
            appearance_note = (f'<p>Bottle texture experiment: {experiment["accepted"]}/240 accepted; '
                f'estimated-reference attachment error {agreement["median_720"]:.2f} px median / '
                f'{agreement["p95_720"]:.2f} px P95. The previous complete pipeline had 188/240 accepted '
                'and 44.13 px P95. The new result still misses the 10 px P95 target. '
                'Independent accuracy and repeatability remain unverified.</p>')
    stability_report = output/'render-stability-results.json'
    if stability_report.exists():
        experiment = json.loads(stability_report.read_text())
        agreement = experiment['secondary_pose_agreement']
        if experiment['full_window']:
            appearance_note += (f'<p>Keyboard single-sample experiment: {experiment["accepted"]}/240 accepted '
                f'(previously 203/240); estimated-reference attachment error {agreement["median_720"]:.2f} px median / '
                f'{agreement["p95_720"]:.2f} px P95. First 30 predictions match the separate 30-frame run: '
                f'{"yes" if experiment["prefix_identical"] else "no"}. The P95 error still misses the 10 px target; '
                'independent accuracy remains unverified.</p>')
    recovery_report = output/'recovery-results.json'
    if recovery_report.exists():
        recovery_data=json.loads(recovery_report.read_text());recovery=recovery_data['recovery']
        latency=recovery['recovery_source_frames']
        recovery_label='not recovered' if latency is None else f'{latency} source frame'+('' if latency==1 else 's')+' after visibility returns'
        detection_seconds=max((f['elapsed_ms']/1000 for f in recovery_data['automatic_full_image_detection_attempts']),default=None)
        detection_note='' if detection_seconds is None else f' Detector call: {detection_seconds:.1f} s offline.'
        appearance_note += (f'<p>Keyboard forced interruption: hidden rendering suppressed '
            f'{"yes" if recovery["hidden_rendering_suppressed"] else "no"}; reacquisition '
            f'{recovery_label}.{detection_note} '
            'Independent attachment and physical hand-occlusion accuracy remain unverified.</p>')
    recovery_prefix_report=output/'recovery-prefix-results.json'
    if recovery_prefix_report.exists():
        prefix=json.loads(recovery_prefix_report.read_text())
        appearance_note += (f'<p>Keyboard interruption repeatability: separate {prefix["scored_prefix_frames"]}-frame '
            f'prefix versus the full 240-frame run, including loss and recovery. Native masks match: '
            f'{"yes" if prefix["mask_prefix_identical"] else "no"}; poses/states match: '
            f'{"yes" if prefix["pose_prefix_identical"] else "no"}; source/settings match: '
            f'{"yes" if prefix["same_provenance"] and prefix["same_settings"] else "no"}. '
            'This is repeatability evidence, not an independent accuracy result.</p>')
    surface_report=output/'surface-agreement-results.json'
    if surface_report.exists():
        surface=json.loads(surface_report.read_text());before=surface['runs']['previous'];after=surface['runs']['single_sample']
        appearance_note += (f'<p>Additional whole-surface diagnostic: 128 points sampled by triangle area, '
            f'rather than vertex index. Keyboard estimated-reference median/P95: '
            f'{before["median_720"]:.2f}/{before["p95_720"]:.2f} px previously, '
            f'{after["median_720"]:.2f}/{after["p95_720"]:.2f} px with single-sample rendering. '
            'This exposes more remaining drift than the legacy vertex score. It uses model self-visibility '
            'and estimated reference poses; physical hands and independent attachment accuracy remain unverified.</p>')
    bottle_stability=output/'bottle-render-stability-results.json'
    if bottle_stability.exists():
        bottle=json.loads(bottle_stability.read_text());new=bottle['new'];old=bottle['previous_unlit']
        metric=new['secondary_pose_agreement'];area=bottle['current_full_surface_agreement']
        rejected='Rejected bottle single-sample experiment' if bottle.get('candidate_rejected_for_quality') else 'Bottle single-sample experiment'
        appearance_note += (f'<p>{rejected}: {new["accepted"]}/240 accepted '
            f'(earlier unlit control {old["accepted"]}/240); estimated-reference vertex median/P95 '
            f'{metric["median_720"]:.2f}/{metric["p95_720"]:.2f} px; area-sampled median/P95 '
            f'{area["median_720"]:.2f}/{area["p95_720"]:.2f} px. Separate 30-frame prefix matches: '
            f'{"yes" if bottle["prefix_identical"] and bottle["prefix_same_provenance"] and bottle["prefix_same_settings"] else "no"}. '
            'Earlier control has a different code revision; independent accuracy remains unverified. '
            'The result is preserved as an experiment; no default is promoted.</p>')
    stable_appearance=output/'bottle-stable-appearance-results.json'
    if stable_appearance.exists():
        trial=json.loads(stable_appearance.read_text());candidate=trial['new'];control=trial['control']
        area=trial['current_full_surface_agreement'];before=trial['control_full_surface_agreement']
        appearance_note += (f'<p>Bottle stable rendering + texture recovery: {candidate["accepted"]}/240 accepted '
            f'(same-code control {control["accepted"]}/240); estimated-reference area median/P95 '
            f'{area["median_720"]:.2f}/{area["p95_720"]:.2f} px versus '
            f'{before["median_720"]:.2f}/{before["p95_720"]:.2f} px control. '
            f'Actual texture vetoes: {trial["actual_appearance_rejections"]}. Separate 30-frame prefix matches: '
            f'{"yes" if trial["prefix_identical"] and trial["prefix_same_provenance"] and trial["prefix_same_settings"] else "no"}. '
            'All original frames count, including suppressed mapping. Independent accuracy remains unverified; '
            'this is an experiment, with no default promoted.</p>')
        appearance_note += (f'<p>Accepted poses with more than 90 degrees of estimated-reference orientation '
            f'disagreement: {trial["accepted_large_reference_orientation_disagreements"]} in the new bottle trial '
            'versus 32 in its same-code control. Remaining orientation errors need image review; '
            'returned poses are not automatically successful attachment.</p>')
        common=trial['common_current_surface_agreement'];paired=trial['common_control_surface_agreement']
        appearance_note += (f'<p>On the {trial["common_accepted_frames"]} frames accepted by both bottle variants, '
            f'estimated-reference area median/P95: {common["median_720"]:.2f}/{common["p95_720"]:.2f} px candidate '
            f'versus {paired["median_720"]:.2f}/{paired["p95_720"]:.2f} px control. '
            'This paired diagnostic helps separate changed attachment from suppressed failures.</p>')
    neural_audit=CACHE/'diagnostics/keyboard-contour-neural-evaluation.json'
    if neural_audit.exists():
        audit=json.loads(neural_audit.read_text());control=audit['secondary_surface_agreement']['control'];candidate=audit['secondary_surface_agreement']['contour_seed']
        appearance_note += (f'<p>Rejected keyboard contour-seed diagnostic: {audit["requested"]} selected frames, '
            f'{audit["common_validated_frames"]} comparable current-image validations. Secondary area-sampled P95 '
            f'{control["p95_720"]:.2f} px control versus {candidate["p95_720"]:.2f} px candidate. '
            'The proposal is not integrated into tracking or displayed as a successful pose.</p>')
    diagnostic_report=output/'bottle-texture-diagnostic-results.json'
    if diagnostic_report.exists():
        diagnostic=json.loads(diagnostic_report.read_text());trials=diagnostic['paired_neural_trials']
        detail=trials['detail']['secondary_surface_agreement'];shape=trials['shape']['secondary_surface_agreement']
        appearance_note += (f'<p>Rejected bottle rotation diagnostics: ten selected frames, all ten current-image '
            f'correspondence validations in each branch. Estimated-reference area median/P95: '
            f'{detail["control"]["median_720"]:.2f}/{detail["control"]["p95_720"]:.2f} px control, '
            f'{detail["detail_seed"]["median_720"]:.2f}/{detail["detail_seed"]["p95_720"]:.2f} px texture-detail seed, '
            f'{shape["shape_seed"]["median_720"]:.2f}/{shape["shape_seed"]["p95_720"]:.2f} px outline seed. '
            'Both worsen typical error and are not integrated. Joint evidence proposes no seed changes. '
            'These selected-frame diagnostics do not establish full-window or independent accuracy.</p>')
    model_figure=CACHE/'diagnostics/keyboard-model-appearance.jpg';model_note=''
    if model_figure.exists():
        encoded=base64.b64encode(model_figure.read_bytes()).decode('ascii')
        model_note=(f'<details><summary>Keyboard model appearance: inspect the missing underside detail</summary>'
            '<p>The fixed model has key-side texture, but its unlit underside is essentially blank. '
            'The real footage includes rubber feet and printed markings. This limits available model-matching cues; '
            'its causal contribution to tracking error has not been isolated. Source frame 1314 and canonical model-only '
            'views are shown below; no video pose was used to render these model views.</p>'
            f'<img alt="Canonical keyboard key-side and blank underside compared with actual underside footage" '
            f'style="width:100%;height:auto" src="data:image/jpeg;base64,{encoded}"></details>')
    bottle_figure=CACHE/'diagnostics/bottle-texture-detail/comparison.jpg'
    if bottle_figure.exists():
        encoded=base64.b64encode(bottle_figure.read_bytes()).decode('ascii')
        model_note += ('<details><summary>Bottle rotation diagnostics: inspect actual images and competing model renders</summary>'
            '<p>The automatic mask still locates the bottle when its label is hidden. '
            'The saved pose and an alternative rotation can produce similar model views. '
            'Alternatives shown here are unvalidated exploratory proposals; neither projected silhouette '
            'nor alternative pose is used as a foreground prediction or displayed tracking result. '
            'No reference pose was loaded to generate these hypotheses. The refinement trials worsened typical error.</p>'
            f'<img alt="Actual bottle images, automatic masks, saved pose renders and unvalidated rotation alternatives" '
            f'style="width:100%;height:auto" src="data:image/jpeg;base64,{encoded}"></details>')
    mug_pnp_note = _mug_pnp_summary_html(mug_pnp_comparison) if mug_pnp_available else ''
    bottle_pose_note = _bottle_pose_summary_html(bottle_pose_summary)
    bottle_r8_capacity_note = _bottle_r8_capacity_summary_html(bottle_r8_capacity_summary)
    table = '<h2>Evidence status</h2><p>'+runtime_status+' Independent annotation gates remain unverified. The original benchmark and prior failed trackers are preserved.</p>'+appearance_note+bottle_pose_note+bottle_r8_capacity_note+_bottle_patch_calibration_summary_html(_load_bottle_patch_calibration_summary())+mug_pnp_note+'<table><tr><th>Object</th><th>Original frames</th><th>Reviewed annotations</th><th>SAM frames processed</th><th>Controlled accepted / processed</th><th>Complete accepted / processed</th><th>Unlit control accepted / processed</th><th>Texture recovery accepted / processed</th><th>Gate</th></tr>'+''.join(rows)+'</table>'+model_note
    html = HTML.replace('__TABLE__', table).replace('__CASES__', json.dumps(cases))
    html = replace_once(html, '</style>', '''#frameStatus{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin:0 0 12px}
#frameStatus>div{min-width:0;border:1px solid #31505a;border-radius:6px;padding:6px 8px;font-size:13px;line-height:1.35}
#frameStatus .label{display:block;color:#9bb7c0;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
#frameStatus .value{display:block;overflow-wrap:anywhere}
#status{min-width:0;flex:1 1 15rem;overflow-wrap:anywhere}
#replayNotice{padding:10px 12px;border:1px solid #31505a;border-radius:6px;line-height:1.5}
@media(max-width:600px){
 main{padding:12px}
 button,select{min-height:44px}
 input[type=range]{min-height:44px}
 #frameStatus{grid-template-columns:repeat(2,minmax(0,1fr))}
}
</style>''')
    html = replace_once(html, '<title>VisualizeIt · real objects at 60 FPS</title>', '<title>VisualizeIt · model tracking comparison</title>')
    html = replace_once(html, '<h1>Real moving objects · 60 FPS recordings</h1>', '<h1>Model tracking · same footage, four views</h1>'+_OFFLINE_REPLAY_NOTICE)
    html = _insert_object_selection_help(html)
    html = html.replace('width="960" height="436"', 'width="1280" height="436"').replace('ctx.fillRect(0,0,960,436)', 'ctx.fillRect(0,0,1280,436)')
    html = html.replace('Original footage, reference mesh silhouette and mapped surface', 'Original footage, SAM 2 mask, previous tracker and new mapped surface')
    begin = html.index('<label>Mapping mode'); end = html.index('</label>', begin)+len('</label>')
    mug_pnp_options = (''.join(f'<option value="{mode}" data-mug-pnp="true" hidden>{MUG_PNP_MODE_LABELS[mode]}</option>'
                               for mode in MUG_PNP_MODES) if mug_pnp_available else '')
    html = html[:begin]+'''<label>New mapping <select id="mode"><option value="complete">Automatic baseline (original)</option><option value="texture_control">Unlit texture control</option><option value="appearance">Texture recovery experiment</option><option value="single_sample">Single-sample render experiment</option><option value="stable_appearance">Stable rendering + texture recovery</option><option value="recovery">Forced interruption experiment</option><option value="recovery_association">Recovery with object association</option><option value="memory">Model memory experiment</option><option value="controlled">Controlled initialization</option>'''+mug_pnp_options+'''<option value="reference">Reference-pose control</option></select></label> <a href="annotate.html">Review image annotations</a>'''+html[end:]
    begin = html.index('<p class="note">The cyan view'); end = html.index('<div id="table">', begin)
    html = html[:begin]+'''<p class="note">SAM 2 masks, automatic CNOS recovery and GoTrack poses appear only after inference completes. An empty view means unavailable or not run. Reference control uses estimated dataset poses and a geometric silhouette; it can cover a hand and does not validate our tracker. Checker coordinates remain attached to the 3D mesh.</p><p class="note">Yellow markers: independently labelled image landmarks on reviewed frames. Magenta markers: their fixed 3D surface points projected from the selected pose. Until model correspondence annotations are reviewed, markers remain absent. Smooth 60 FPS video playback does not establish live inference FPS.</p>'''+html[end:]
    html = replace_once(html, "['Original footage','Reference mesh silhouette','Mapped design · '+(mode.value==='reference'?'reference pose':mode.value==='1'?'fast tracker':'baseline')]",
        "[mode.value.startsWith('recovery')?'Tracker input · forced interruption':'Original footage','Visible-object mask','Previous baseline · unmodified','New mapping · '+mode.value]")
    html = replace_once(html, "[mode.value.startsWith('recovery')?'Tracker input · forced interruption':'Original footage','Visible-object mask','Previous baseline · unmodified','New mapping · '+mode.value]",
        "[mode.value.startsWith('recovery')?'Tracker input · forced interruption':'Original footage','Visible-object mask',mode.value==='mug_pnp_no_extrinsic_guess'?'Full-window PnP control':'Previous baseline · unmodified','New mapping · '+(mode.options[mode.selectedIndex]?.text||mode.value)]")
    html = replace_once(html, 'for(let i=0;i<3;i++)', 'for(let i=0;i<4;i++)')
    html = replace_once(html, 'const f=data.frames[Math.min(data.frames.length-1,frameIndex)];',
        'const sourceFrame=data.frames[Math.min(data.frames.length-1,frameIndex)],f={...sourceFrame,...(sourceFrame.mode_masks?.[mode.value]||{})};')
    html = replace_once(html,'for(let i=0;i<4;i++)ctx.drawImage(video,i*320,32,320,400);',
        "for(let i=0;i<4;i++){if(f.blank_input&&i!==2){ctx.fillStyle='#000';ctx.fillRect(i*320,32,320,400);}else ctx.drawImage(video,i*320,32,320,400);}")
    old = "maskRenderer.draw(f.reference,true);ctx.drawImage(maskRenderer.canvas,320,32,320,400);\n mappingRenderer.draw(mode.value==='reference'?f.reference:f.poses[Number(mode.value)],false);ctx.drawImage(mappingRenderer.canvas,640,32,320,400);"
    new = '''if(f.mask){maskPath(f.mask,320);ctx.fillStyle='#2ceab680';ctx.fill('evenodd');}
 ctx.fillStyle='#fff';ctx.fillText('Mask: '+f.mask_state+(f.mask_source?' · '+f.mask_source:''),328,420);
 const pairedPnp=mode.value==='mug_pnp_no_extrinsic_guess',controlPrediction=f.new.mug_pnp_control;
 if(pairedPnp){mappingRenderer.draw(controlPrediction?.cameraFromObject,false);if(controlPrediction?.pose_state==='tracking'&&controlPrediction?.render_state==='visible'&&f.mask){ctx.save();maskPath(f.mask,640);ctx.clip('evenodd');ctx.drawImage(mappingRenderer.canvas,640,32,320,400);ctx.restore();}}
 else{mappingRenderer.draw(f.poses[0],false);ctx.drawImage(mappingRenderer.canvas,640,32,320,400);}
 if(pairedPnp){ctx.fillStyle='#fff';ctx.fillText('Control pose: '+(controlPrediction?.pose_state||'not run')+' · render: '+(controlPrediction?.render_state||'not run'),648,420);}
 const prediction=f.new[mode.value],pose=mode.value==='reference'?f.reference:prediction?.cameraFromObject;
 mappingRenderer.draw(pose,false);
 if(mode.value==='reference'){ctx.drawImage(mappingRenderer.canvas,960,32,320,400);}else if(prediction?.pose_state==='tracking'&&prediction?.render_state==='visible'&&f.mask){ctx.save();maskPath(f.mask,960);ctx.clip('evenodd');ctx.drawImage(mappingRenderer.canvas,960,32,320,400);ctx.restore();}
 if(mode.value!=='reference'&&!prediction){ctx.fillStyle='#07141acc';ctx.fillRect(964,40,308,58);ctx.fillStyle='#fff';ctx.fillText('Pose not processed for this frame',974,62);ctx.fillText('Empty mapping is not a tracking failure',974,84);}
 updateFrameStatus(f);
 ctx.fillStyle='#fff';ctx.fillText(mode.options[mode.selectedIndex]?.text||'New mapping',968,420);
 for(const p of f.manual||[]){ctx.fillStyle='#ffdd52';ctx.beginPath();ctx.arc(p.pixel[0]*400/data.height,32+p.pixel[1]*400/data.height,3,0,Math.PI*2);ctx.fill();}
 if(pose)for(const xyz of Object.values(data.landmarks||{})){const v=pose.slice(0,3).map(row=>row[0]*xyz[0]+row[1]*xyz[1]+row[2]*xyz[2]+row[3]);if(v[2]<=0)continue;const x=(data.k[0]*v[0]/v[2]+data.k[2])*400/data.height,y=(data.k[4]*v[1]/v[2]+data.k[5])*400/data.height;ctx.fillStyle='#ff5fca';ctx.beginPath();ctx.arc(960+x,32+y,3,0,Math.PI*2);ctx.fill();}'''
    html = replace_once(html, old, new)
    html = replace_once(html, 'let maskRenderer,mappingRenderer;', '''function maskPath(paths,x){ctx.beginPath();for(const path of paths){const pts=path.points;if(!pts.length)continue;ctx.moveTo(x+pts[0][0]*400/data.height,32+pts[0][1]*400/data.height);for(const p of pts.slice(1))ctx.lineTo(x+p[0]*400/data.height,32+p[1]*400/data.height);ctx.closePath();}}let maskRenderer,mappingRenderer;''')
    html = replace_once(html, "fetch(c.name+'/mesh.bin')", "fetch('../video-60/'+c.name+'/mesh.bin')")
    html = replace_once(html, "video.src=c.name+'/original.webm'", "video.src='../video-60/'+c.name+'/original.webm'")
    # Four-second previews are small enough to load fully. A completed blob
    # avoids a disconnected tunnel range request leaving a seek stuck forever.
    html = replace_once(html, 'frameIndex=0;video.src=', 'frameIndex=0;const clipResponse=await fetch(')
    html = replace_once(html, "fetch('../video-60/'+c.name+'/original.webm';video.load();", "fetch('../video-60/'+c.name+'/original.webm',{signal:AbortSignal.timeout(60000)});if(!clipResponse.ok)throw Error('Source video download failed.');const clip=await clipResponse.blob();if(ticket!==loadNumber)return;if(videoBlobUrl)URL.revokeObjectURL(videoBlobUrl);videoBlobUrl=URL.createObjectURL(clip);video.src=videoBlobUrl;video.load();")
    html = replace_once(html, "(mode.value==='reference'?'dataset reference':f.states[Number(mode.value)])", "(mode.value==='reference'?'dataset reference':f.new[mode.value]?.pose_state||'not run')")
    html = replace_once(html, "status.textContent='Frame '+f.frameId+' · '+(mode.value==='reference'?'dataset reference':f.new[mode.value]?.pose_state||'not run')+' · browser draw '+drawRate.toFixed(0)+' FPS';", "status.textContent=(video.paused?'Paused':'Playing')+' · Frame '+f.frameId+' · '+(mode.value==='reference'?'dataset reference':f.new[mode.value]?.pose_state||'not run')+' · browser draw '+drawRate.toFixed(0)+' FPS';")
    html = replace_once(html, 'video.onended=', "video.addEventListener('seeked',()=>{frameIndex=Math.round(video.currentTime*60);draw();});video.onended=")
    html = replace_once(html,'seek.oninput=()=>{if(ready)video.currentTime=Number(seek.value);};',
        'seek.oninput=()=>{if(ready)seekFrame(Math.round(Number(seek.value)*60));};')
    html = html.replace('href="report.json"', 'href="manifest.json"')
    html = html.replace('<a href="manifest.json">', '<a href="measured-runs.json">Latest measured results</a> · <a href="manifest.json">')
    html = html.replace('<canvas id="view"', '''<label>View <select id="detail"><option value="all">Four views</option><option value="3">New mapping</option><option value="1">Object mask</option><option value="0">Original footage</option><option value="2">Previous tracker</option></select></label><p class="note">Saved results refresh automatically. On a phone, switch between the mask, original and mapping for a larger view.</p><canvas id="detailView" width="320" height="436" hidden aria-label="Selected comparison view" style="max-width:480px;width:100%"></canvas><canvas id="view"''')
    html = html.replace('<label>View <select', '<label>Surface design <select id="design"><option value="checker">Checkerboard</option><option value="directional">Directional color grid</option></select></label> <label>View <select')
    html = replace_once(html, '<canvas id="view" width="1280" height="436" aria-label="Original footage, SAM 2 mask, previous tracker and new mapped surface"></canvas>', '<canvas id="view" width="1280" height="436" aria-label="Original footage, SAM 2 mask, previous tracker and new mapped surface"></canvas><div id="frameStatus" aria-label="New mapping state for the current frame"></div>')
    html = replace_once(html, 'uniform bool silhouette;', 'uniform bool silhouette;uniform bool directional;')
    html = replace_once(html, 'float light=.65+', 'if(directional){base=mix(vec3(.95,.22,.07),vec3(.03,.28,.95),clamp(uv.x,0.,1.));base=mix(base,vec3(.96,.92,.3),clamp(uv.y,0.,1.)*.65);base*=.7+.3*c;}float light=.65+')
    html = replace_once(html, "gl.uniform1i(gl.getUniformLocation(program,'silhouette'),silhouette?1:0);", "gl.uniform1i(gl.getUniformLocation(program,'silhouette'),silhouette?1:0);gl.uniform1i(gl.getUniformLocation(program,'directional'),document.querySelector('#design').value==='directional'?1:0);")
    html = replace_once(html, 'function draw(){ctx.fillStyle=', "function draw(){if(ready&&video.readyState<2){status.textContent='Buffering source video · mapping has not been reevaluated';return;}ctx.fillStyle=")
    html = replace_once(html, 'mode.onchange=draw;', "mode.onchange=()=>{reviewButtons();draw();};document.querySelector('#design').onchange=draw;")
    html = replace_once(html, 'let data=null,frameIndex=0', '''const detail=document.querySelector('#detail'),detailView=document.querySelector('#detailView'),frameStatus=document.querySelector('#frameStatus');
function updateFrameStatus(frame){
 const reference=mode.value==='reference',result=frame.new?.[mode.value];
 const display=value=>String(value??'not reported').replaceAll('_',' ');
 const fields=[
  ['Mask',display(frame.mask_state||'not_run')+(frame.mask_source?' · '+frame.mask_source:'')],
  ['Pose',reference?'estimated reference':display(result?.pose_state??(result?'not_reported':'not_run'))],
  ['Render',reference?'reference control':display(result?.render_state??(result?'not_reported':'not_run'))],
  ['Failure reason',reference?'not applicable':result?(result.failure_reason?display(result.failure_reason):'none'):'not run']
 ];
 const signature=JSON.stringify(fields);
 if(frameStatus.dataset.signature===signature)return;
 frameStatus.dataset.signature=signature;
 frameStatus.replaceChildren(...fields.map(([name,value])=>{
  const field=document.createElement('div'),label=document.createElement('span'),text=document.createElement('span');
  label.className='label';label.textContent=name;text.className='value';text.textContent=value;
  field.append(label,text);return field;
 }));
}
if(matchMedia('(max-width:600px)').matches)detail.value='3';
function refreshDetail(){const single=detail.value!=='all';view.hidden=single;detailView.hidden=!single;if(single){const dc=detailView.getContext('2d');dc.clearRect(0,0,320,436);dc.drawImage(view,Number(detail.value)*320,0,320,436,0,0,320,436);}}
detail.onchange=refreshDetail;
let activeCase=null,resultRevision=null,videoBlobUrl=null;
function updateMugPnpOptions(){const enabled=activeCase?.name==='mug'&&Boolean(data?.mug_pnp_comparison);for(const option of mode.options){if(option.dataset.mugPnp==='true'){option.hidden=!enabled;option.disabled=!enabled;}}if(!enabled&&mode.value.startsWith('mug_pnp_'))mode.value='complete';}
let data=null,frameIndex=0''')
    html = replace_once(html, 'async function selectCase(c){', "async function selectCase(c){activeCase=c;resultRevision=null;updateMugPnpOptions();document.querySelector('#reviewMoments')?.replaceChildren();")
    html = replace_once(html, "async function selectCase(c){activeCase=c;resultRevision=null;updateMugPnpOptions();document.querySelector('#reviewMoments')?.replaceChildren();", "async function selectCase(c){activeCase=c;resultRevision=null;updateMugPnpOptions();frameStatus.replaceChildren();frameStatus.dataset.signature='';ctx.clearRect(0,0,view.width,view.height);detailView.getContext('2d').clearRect(0,0,detailView.width,detailView.height);document.querySelector('#reviewMoments')?.replaceChildren();")
    html = replace_once(html, 'video.pause();ready=false;', "video.pause();play.textContent='Play';ready=false;")
    html = replace_once(html, "async function start(){if(!ready)return;try{await video.play();play.textContent='Pause';}catch(e){play.textContent='Play';}}", "async function start(){if(!ready)return;try{await video.play();play.textContent='Pause';}catch(e){play.textContent='Play';}draw();}")
    html = replace_once(html, "play.onclick=()=>{if(video.paused)start();else{video.pause();play.textContent='Play';}};", "play.onclick=()=>{if(video.paused)start();else{video.pause();play.textContent='Play';draw();}};")
    html = replace_once(html, "fetch(c.name+'/player.json').then(r=>{if(!r.ok)throw Error('Missing clip data.');return r.json();})", "previewAsset(c.name+'/player.json','json',ticket)")
    html = replace_once(html, "fetch('../video-60/'+c.name+'/mesh.bin').then(r=>{if(!r.ok)throw Error('Missing mesh.');return r.arrayBuffer();})", "previewAsset('../video-60/'+c.name+'/mesh.bin','arrayBuffer',ticket)")
    html = replace_once(html, "const clipResponse=await fetch('../video-60/'+c.name+'/original.webm',{signal:AbortSignal.timeout(60000)});if(!clipResponse.ok)throw Error('Source video download failed.');const clip=await clipResponse.blob();", "const clip=await previewAsset('../video-60/'+c.name+'/original.webm','blob',ticket);")
    html = replace_once(html, 'data=d;maskRenderer.load(bytes,d);', 'data=d;updateMugPnpOptions();maskRenderer.load(bytes,d);')
    html = replace_once(html, "catch(e){status.textContent=e.message;}}", "catch(e){if(ticket===loadNumber)status.textContent='Comparison could not load. Tap the object to retry.';}}")
    html = replace_once(html, 'ready=true;paintCount=0;', 'ready=true;reviewButtons();paintCount=0;')
    html = replace_once(html, 'seek.value=video.currentTime;', 'refreshDetail();seek.value=video.currentTime;')
    html = replace_once(html, 'refreshDetail();seek.value=video.currentTime;', "refreshDetail();const ex=data.experiment_status?.[mode.value];if(ex)status.textContent+=' · experiment '+ex.processed+'/240 processed, '+ex.accepted+' accepted';seek.value=video.currentTime;")
    html = replace_once(html, "()=>view.requestFullscreen?.()", "()=>((detail.value==='all')?view:detailView).requestFullscreen?.()")
    html = html.replace('</script></main>', '''
const reviewControls=document.createElement('div');reviewControls.className='controls';reviewControls.id='reviewMoments';
document.querySelector('#table').before(reviewControls);
// WebM timestamps have millisecond rounding. Seek inside the desired frame,
// rather than a fractional boundary which can decode the preceding frame.
function seekFrame(index){if(!ready||!data)return;const i=Math.max(0,Math.min(data.frames.length-1,index));video.currentTime=Math.min((i+.25)/60,Math.max(0,video.duration-.001));}
function reviewButtons(){reviewControls.replaceChildren();const marks=data?.bookmarks_by_mode?.[mode.value]||data?.review_bookmarks;if(!marks?.length)return;
const note=document.createElement('p');note.className='note';note.textContent=mode.value.startsWith('recovery')?'Forced 15-frame black input, then automatic recovery. Display masks are simplified; measurements use native cached pixels.':mode.value==='stable_appearance'?'Automatic texture-check state transitions. These bookmarks use tracker output, not scoring labels.':'Review moments: large disagreement with estimated reference orientation. These are diagnostic bookmarks, not independent accuracy labels.';reviewControls.append(note);
for(const mark of marks){const b=document.createElement('button');b.textContent=mark.label;b.onclick=()=>{if(!ready)return;video.pause();play.textContent='Play';draw();seekFrame(mark.frame_id-data.frames[0].frameId);};reviewControls.append(b);}}
let checkingResults=false;
async function previewAsset(url,kind,ticket){for(let attempt=0;attempt<3;attempt++){
if(ticket!==loadNumber)throw Error('Clip selection changed');
try{const response=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!response.ok)throw Error('Preview asset unavailable');return await response[kind]();}
catch(error){if(ticket!==loadNumber||attempt===2)throw error;status.textContent='Connection interrupted · retrying preview';await new Promise(resolve=>setTimeout(resolve,2000*(attempt+1)));}}}
setInterval(async()=>{if(!ready||!activeCase||checkingResults)return;checkingResults=true;const ticket=loadNumber,c=activeCase;
try{const url=c.name+'/player.json',head=await fetch(url,{method:'HEAD',cache:'no-store'});if(!head.ok)return;
const revision=head.headers.get('etag')||head.headers.get('last-modified')||head.headers.get('content-length');
if(revision===resultRevision)return;const response=await fetch(url,{cache:'no-store'});if(!response.ok)return;
const next=await response.json();if(ticket!==loadNumber)return;data=next;resultRevision=revision;updateMugPnpOptions();
const row=document.querySelector('#table tr[data-object="'+c.name+'"]');if(row){row.children[3].textContent=data.frames.filter(f=>f.mask_state!=='not_run').length+' / 240';
for(const [mode,column]of[['controlled',4],['complete',5],['texture_control',6],['appearance',7]]){const measured=data.frames.map(f=>f.new[mode]).filter(Boolean);row.children[column].textContent=measured.length?(measured.filter(f=>f.pose_state==='tracking').length+' / '+measured.length):'Not measured';}}
reviewButtons();draw();
}catch(e){console.debug('Preview refresh deferred');}finally{checkingResults=false;}},20000);
</script></main>''')
    if include_r8_mesh_parity:
        html = insert_r8_mesh_parity_panel(html)
    write_artifact(output/'index.html', html)


if __name__ == '__main__': build()
