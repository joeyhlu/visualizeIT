"""Post-hoc diagnostic screening; never an automatic recovery benchmark."""
import json
from dataclasses import asdict
from .quality_assets import CACHE, ROOT, save_result
from .quality_appearance import AppearanceSettings, appearance_decision


def main():
    data = json.loads((CACHE/'diagnostics/appearance-bottle-full.json').read_text())
    labels = json.loads((ROOT/'artifacts/model-quality/failure-review.json').read_text())['objects']['ranch']['frames']
    angles = {f['frame_id']: f['reference_rotation_disagreement_degrees'] for f in labels}
    settings = AppearanceSettings(grayscale_min=.2)  # Preserve the shaded-render diagnostic configuration.
    rejected = [r['frame_id'] for r in data['rows'] if not appearance_decision(r['prediction'], settings)[0]]
    reference_rejected = [r['frame_id'] for r in data['rows'] if not appearance_decision(r['estimated_reference_control'], settings)[0]]
    disagreements = [fid for fid, angle in angles.items() if angle is not None and angle >= 90]
    report = dict(scope=__doc__, frames=len(data['rows']), thresholds=asdict(settings),
        thresholds_selected_after_original_window_review=True,
        visible_overlap_gate_not_measured_in_this_probe=True,
        saved_accepted_poses=sum(r['saved_pose_state'] == 'tracking' for r in data['rows']),
        texture_contradictions=rejected, reference_control_contradictions=reference_rejected,
        reference_orientation_disagreements=disagreements,
        disagreements_flagged=sorted(set(disagreements)&set(rejected)),
        disagreements_missed=sorted(set(disagreements)-set(rejected)),
        full_model_inference_rerun=False, independent_accuracy_verified=False, overall_gate_passed=False)
    save_result(CACHE/'diagnostics/appearance-screening-report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
