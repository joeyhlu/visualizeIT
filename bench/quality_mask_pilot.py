"""Coarse source-image mask pilot by Codex; requires human audit, excluded from release gates."""
import json
import numpy as np
from .quality_assets import ROOT,CACHE,digest,save_result
from .quality_evaluate import polygon_mask,mask_metrics
from .quality_player import contours
from .vision import cv2


# Coordinates were selected from original-image coordinate views only.
# They deliberately remain separate from the frozen 40-frame annotation files.
LABELS=[dict(frame_id=1149,
    outline=[[144,579],[155,576],[664,598],[691,604],[722,613],[741,620],[760,626],[768,643],
        [770,753],[765,781],[755,789],[683,792],[156,798],[130,797],[130,780],[145,742],[140,720],
        [120,715],[120,686],[121,663],[118,636],[121,617],[135,603],[144,596]],
    hands=[[[60,650],[75,628],[91,618],[104,620],[115,628],[120,643],[122,663],[120,685],
        [118,708],[127,727],[143,738],[154,753],[157,773],[151,792],[134,810],[60,840]],
        [[678,587],[681,578],[693,577],[719,585],[740,594],[753,608],[764,620],[774,634],
         [806,630],[825,570],[850,850],[778,804],[770,761],[770,697],[773,663],[770,646],[763,635],
         [747,624],[733,611],[710,600],[688,595]]]),
    dict(frame_id=1314,
    outline=[[530,436],[639,447],[650,452],[656,465],[682,950],[676,973],[548,985],
        [474,979],[469,950],[455,947],[444,940],[432,923],[426,904],[438,816],[479,580],[514,453],[520,439]],
    hands=[[[680,905],[666,870],[647,833],[631,794],[624,762],[617,741],[609,734],[602,735],
        [600,744],[600,765],[604,786],[609,815],[604,834],[591,813],[575,790],[561,764],
        [548,740],[535,725],[523,717],[514,720],[510,729],[511,747],[518,770],[532,799],
        [549,830],[559,849],[559,856],[550,850],[534,831],[520,802],[503,775],[487,754],
        [478,750],[469,755],[468,766],[470,790],[478,824],[482,851],[491,884],[504,919],
        [519,951],[529,983],[560,1030],[700,1100],[800,1000]],
        [[477,674],[478,615],[479,581],[481,567],[486,563],[492,566],[496,578],[491,616],[483,655]],
        [[664,568],[671,565],[677,572],[676,583],[666,590]],
        [[666,601],[680,580],[690,578],[698,581],[701,590],[698,600],[684,613],[668,626]],
        [[670,643],[688,602],[701,596],[715,594],[724,599],[728,607],[724,617],[707,630],[674,663]],
        [[672,676],[687,649],[696,643],[704,645],[706,653],[700,665],[673,692]]])]


def main():
    out=CACHE/'diagnostics/keyboard-mask-pilot';out.mkdir(parents=True,exist_ok=True)
    rows=[];labels=[]
    manifest=json.loads((CACHE/'inputs/keyboard/input.json').read_text())
    for label in LABELS:
        fid=label['frame_id'];source=ROOT/f'artifacts/model-quality/annotations/keyboard/{fid}.jpg'
        hands=polygon_mask(label['hands'],(1280,1024))
        # Coarse outline minus source-only hand labels. Thin/blurred boundaries
        # remain uncertain and are not eligible for release-gate scoring.
        visible=polygon_mask([label['outline']],(1280,1024))&~hands
        cv2.imwrite(str(out/f'{fid}-visible.png'),visible.astype(np.uint8)*255)
        cv2.imwrite(str(out/f'{fid}-hands.png'),hands.astype(np.uint8)*255)
        predictions={}
        for name,folder in [('original','segmentation'),('interruption_association','recovery-association-1239/segmentation')]:
            mask=cv2.imread(str(CACHE/'results/keyboard'/folder/'masks'/f'{fid}.png'),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise ValueError('Actual cached prediction required')
            predictions[name]=mask_metrics(mask>0,visible,hands)
        rows.append(dict(frame_id=fid,metrics=predictions))
        labels.append(dict(frame_id=fid,source_image_sha256=digest(source),
            visible_object=contours(visible.astype(np.uint8)*255),overlapping_hands=label['hands']))
    save_result(out/'annotations.json',dict(scope=__doc__,operator='Codex visual inspection',
        source_video_sha256=digest(CACHE/'inputs/keyboard'/manifest['video']),
        native_resolution=[1024,1280],status='coarse_agent_review_requires_human_audit',
        selected_after_candidate_results=True,reference_or_predicted_poses_used_for_label_coordinates=False,
        boundary_uncertainty_native_pixels='approximately 3–8; blurred edges may be worse',frames=labels))
    report=dict(scope=__doc__,frames=rows,independent_accuracy_verified=False,overall_gate_passed=False,
        frozen_40_frame_annotations_modified=False)
    save_result(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
