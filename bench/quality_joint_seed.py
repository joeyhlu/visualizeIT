"""Exploratory joint evidence for a refinement seed, never pose acceptance."""
import json
from .quality_assets import CACHE,save_result,digest


def choose(candidates,minimum_shape_gain=.01,minimum_detail_gain=.05):
    seed=next(c for c in candidates if c['degrees']==0)
    baseline=seed['detail'].get('detail_correlation')
    if baseline is None:return seed,'texture_uninformative'
    supported=[c for c in candidates if c['visible_overlap']>=.7
        and c['visible_mask_silhouette_iou']>=seed['visible_mask_silhouette_iou']+minimum_shape_gain
        and c['detail'].get('detail_correlation') is not None
        and c['detail']['detail_correlation']>=baseline+minimum_detail_gain]
    if not supported:return seed,'no_joint_evidence'
    return max(supported,key=lambda c:c['detail']['detail_correlation']),'joint_evidence_proposal'


def main():
    source=CACHE/'diagnostics/bottle-texture-shape/probe.json';data=json.loads(source.read_text());rows=[]
    for row in data['rows']:
        if 'candidates' not in row:rows.append(row);continue
        selected,reason=choose(row['candidates']);rows.append({**row,'best_degrees':selected['degrees'],'selection_reason':reason})
        print(row['frame_id'],selected['degrees'],reason,flush=True)
    save_result(CACHE/'diagnostics/bottle-texture-joint/probe.json',{**data,'scope':__doc__,'rows':rows,
        'proposal_label':'joint_seed','previous_probe_sha256':digest(source),
        'selection_settings':dict(minimum_shape_gain=.01,minimum_detail_gain=.05),
        'caveat':'Exploratory original-window settings, chosen after separate-cue diagnostics; no held-out or independent gate.'})


if __name__=='__main__':main()
