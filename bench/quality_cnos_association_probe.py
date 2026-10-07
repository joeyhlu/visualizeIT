"""Cached current-image detection ranking with a strictly earlier observed mask."""
import json
import numpy as np
from .quality_assets import CACHE,save_result,digest
from .quality_detection_association import associate
from .vision import cv2


def main():
    root=CACHE/'diagnostics';rank=json.loads((root/'cnos-1254-pbr-ranking.json').read_text())
    with np.load(root/'cnos-1254-proposals.npz',allow_pickle=False) as z:masks=z['masks'].copy()
    scores=np.zeros(len(masks));
    for f in rank['ranked']:scores[f['proposal_id']]=f['identity_score']
    prior_path=CACHE/'results/keyboard/recovery-1239/segmentation/masks/1238.png'
    prior=cv2.imread(str(prior_path),cv2.IMREAD_GRAYSCALE)>0
    selected,stats=associate(scores,masks,prior,1254-1238)
    save_result(root/'cnos-1254-association.json',dict(scope=__doc__,selected_proposal=selected,**stats,
        query_sha256=rank['query_sha256'],bank_sha256=rank['bank_sha256'],prior_sha256=digest(prior_path),
        independent_accuracy_verified=False))
    print(json.dumps(dict(selected_proposal=selected,**stats),indent=2),flush=True)


if __name__=='__main__':main()
