"""Associate current foreground proposals with recent observed object identity.

The prior is an observed mask, never projected mesh geometry. It only ranks
current-image proposals and cannot supply a replacement foreground mask.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AssociationSettings:
    maximum_age_frames: int = 30
    minimum_overlap: float = .05
    minimum_identity_score: float = .25
    ambiguity_ratio: float = .95


def associate(scores,masks,prior,age,settings=None):
    settings=settings or AssociationSettings()
    scores=np.asarray(scores,float);masks=np.asarray(masks,bool);prior=np.asarray(prior,bool)
    if masks.ndim!=3 or masks.shape[1:]!=prior.shape or len(scores)!=len(masks):
        raise ValueError('Current proposal and prior dimensions disagree')
    if age<=0:raise ValueError('Prior must precede the current frame')
    if age>settings.maximum_age_frames:return None,dict(reason='object_association_memory_expired',age_frames=age)
    if not prior.any():return None,dict(reason='object_association_prior_empty',age_frames=age)
    intersections=np.count_nonzero(masks & prior,axis=(1,2))
    unions=np.count_nonzero(masks | prior,axis=(1,2));overlaps=intersections/np.maximum(1,unions)
    # Appearance remains an independent minimum. Association is not a new
    # positive detection when the descriptor cannot support object identity.
    eligible=np.isfinite(scores)&(scores>=settings.minimum_identity_score)&(overlaps>=settings.minimum_overlap)
    values=np.where(eligible,scores*overlaps,-1.)
    order=np.argsort(values)[::-1];ranked=[dict(proposal_id=int(i),identity_score=float(scores[i]),
        prior_iou=float(overlaps[i]),association_score=float(values[i]),eligible=bool(eligible[i])) for i in order]
    diagnostic=dict(reason=None,age_frames=age,ranked=ranked)
    if not len(order) or values[order[0]]<0:return None,dict(diagnostic,reason='no_supported_current_object_association')
    best=int(order[0])
    if len(order)>1 and values[order[1]]>=values[best]*settings.ambiguity_ratio:
        other=int(order[1]);union=np.count_nonzero(masks[best]|masks[other])
        if np.count_nonzero(masks[best]&masks[other])/max(1,union)<.5:
            return None,dict(diagnostic,reason='ambiguous_current_object_association')
    return best,diagnostic
