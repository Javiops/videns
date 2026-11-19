"""Fresh current-state/history imitation models; no outcomes or frozen BC weights."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from winprob.iql_models import PublicStateEncoder,set_input_content_validation

MAIN_GROUP_COUNT=9333
HISTORY_SLOTS=2
HISTORY_SLOT_DIM=57


@dataclass(frozen=True)
class HistoryBatch:
    item_indices:torch.Tensor        # [B,2,L], complete retained past baskets
    item_counts:torch.Tensor         # [B,2,L], copies preserved, absent content zero
    static_features:torch.Tensor     # [B,2,23], native attributes; history SAVE coordinate zero
    age:torch.Tensor                 # [B,2], (current start - prior END)/360000
    present:torch.Tensor             # [B,2], recipient's real causal availability


class HistoryProbeActor(nn.Module):
    """Same architecture for state, real history and shuffled-content controls."""
    def __init__(self,num_champions,num_items,class_count=MAIN_GROUP_COUNT):
        super().__init__()
        if type(class_count) is not int or class_count<2:raise ValueError('categorical main-group count required')
        self.state=PublicStateEncoder(num_champions,num_items)
        self.combined=nn.Sequential(nn.Linear(242,128),nn.ReLU(),nn.Linear(128,class_count))
        self.class_count=class_count
        self.check_history_contents=True

    def forward(self,state,history):
        n=state.batch_size
        if (history.item_indices.ndim!=3 or history.item_indices.shape[:2]!=(n,2)
                or history.item_indices.dtype!=torch.long or history.item_counts.shape!=history.item_indices.shape
                or history.static_features.shape!=(n,2,23) or history.age.shape!=(n,2)
                or history.present.shape!=(n,2) or history.present.dtype!=torch.bool
                or history.item_counts.dtype!=torch.float32 or history.static_features.dtype!=torch.float32
                or history.age.dtype!=torch.float32):
            raise ValueError('fixed two-slot history tensor schema required')
        if self.check_history_contents:
            if (not bool(torch.isfinite(history.static_features).all()) or not bool(torch.isfinite(history.age).all())
                    or bool((history.age<0).any()) or bool((history.age[~history.present]!=0).any())
                    or bool((history.static_features[~history.present]!=0).any())
                    or bool((history.static_features[:,:,22]!=0).any())
                    or bool((history.item_indices[~history.present]!=0).any())
                    or bool((history.item_counts[~history.present]!=0).any())):
                raise ValueError('history contents are nonfinite, negative-age or unmasked')
        # This is exactly the SAME item embedding used for current inventories.
        # There is no all-FIT actor, extra learned vocabulary, or duplicate copy.
        baskets=self.state.inventory(history.item_indices,history.item_counts)
        h=torch.cat((baskets,history.static_features,history.age.unsqueeze(-1),
                     history.present.to(torch.float32).unsqueeze(-1)),dim=2).reshape(n,114)
        return self.combined(torch.cat((self.state(state),h),dim=1))


def set_history_validation(actor,enabled=True):
    if type(enabled) is not bool:raise ValueError('boolean validation setting required')
    set_input_content_validation(actor,enabled)
    actor.check_history_contents=enabled


def main_action_loss(logits,labels):
    if (logits.ndim!=2 or logits.shape[0]==0 or labels.shape!=(len(logits),)
            or labels.dtype!=torch.long or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite main logits and aligned categorical targets required')
    loss=nn.functional.cross_entropy(logits,labels)
    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite history-probe loss')
    return loss
