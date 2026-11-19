"""State-only neural input store; cannot access purchase labels."""
from __future__ import annotations
import numpy as np
import torch
from winprob.history_probe_data import STATE_ARRAYS
from winprob.iql_models import PublicBoardBatch
from winprob.iql_training import padded_bags


class StateOnlyStore:
    def __init__(self, features, device='cuda'):
        if set(features) != set(STATE_ARRAYS): raise ValueError('only public state arrays may enter the neural store')
        self.features = dict(features); self.device = torch.device(device)
        self.ally = torch.tensor([False]+[True]*4+[False]*5, dtype=torch.bool, device=device)
        self.enemy = torch.tensor([False]*5+[True]*5, dtype=torch.bool, device=device)

    def states(self, indices):
        rows = np.asarray(indices, dtype=np.int64)
        if rows.ndim != 1 or not 1 <= len(rows) <= 256: raise ValueError('bounded nonempty state batch required')
        a = self.features
        if np.any(rows < 0) or np.any(rows >= len(a['champion_indices'])): raise ValueError('state index outside cache')
        bags = (rows[:, None]*10+np.arange(10)).ravel()
        ids, counts = padded_bags(a['inventory_offsets'], a['inventory_item_indices'], a['inventory_item_counts'], bags)
        return PublicBoardBatch(
            torch.as_tensor(np.array(a['champion_indices'][rows], dtype=np.int64), device=self.device),
            torch.as_tensor(ids.reshape(len(rows), 10, -1), device=self.device),
            torch.as_tensor(counts.reshape(len(rows), 10, -1), device=self.device),
            torch.as_tensor(np.array(a['inventory_static'][rows]), device=self.device),
            torch.as_tensor(np.array(a['visible_level'][rows]), device=self.device),
            self.ally.expand(len(rows), -1), self.enemy.expand(len(rows), -1),
            torch.as_tensor(np.array(a['global_public'][rows]), device=self.device))


def embedding(model, store, rows):
    state = model.state(store.states(rows))
    padded = torch.cat((state, torch.zeros(len(rows), 114, device=state.device)), dim=1)
    return model.combined[1](model.combined[0](padded))


def state_probabilities(model, embeddings):
    logits = model.combined[2](embeddings).cpu().numpy().astype(np.float64)
    logits -= logits.max(1, keepdims=True); np.exp(logits, out=logits); logits /= logits.sum(1, keepdims=True)
    if not np.isfinite(logits).all() or np.any(logits <= 0): raise ValueError('strictly positive finite proxy probabilities required')
    return logits
