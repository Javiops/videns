"""Public focal-only input materialization, with original global padding width."""
import numpy as np
import torch
from winprob.causal_student_data_v1 import SelectedPublicStore, require
from winprob.iql_training import padded_bags


def focal_store(data, guard=lambda: None, *, device='cuda'):
    # Production callers require CUDA; CPU is available for synthetic input tests.
    require(data.labels_gpu is None and data.labels_cpu is None and data.teacher_gpu is None,
            'public-only inference unexpectedly has supervision tensors')
    n = len(data.focals); width = data.inventory_width
    shapes = {'champion_indices': ((n, 10), torch.long), 'inventory_indices': ((n, 10, width), torch.long),
        'inventory_counts': ((n, 10, width), torch.float32), 'inventory_static': ((n, 10, 22), torch.float32),
        'visible_level': ((n, 10, 1), torch.float32), 'global_public': ((n, 37), torch.float32)}
    tensors = {k: torch.empty(shape, dtype=dtype, device=device) for k, (shape, dtype) in shapes.items()}
    a = data.features
    for lo in range(0, n, 128):
        guard(); hi = min(lo+128, n); selected = data.indices[data.focals[lo:hi]]
        bags = (selected[:, None]*10+np.arange(10)).ravel()
        ids, counts = padded_bags(a['inventory_offsets'], a['inventory_item_indices'], a['inventory_item_counts'], bags, width)
        values = {'champion_indices': a['champion_indices'][selected].astype(np.int64),
            'inventory_indices': ids.reshape(hi-lo, 10, width), 'inventory_counts': counts.reshape(hi-lo, 10, width),
            **{k: a[k][selected] for k in ('inventory_static', 'visible_level', 'global_public')}}
        require(all(np.isfinite(v).all() for v in values.values()), 'nonfinite public inference tensor')
        for key, value in values.items(): tensors[key][lo:hi].copy_(torch.as_tensor(value, device=device))
    runes = torch.as_tensor(data.arrays['rune_indices'][data.focals].astype(np.int64), device=device)
    actions = (torch.as_tensor(data.arrays['action_item_indices'].astype(np.int64), device=device),
        torch.as_tensor(data.arrays['action_item_counts'], device=device), torch.as_tensor(data.arrays['action_attributes'], device=device),
        torch.as_tensor(data.arrays['empty_main_mask'], device=device))
    return SelectedPublicStore(tensors, runes), actions
