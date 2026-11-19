"""Scoped safe loading of our own digest-bound NumPy-string metadata checkpoints."""
from pathlib import Path
import numpy as np
import torch
from winprob.iql_dataset_contract import file_hash


def load_checkpoint(path, expected_sha):
    path = Path(path).resolve(strict=True)
    if file_hash(path) != expected_sha: raise ValueError('checkpoint digest changed before safe loading')
    allowed = [np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType]
    with torch.serialization.safe_globals(allowed):
        value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or value.get('schema') != 'causal-student-checkpoint-v1':
        raise ValueError('wrong owned student checkpoint schema')
    if not all(isinstance(x, (str, np.str_)) for x in value['training_game_ids']):
        raise ValueError('unexpected training identity metadata type')
    value['training_game_ids'] = [str(x) for x in value['training_game_ids']]
    if file_hash(path) != expected_sha: raise ValueError('checkpoint changed while loading')
    return value
