"""Causal board-attention/count-compositional student and separated training loss."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from winprob.iql_models import MultisetSum

KEYSTONES = (8005, 8008, 8010, 8021, 8112, 8128, 9923, 8214, 8229, 8230,
             8437, 8439, 8465, 8351, 8360, 8369, 8992)
STYLES = (8000, 8100, 8200, 8300, 8400)


def require(ok, message):
    if not ok: raise ValueError(message)


class CausalStudent(nn.Module):
    def __init__(self, num_champions, num_items):
        super().__init__()
        self.champions = nn.Embedding(num_champions+1, 16, padding_idx=0)
        self.items = MultisetSum(num_items, 32)
        nn.init.normal_(self.items.embedding.weight)
        with torch.no_grad(): self.items.embedding.weight[0].zero_()
        self.player = nn.Sequential(nn.Linear(71, 64), nn.GELU(), nn.Linear(64, 64))
        self.types = nn.Embedding(4, 64)
        self.keystone = nn.Embedding(len(KEYSTONES)+1, 8)
        self.style = nn.Embedding(len(STYLES)+1, 8)
        self.query = nn.Sequential(nn.Linear(53, 64), nn.GELU(), nn.Linear(64, 64))
        layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=.1, activation='gelu', batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.state_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))
        self.action_head = nn.Sequential(nn.Linear(55, 64), nn.GELU(), nn.Linear(64, 64))
        self.empty_main = nn.Parameter(torch.zeros(64))
        self.interaction = nn.Sequential(nn.Linear(192, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def state(self, board, rune_indices):
        n = board.batch_size
        require(board.champion_indices.shape == (n, 10) and board.inventory_static.shape == (n, 10, 22)
                and board.visible_level.shape == (n, 10, 1) and board.global_public.shape == (n, 37), 'fixed public board geometry required')
        require(rune_indices.shape == (n, 2) and rune_indices.dtype == torch.long, 'two categorical pregame rune indices required')
        require(board.ally_mask.dtype == board.enemy_mask.dtype == torch.bool
                and board.ally_mask.shape == board.enemy_mask.shape == (n, 10), 'typed ten-player relation masks required')
        require(not bool((board.ally_mask & board.enemy_mask).any())
                and bool((board.ally_mask.sum(1) == 4).all()) and bool((board.enemy_mask.sum(1) == 5).all())
                and not bool(board.ally_mask[:, 0].any()) and not bool(board.enemy_mask[:, 0].any()), 'buyer/ally/enemy identity changed')
        types = torch.zeros_like(board.champion_indices)
        types[board.ally_mask] = 1; types[board.enemy_mask] = 2
        player = self.player(torch.cat((self.champions(board.champion_indices),
            self.items(board.inventory_indices, board.inventory_counts), board.inventory_static, board.visible_level), dim=-1))
        player = player+self.types(types)
        query = self.query(torch.cat((board.global_public, self.keystone(rune_indices[:, 0]), self.style(rune_indices[:, 1])), dim=1))
        query = query+self.types.weight[3]
        encoded = self.attention(torch.cat((query[:, None], player), dim=1))
        return self.state_head(torch.cat((encoded[:, 0], encoded[:, 1]), dim=1))

    def actions(self, item_indices, item_counts, attributes, empty_main_mask):
        require(item_indices.ndim == 2 and item_counts.shape == item_indices.shape
                and attributes.shape == (len(item_indices), 23)
                and empty_main_mask.shape == (len(item_indices),) and empty_main_mask.dtype == torch.bool,
                'counted action catalogue tensors required')
        require(bool(((item_counts.sum(1) == 0) == empty_main_mask).all()), 'empty-main identity inconsistent with counts')
        vectors = self.action_head(torch.cat((self.items(item_indices, item_counts), attributes), dim=1))
        return torch.where(empty_main_mask[:, None], self.empty_main[None], vectors)

    def _deep_chunk(self, state, actions):
        a = state[:, None].expand(-1, len(actions), -1)
        b = actions[None].expand(len(state), -1, -1)
        return self.interaction(torch.cat((a, b, a*b), dim=-1)).squeeze(-1)

    def scores(self, state, actions, chunk=512, checkpoint_deep=True):
        require(state.ndim == actions.ndim == 2 and state.shape[1] == actions.shape[1] == 64
                and type(chunk) is int and chunk > 0, 'state/action score geometry required')
        scores = []
        for lo in range(0, len(actions), chunk):
            a = actions[lo:lo+chunk]
            deep = checkpoint(self._deep_chunk, state, a, use_reentrant=False, preserve_rng_state=False) if (
                checkpoint_deep and self.training and torch.is_grad_enabled()) else self._deep_chunk(state, a)
            scores.append((state @ a.T)/math.sqrt(64)+deep)
        return torch.cat(scores, dim=1)

    def forward(self, board, rune_indices, item_indices, item_counts, attributes, empty_main_mask,
                chunk=512, checkpoint_deep=True):
        return self.scores(self.state(board, rune_indices), self.actions(item_indices, item_counts, attributes, empty_main_mask),
                           chunk=chunk, checkpoint_deep=checkpoint_deep)


def purchase_objective(logits, observed_main, teacher=None):
    """Observed purchase stays the hard label; teacher is an optional TRAIN regularizer."""
    require(logits.ndim == 2 and observed_main.shape == (len(logits),) and observed_main.dtype == torch.long
            and bool(torch.isfinite(logits).all()), 'finite aligned logits and actual purchase labels required')
    logq = F.log_softmax(logits, dim=1)
    hard = F.nll_loss(logq, observed_main)
    if teacher is None: return hard, {'hard': hard, 'teacher_KL': torch.zeros_like(hard)}
    require(teacher.shape == logits.shape and not teacher.requires_grad and bool(torch.isfinite(teacher).all())
            and not bool((teacher < 0).any()) and bool((torch.abs(teacher.sum(1)-1) < 2e-6).all()), 'frozen normalized training-only teacher required')
    p = teacher/teacher.sum(1, keepdim=True)
    entropy = -torch.xlogy(p, p).sum(1).mean()
    teacher_kl = -(p*logq).sum(1).mean()-entropy
    return hard+teacher_kl, {'hard': hard, 'teacher_KL': teacher_kl}
