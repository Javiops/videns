"""Flexible full-basket IQL models and update primitives in the own Torch runtime.

No datasets, references, source-project code, or checkpoint is loaded here.
Public tensor semantics and full-action IDs must be validated upstream.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from torch import nn
from winprob.iql_features import GLOBAL_NAMES, INVENTORY_NAMES

PLAYER_STATIC_DIM = len(INVENTORY_NAMES)
GLOBAL_PUBLIC_DIM = len(GLOBAL_NAMES)
ACTION_STATIC_DIM = 23
FULL_ACTION_COUNT = 13166


@dataclass(frozen=True)
class PublicBoardBatch:
    champion_indices: torch.Tensor       # [B,10], buyer first, immutable vocabulary
    inventory_indices: torch.Tensor      # [B,10,L], no fixed-L truncation
    inventory_counts: torch.Tensor       # [B,10,L], counts retained
    inventory_static: torch.Tensor       # [B,10,22], public static item aggregation
    visible_level: torch.Tensor          # [B,10,1], preprocessed public level feature
    ally_mask: torch.Tensor              # [B,10], four others, buyer excluded
    enemy_mask: torch.Tensor             # [B,10], five opponents
    global_public: torch.Tensor         # [B,37], independently audited POV allowlist

    @property
    def batch_size(self):
        return self.champion_indices.shape[0]

    def select(self, indices):
        return PublicBoardBatch(**{name: getattr(self, name).index_select(0, indices)
                                   for name in self.__dataclass_fields__})

    def to(self, device, non_blocking=False):
        return PublicBoardBatch(**{name: getattr(self, name).to(device, non_blocking=non_blocking)
                                   for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class FullActionBatch:
    item_indices: torch.Tensor           # [B,L], complete action, including wards
    item_counts: torch.Tensor            # [B,L]
    static_features: torch.Tensor        # [B,23], SAVE distinct from ward-only

    def to(self, device, non_blocking=False):
        return FullActionBatch(**{name: getattr(self, name).to(device, non_blocking=non_blocking)
                                  for name in self.__dataclass_fields__})


def _padded_bags(bags, leading_shape, device):
    length = max((len(bag) for bag in bags), default=0)
    indices = torch.zeros((*leading_shape, length), dtype=torch.long, device=device)
    counts = torch.zeros((*leading_shape, length), dtype=torch.float32, device=device)
    flat_indices, flat_counts = indices.reshape(len(bags), length), counts.reshape(len(bags), length)
    for row, bag in enumerate(bags):
        if len({item for item, _ in bag}) != len(bag):
            raise ValueError('ragged bags must combine duplicate item multiplicities')
        for column, (item, count) in enumerate(bag):
            if type(item) is not int or type(count) is not int or item <= 0 or count <= 0:
                raise ValueError('ragged bags require known positive vocabulary indices and integer counts')
            flat_indices[row, column], flat_counts[row, column] = item, count
    return indices, counts


def collate_public_boards(compiled_states, device='cpu'):
    """Pad only to this batch's true longest ragged inventory; never truncate."""
    if not compiled_states or any(len(row['inventory_bags']) != 10 for row in compiled_states):
        raise ValueError('nonempty compiled ten-player public states required')
    bags = [bag for row in compiled_states for bag in row['inventory_bags']]
    indices, counts = _padded_bags(bags, (len(compiled_states), 10), device)
    import numpy as np
    def tensor(key, dtype):
        return torch.as_tensor(np.stack([row[key] for row in compiled_states]), dtype=dtype, device=device)
    return PublicBoardBatch(tensor('champion_indices', torch.long), indices, counts,
                            tensor('inventory_static', torch.float32), tensor('visible_level', torch.float32),
                            tensor('ally_mask', torch.bool), tensor('enemy_mask', torch.bool),
                            tensor('global_public', torch.float32))


def collate_full_actions(compiled_actions, device='cpu'):
    if not compiled_actions:
        raise ValueError('nonempty compiled full-action list required')
    import numpy as np
    indices, counts = _padded_bags([pair[0] for pair in compiled_actions], (len(compiled_actions),), device)
    features = torch.as_tensor(np.stack([pair[1] for pair in compiled_actions]), dtype=torch.float32, device=device)
    return FullActionBatch(indices, counts, features)


def _finite(values, name, check_contents=True):
    if not torch.is_floating_point(values) or (check_contents and not bool(torch.isfinite(values).all())):
        raise ValueError(name+' must contain finite floating features')


def _vector(values, name, n=None):
    if values.ndim != 1 or not len(values) or (n is not None and len(values) != n):
        raise ValueError(name+' must be a nonempty observed-row vector')
    _finite(values, name)
    return values


def sampling_mass(weights, n, like):
    if weights is None:
        return torch.full((n,), 1/n, dtype=like.dtype, device=like.device)
    _vector(weights, 'sampling weights', n)
    if bool((weights <= 0).any()):
        raise ValueError('sampling weights must be positive')
    return weights.detach()/weights.detach().sum()


class MultisetSum(nn.Module):
    """Count-weighted inventory/action sum; padding is not an item or a cap."""
    def __init__(self, num_items, dimensions=32):
        super().__init__()
        self.check_input_contents = True
        self.embedding = nn.Embedding(num_items+1, dimensions, padding_idx=0)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, indices, counts):
        if indices.dtype != torch.long or indices.shape != counts.shape or indices.ndim < 2:
            raise ValueError('aligned long item indices and counts required')
        if self.check_input_contents and (bool((indices < 0).any()) or bool((indices >= self.embedding.num_embeddings).any())):
            raise ValueError('unknown item index; do not silently map it to padding')
        _finite(counts, 'item multiplicities', self.check_input_contents)
        if self.check_input_contents and (bool((counts < 0).any()) or not torch.equal(counts, counts.round())
                or bool(((indices == 0) & (counts != 0)).any())
                or bool(((indices != 0) & (counts <= 0)).any())):
            raise ValueError('canonical counts/padding disagree')
        return (self.embedding(indices)*counts.unsqueeze(-1)).sum(dim=-2)


class PublicStateEncoder(nn.Module):
    """71->64->64 per player; symmetric 320+37 ->256->128 board encoder."""
    def __init__(self, num_champions, num_items):
        super().__init__()
        self.check_input_contents = True
        self.champions = nn.Embedding(num_champions+1, 16, padding_idx=0)
        nn.init.zeros_(self.champions.weight)
        self.inventory = MultisetSum(num_items, 32)
        self.player = nn.Sequential(nn.Linear(71, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.board = nn.Sequential(nn.Linear(357, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128))

    def forward(self, batch):
        n = batch.batch_size
        if batch.champion_indices.shape != (n, 10) or batch.champion_indices.dtype != torch.long:
            raise ValueError('exact ten-player long-index roster required')
        if self.check_input_contents and (bool((batch.champion_indices <= 0).any()) or bool((batch.champion_indices >= self.champions.num_embeddings).any())):
            raise ValueError('unknown immutable champion identity')
        if (batch.inventory_indices.ndim != 3 or batch.inventory_indices.shape[:2] != (n, 10)
                or batch.inventory_static.shape != (n, 10, PLAYER_STATIC_DIM)
                or batch.visible_level.shape != (n, 10, 1) or batch.global_public.shape != (n, GLOBAL_PUBLIC_DIM)):
            raise ValueError('public board tensor schema mismatch')
        for values, name in ((batch.inventory_static, 'inventory statics'), (batch.visible_level, 'visible level'),
                             (batch.global_public, 'public globals')):
            _finite(values, name, self.check_input_contents)
        for mask, expected in ((batch.ally_mask, 4), (batch.enemy_mask, 5)):
            if mask.dtype != torch.bool or mask.shape != (n, 10):
                raise ValueError('buyer-first four-ally/five-enemy pooling masks required')
            if self.check_input_contents and (bool(mask[:, 0].any()) or not bool((mask.sum(dim=1) == expected).all())):
                raise ValueError('buyer-first four-ally/five-enemy pooling masks required')
        if self.check_input_contents and bool((batch.ally_mask & batch.enemy_mask).any()):
            raise ValueError('ally and enemy masks overlap')
        embedded = torch.cat((self.champions(batch.champion_indices),
                              self.inventory(batch.inventory_indices, batch.inventory_counts),
                              batch.inventory_static, batch.visible_level), dim=-1)
        players = self.player(embedded)
        pooled = [players[:, 0]]
        for mask in (batch.ally_mask, batch.enemy_mask):
            pooled.append((players*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True))
            pooled.append(players.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values)
        return self.board(torch.cat((*pooled, batch.global_public), dim=-1))


class FullBasketEncoder(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.check_input_contents = True
        self.items = MultisetSum(num_items, 32)
        self.network = nn.Sequential(nn.Linear(55, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

    def forward(self, action):
        if action.item_indices.ndim != 2 or action.static_features.shape != (action.item_indices.shape[0], ACTION_STATIC_DIM):
            raise ValueError('complete observed basket tensor schema mismatch')
        _finite(action.static_features, 'full-action statics', self.check_input_contents)
        return self.network(torch.cat((self.items(action.item_indices, action.item_counts), action.static_features), dim=1))


class FullBasketCritic(nn.Module):
    def __init__(self, num_champions, num_items):
        super().__init__()
        self.state = PublicStateEncoder(num_champions, num_items)
        self.action = FullBasketEncoder(num_items)
        self.head = nn.Sequential(nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state, observed_action):
        if state.batch_size != observed_action.item_indices.shape[0]:
            raise ValueError('observed action is not aligned to its state')
        return torch.sigmoid(self.head(torch.cat((self.state(state), self.action(observed_action)), dim=1))).squeeze(-1)


class TwinFullBasketCritic(nn.Module):
    def __init__(self, num_champions, num_items):
        super().__init__()
        self.q1 = FullBasketCritic(num_champions, num_items)
        self.q2 = FullBasketCritic(num_champions, num_items)

    def forward(self, state, observed_action):
        return self.q1(state, observed_action), self.q2(state, observed_action)


class StateValue(nn.Module):
    def __init__(self, num_champions, num_items):
        super().__init__()
        self.state = PublicStateEncoder(num_champions, num_items)
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state):
        return torch.sigmoid(self.head(self.state(state))).squeeze(-1)


class FullCategoricalActor(nn.Module):
    """Ordinary complete-basket actor; M2 lift is an external comparator only."""
    def __init__(self, num_champions, num_items, action_count=FULL_ACTION_COUNT):
        super().__init__()
        if type(action_count) is not int or action_count < 2:
            raise ValueError('explicit categorical full-action count required')
        self.state = PublicStateEncoder(num_champions, num_items)
        self.head = nn.Linear(128, action_count)
        self.action_count = action_count

    def forward(self, state):
        return self.head(self.state(state))

    def log_probabilities(self, state):
        return torch.log_softmax(self(state), dim=-1)


def set_input_content_validation(module, enabled=True):
    """Use strict validation by default; cache owners may skip redundant syncs.

    Only input-content checks change. Tensor structure, every forward numeric
    operation, output/loss finiteness and gradient guards remain unchanged.
    Disable only after immutable cache validation; re-enable for new inputs.
    This flag is not a parameter and does not change checkpoint weights.
    """
    if type(enabled) is not bool:
        raise ValueError('explicit boolean content-validation setting required')
    for child in module.modules():
        if isinstance(child, (MultisetSum, PublicStateEncoder, FullBasketEncoder)):
            child.check_input_contents = enabled


def make_target(online_critic):
    target = deepcopy(online_critic)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def polyak_update(online, target, online_fraction=.005):
    if not math.isfinite(online_fraction) or not 0 <= online_fraction <= 1:
        raise ValueError('valid online Polyak fraction required')
    source, destination = dict(online.named_parameters()), dict(target.named_parameters())
    if source.keys() != destination.keys():
        raise ValueError('target/online parameter names differ')
    for key in source:
        if source[key].shape != destination[key].shape or source[key].dtype != destination[key].dtype:
            raise ValueError('target/online parameter tensor mismatch')
        destination[key].lerp_(source[key], online_fraction)
    source_buffers, target_buffers = dict(online.named_buffers()), dict(target.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError('target/online buffer names differ')
    for key in source_buffers:
        target_buffers[key].copy_(source_buffers[key])


def expectile_loss(detached_q_target, online_value, expectile=.7, weights=None):
    _vector(online_value, 'online V')
    _vector(detached_q_target, 'observed target Q', len(online_value))
    if not 0 < expectile < 1:
        raise ValueError('valid expectile required')
    difference = detached_q_target.detach()-online_value
    factor = torch.where(difference > 0, expectile, 1-expectile)
    return (sampling_mass(weights, len(difference), difference)*factor*difference.square()).sum()


@torch.no_grad()
def bellman_target(value, packed_nonterminal_next_states, rewards, terminated):
    _vector(rewards, 'rewards')
    if terminated.shape != rewards.shape or terminated.dtype != torch.bool:
        raise ValueError('aligned boolean terminated required')
    count = int((~terminated).sum().item())
    target = rewards.detach().clone()
    if count == 0:
        if packed_nonterminal_next_states is not None:
            raise ValueError('terminal-only batch must not fabricate a next state')
        return target
    if packed_nonterminal_next_states is None or packed_nonterminal_next_states.batch_size != count:
        raise ValueError('provide exactly the packed actual nonterminal next states')
    target[~terminated] += value(packed_nonterminal_next_states)
    return target


def critic_loss(online_q1, online_q2, detached_bellman_target, weights=None):
    _vector(online_q1, 'online Q1')
    _vector(online_q2, 'online Q2', len(online_q1))
    _vector(detached_bellman_target, 'Bellman target', len(online_q1))
    target = detached_bellman_target.detach()
    return (sampling_mass(weights, len(target), target)*((online_q1-target).square()+(online_q2-target).square())).sum()


@torch.no_grad()
def clipped_awr_weights(target_q1, target_q2, frozen_values, temperature=.1, cap=20.):
    _vector(target_q1, 'OOF target Q1')
    _vector(target_q2, 'OOF target Q2', len(target_q1))
    _vector(frozen_values, 'OOF value', len(target_q1))
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite actor temperature/cap required')
    return torch.exp(torch.clamp((torch.minimum(target_q1, target_q2)-frozen_values)/temperature, max=math.log(cap)))


def categorical_actor_loss(full_action_logits, observed_full_action_ids, advantage_weights, weights=None):
    if full_action_logits.ndim != 2 or not full_action_logits.shape[0]:
        raise ValueError('full categorical action logits required')
    _finite(full_action_logits, 'actor logits')
    n, k = full_action_logits.shape
    if (observed_full_action_ids.shape != (n,) or observed_full_action_ids.dtype != torch.long
            or bool((observed_full_action_ids < 0).any()) or bool((observed_full_action_ids >= k).any())):
        raise ValueError('observed full action IDs not in the actor catalogue')
    _vector(advantage_weights, 'frozen advantage weights', n)
    if bool((advantage_weights < 0).any()):
        raise ValueError('advantage weights must be nonnegative')
    logp = torch.log_softmax(full_action_logits, dim=1).gather(1, observed_full_action_ids[:, None]).squeeze(1)
    return -(sampling_mass(weights, n, logp)*advantage_weights.detach()*logp).sum()


def critic_value_step(online, target, value, value_optimizer, critic_optimizer,
                      states, observed_actions, packed_nonterminal_next_states, rewards, terminated,
                      weights=None, expectile=.7, online_fraction=.005, gradient_clip=10.):
    """One bounded Q/V update, not a data loader or a whole training recipe."""
    with torch.no_grad():
        tq1, tq2 = target(states, observed_actions)
        target_min = torch.minimum(tq1, tq2)
    value_optimizer.zero_grad(set_to_none=True)
    loss_v = expectile_loss(target_min, value(states), expectile, weights)
    loss_v.backward()
    value_gradient = torch.nn.utils.clip_grad_norm_(value.parameters(), gradient_clip, error_if_nonfinite=True)
    value_optimizer.step()
    targets = bellman_target(value, packed_nonterminal_next_states, rewards, terminated)
    critic_optimizer.zero_grad(set_to_none=True)
    q1, q2 = online(states, observed_actions)
    loss_q = critic_loss(q1, q2, targets, weights)
    loss_q.backward()
    critic_gradient = torch.nn.utils.clip_grad_norm_(online.parameters(), gradient_clip, error_if_nonfinite=True)
    critic_optimizer.step()
    polyak_update(online, target, online_fraction)
    return {'value_loss': float(loss_v.detach()), 'critic_loss': float(loss_q.detach()),
            'value_gradient_norm_before_clip': float(value_gradient),
            'critic_gradient_norm_before_clip': float(critic_gradient),
            'terminal_transitions': int(terminated.sum()), 'observed_actions_queried': states.batch_size}
