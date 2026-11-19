# Exact unordered purchase likelihood: design and feasibility stage

2026-10-01. The fixed purchase-memory recipe is closed after its native M2
comparisons failed. This is a different learning objective, not a new memory
weight, filter or candidate reranker. The player's retained purchase multiset
remains the supervised target.

## Verified distinction from the existing models

Read `build_tracker/scripts/train_basket_ab.py`: BasketEncoder pools nonlinear
item/count representations; BasketNet gives whole catalogue baskets a joint
score. The older item model uses independent presence/count losses and greedy
assembly. Current own-project HistoryProbeActor has a categorical main-group
head. None of those inspected paths marginalizes a sequential probability over
all distinct orderings of a purchase multiset. This is not a claim that M2
ignores interactions or treats every item independently.

## Mathematical construction

Let x be the predecision state and c the item-count vector of a retained MAIN
purchase. A transition model q_theta(j | x, s) sees the current generated
multiset s, and predicts the next item j or STOP. The prefix representation is
permutation invariant, includes multiplicities, and has no recurrent hidden
state depending on a particular order. This restriction makes exact dynamic
programming possible.

Set A(0)=1. For each nonempty submultiset s<=c:

    A(s) = sum over distinct j with s_j>0 of
           A(s-e_j) * q_theta(j | x, s-e_j).

Then P_theta(c|x) = A(c) * q_theta(STOP | x,c). At maximum length six,
STOP is forced. The empty multiset uses q(STOP|x,0). No factorial correction
or factor s_j belongs in the recurrence: identical copies are indistinguishable
and each distinct item sequence must be counted once. Computation uses logsumexp.

This sums the probabilities of every distinct sequence whose multiset is c.
All bounded ordered sequences terminate and map to exactly one multiset, so
the distribution over ALL multisets of length<=six is normalized. Restricting
scoring to a finite catalogue leaves mass outside it; that mass must be
reported, not silently treated as zero. A probability conditional on catalogue
membership must be explicitly identified as such.

The proposal builds on autoregressive factorization and the output-order issue
described in [Vinyals et al., Order Matters](https://arxiv.org/abs/1511.06391)
and [Uria et al., NADE](https://jmlr.org/papers/v17/16-272.html). The particular
multiset recurrence above is our application of the law of total probability;
do not attribute this exact implementation or a League benefit to those papers.

## Bounded catalogue inventory, read without game payloads

The existing frozen 9,333 main groups use 196 item IDs, lengths0..6 and at most
32 submultisets per target. Across all groups there are 10,037 unique prefix
multisets (including the empty one), 60,625 per-target lattice nodes and
85,274 per-target edges; at most120 distinct orderings per target. The catalogue
digest remains14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8.
These are catalogue properties, not purchase frequencies or a model result.

## Stage zero

Implement exact log-likelihood and gradients, test against exhaustive ordered
enumeration (including repeated items, STOP and forced termination), verify
normalization on a complete small vocabulary and order invariance. Benchmark
synthetic training batches and full-catalogue inference on the idle GPU, using
the real catalogue topology but NO game states/labels/outcomes or model fit.

Use a128-dimensional context,64-dimensional count-weighted item embedding,
8-dimensional size embedding, and a200->128->197 conditional head (196items
plus STOP). Predeclare seed20261001, batch256, one CPU thread, CUDA float32,
no TF32, bounded chunks for catalogue inference. No hyperparameter sweep.
The purpose is to establish correctness and cost before declaring any real
training comparison. STOP is not an observed chronological shop exit event:
this is a probabilistic factorization of retained baskets, not a claim about
the physical transaction ordering or passive item transformations.

No legality masks on observed training targets inferred from estimated wallets.
Any later practical comparison uses unchanged source-native legal decoding.
No 859-game target analysis, raw data, seals, new collection, source edits or
index changes in this stage. A real training/evaluation experiment requires a
separate fixed protocol and disjoint whole-game partitions. Stage zero is not
the breakthrough, a forecast of gains, or proof of improvement over M2.
