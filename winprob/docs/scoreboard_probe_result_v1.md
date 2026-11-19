# Public scoreboard: the fixed purchase-prediction recipe failed

2026-10-01. The completed real/zero/shuffled-KDA comparison does not justify
an M2 trial. Implementation verification passed; the predictive gate failed.
The broader practical-improvement goal remains active and unachieved.

## Population and result

Exactly3350 fixed focal decisions from distinct games, out of4001 admitted
FIT games;651 games lack the original qualifying focal episode. Two whole-game
folds train on the opposite fold. This is the already exposed internal screen,
not the859-game ordinary M2 comparison and not independent confirmation.
The target is the recorded retained main-purchase multiset, with counts.

First-purchase exactness: real scoreboard1515/3350 (45.2239%), matched zero
1500/3350 (44.7761%), within-team shuffled1497/3350 (44.6866%), and the unchanged
stronger memory reference1584/3350 (47.2836%). Paired2000-draw whole-game
bootstrap intervals use the fixed seed20261001.

- Real minus zero: +0.4478pp,95% interval[-0.6567,+1.6418]pp;206 corrected,
  191 broken. Top3 difference-0.4478pp [-1.5522,+0.6873].
- Real minus memory: -2.0597pp [-3.4634,-0.7164];239 corrected,308 broken.
  Top3 difference-2.6567pp [-3.9104,-1.4030].
- Real minus shuffled: +0.5373pp [-0.5672,+1.7015]. This unresolved attribution
  comparison cannot rescue the failed primary.

On3169 nonempty-main purchases, real-zero accuracy differs by+0.4733pp
[-0.6937,+1.6944]. The required positive-lower-bound guard also fails.
Real-score main top3 is69.1642%, versus69.6119% zero and71.8209% memory.
NLL is2.354276, versus2.346538 zero and2.282992 memory; lower is better.

The primary required at least3pp improvement over BOTH zero and memory, with
positive paired lower bounds, no mean top3 loss versus memory and a positive
nonempty-purchase lower bound. The failed result closes this fixed recipe.

## What was controlled and verified

The source exporter recovered other-player cumulative KDA at the original
stored snapshot for all392448 FIT states. Scores are public cumulative values,
not exact death recency, future observations or hidden enemy gold. Synthetic
no-buy construction is a reason to exclude recency. The buyer's added channel
is zero. Within-team shuffles preserve the KDA triples and team aggregates.
Availability is verified; useful conditional information is not established.

Six models completed20epochs each,92040 Adam updates in total. All arms have
the same164416-parameter encoder, including a192-parameter score projection
initialized exactly zero, and1220469-parameter categorical head. Initial tensors,
epoch orders and architecture match within each fold. Complete predictions
are unchanged by held-out target mutation. No checkpoint or epoch selection.

The independent functional auditor revalidated input lineage, every normalized
score and permutation, training IDs/orders/counts/settings, frozen reference
arrays, all aggregates/slices/paired intervals and the nonempty-purchase guard.
It replayed all10050 neural queries from the six saved checkpoints with the
original CUDA/batch256/store path: zero top3-cell mismatches and exactly zero
target-logp error. Eleven own feature/model tests and seven source projection
tests passed. These checks establish implementation consistency, not benefit.

Result: `data/runs/scoreboard-probe-20261001-v1/experiment/report.json`, SHA256
91286915a4ad16970031a9541f96efc214cbb867d6b7a3a03e60b24f4e342a17.
Audit: `data/runs/scoreboard-probe-audit-20261001-v1/audit/report.json`, SHA256
6a9b944137548beeab66404b7be8a56360218a6bcaccb37d0f2fde4d773907ef.
Consumed protocol: `scoreboard_probe_protocol_v1.md`; feature lineage and
source receipts: `scoreboard_information_result_v1.md`.

## Costs, limits and decision

Source extraction464.92s; own feature preparation26.0s; training/evaluation
980.85s (16.35minutes); functional audit190.70s. CPU1 and idle RTX4060.
Training peak Torch allocation961451520bytes, peak physical working set
1670987776bytes (1.56GiB), maximum sampled private bytes3386388480.
Minimum sampled host availability1455337472bytes, above the805306368-byte floor.

The original v1 audit waited for its3.25GiB startup gate and started after RAM
became available. A proposed2.5GiB replacement was blocked by the controller's
live-worker guard BEFORE any stop or mutation. It was NOT applied. Both
training and audit finished and their scheduled tasks unregistered normally.

The limited learner/recipe failed. This does not prove that public KDA cannot
help any other model. It does not test native M2 directly, establish live API
parity, identify causal purchase value or justify a deployment. Do not extend
this recipe, choose its shuffled control or retune it on these observed scores.
No new859-game evaluation, sealed payload read or served-policy change occurred.
All1911 own confirmation games remain closed. Original report index SHA256
94832c5dfe951b9728cd5bf56929977703a2682fcc0b0245979c776626b32a46 is unchanged.
