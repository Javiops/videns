# Fixed closed-history probability-ratio comparison

Declared2026-10-03 before new proxy forecasts, posterior evidence or purchase
scores. The previous route and raw memory-mixture recipes remain closed.
This experiment tests a new, history-dependent relative correction to M2.

## Population, models and chronological boundary

Use exactly the2900original16.17focals and15573unique closed real past visits
from bayes-proxy-prefix-20261003-v1, report SHA
89ff1f75c4031bd88c22ea31050e157eed2f50cb6adfed9a37b0f0d309aa1b35.
Keep287zero-history queries and disclose618unavailable original focal games.
No current/future query label may enter its own forecast or posterior. Remove
synthetic SAVE before history construction; retain real ward-only visits.
Histories contain all real visits ending strictly before the fixed cutoff.

Frozen causal state proxies: HistoryProbeActor state checkpoints from
history-probe-20260930-v3/experiment, fold0 SHA
e050c340298e5495fb966378d33cd133b558cf6302f673db5d8a14188a84364a,
fold1 SHA5381e72e1606f773817963c317652ecb2d6b568316ae189a63261df4565a1440.
Validate checkpoint training IDs equal the opposite fold of the original4001
FIT-game cache. No fitting, optimizer, encoder, architecture or epoch selection.

For each fold, memory bank is EVERY opposite-fold state, with its training
label. Keep frozen k32, same champion, one nearest state per distinct training
game, squared distance between normalized128-dimensional penultimate vectors,
immutable state-index tie break, temperature max(d_last/4,1e-6), and the
.5state/.5retrieval mixture. Empty retrieval returns state exactly. Reuse the
audited purchase_memory_v1 implementation; do not tune these parameters.

Bank encoding: ascending state indices, CUDAfloat32 model inference in256-row
batches, noTF32, deterministic configuration, eval mode. Query encoding:
ascending union of that fold's current/past indices, same256-row batches.
The new grouping is fixed here; no claim of bit-identical historical3350-query
batches is made. Audit must replay this experiment's exact batches. Normalize
embeddings/probabilities in float64, as in the original memory calculation.

Use a state-only feature store: no label array or getter in the neural store.
Do not instantiate HistoryDataset, which eagerly validates held-out targets.
In scored fold k, label access is restricted to either opposite-fold TRAIN
rows or k's published strictly closed past indices. Never interpret k's focal
or subsequent action values. The same game may be a legitimate TRAIN game for
the other fold, so this boundary is per recipient fold, not a false global
claim that no OOF query label ever appears in any training bank. Full files
are hashed; interpreted indices are checked before access. No rewards,
terminal arrays, source runtime, raw/DB,859development or1911confirmation reads.

## Fixed posterior and four displayed policies

Let S be the state forecast and M=.5S+.5R the memory forecast. For each closed
past real visit, generate S/M from its PRE-state before reading that visit's
permitted label. Record its two observed log likelihoods, forecast digests,
neighbor IDs/weights/distances and stable physical identity. Equal initial
weights pi=(.5,.5), no forgetting, clipping, temperature or fitted hyperparameter.

At each current query use the frozen complete native M2 prior p0, normalized
in float64. Its exact original float32 is retained for native replay. Only p0
is reused from the closed route artifact; no route correction/model is reused.
Source matrix SHA6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2.

Arms, all fixed before forecast generation:

- m2: unchanged native prior and decoder.
- history: Bayes posterior w from all own closed likelihood pairs, then
  q proportional to p0 * (w_S S+w_M M)/(.5S+.5M).
- scrambled: same evidence magnitudes, visits and count; swap S/M likelihood
  identities independently per past visit when the first byte of SHA256 of
  'bayes-proxy-v1/sign/'+match_id+'|'+participant_id+'|'+start_ms has low bit1.
  This breaks evidence orientation; it is not a causal randomization of humans.
- static_memory: same ratio with w=(0,1) wherever real history is available,
  measuring whether always favoring memory explains the effect. It uses
  pi=(.5,.5) on the same287empty histories so opportunity/fallback matches.

All three corrections retain EXACT normalized p0 on empty histories; also
when current proxy distributions agree. Require positive proxy support on
native support; unsupported/impossible inputs fail without epsilon or dropped
queries. Native decoding uses original float32 M2 whenever a correction is
exactly neutral, preserving ties. No cumulative proxy-regret guarantee is
claimed for anchored M2, single-query correctness or the native displayed set.

## Freeze, decode, score and audit

Lock all input identities/checkpoints/code before forecast generation. Persist
all forecasts/evidence/posteriors/distributions before opening scoring targets.
No individual player's current action may enter neural inference, retrieval
selection, posterior updates or prediction. Validate chronology, synthetic-SAVE
invariance and label-access permissions, including current/future rejection.

Decode all four full9333-class distributions through unchanged native M2
MixturePolicy.options (CUDA top128, original wallet, up to3distinct legal
actions, original SAVE fallback). Require exact M2 option replay and per-query
forbidden-field mutation invariance. Audit all options again in a clean native
16.17 process; no filtering/replacement after either world audit.

Primary: displayed counted-main first exactness, history-M2. Secondary: top3,
complete/typed first, nonempty-main first, full-catalogue NLL, corrected/broken,
candidate changes, posterior distribution and costs.2000paired whole-game
bootstrap draws(seed20261003), one original focal per game. No slices rescue
the primary. The useful internal gate requires at least3pp first gain and
95%lower>0 versus M2; positive first95%lower versus BOTH scrambled and
static_memory; positive nonempty history-M2lower; no mean top3 or catalogue-NLL
worsening; zero native17/18legality failures for every arm; all integrity/replay
audits pass. No control promotion, posterior/neighbor sweep, new window or
forward trial after a failed primary. This exposed retrospective16.17 screen
with native16.18 M2 is not forward/live confirmation or causal purchase value.

Audit: checkpoint/fold identities, public inputs, all bank/query embeddings in
the fixed batches, all retrieval receipts and past likelihoods, posterior and
anchored distribution replay, native M2 reference, all metric/interval/gate
calculations. Independent native source receipts remain source-owned. Preserve
all versions and the original index, no serving change or automated action.

## Resources

CPU1/idleCUDA3072MiBfree, startup RAM2.5GiB/live0.75GiB, no League or competing
protected job, hidden one-shot worker,180min. No whole-cache preload; stream
state batches, normalize bank embeddings by batch, and write large NPY outputs
with bounded seek/write instead of keeping full output mappings resident.
Before past labels/posteriors, measure first1024training embeddings and first64
query state-plus-retrieval forecasts per fold (training labels allowed). Use
2*max_bank_seconds_per_row*392448 +
2*max_query_seconds_per_row*18473 +600seconds, requiring<=80% of remaining
time. Freeze timing result before new past evidence. Do not shrink populations
or publish partial scores if the resource/identity gate fails.
