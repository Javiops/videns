# Purchase memory: internal signal, M2 gain unresolved

2026-10-01. The recorded historical purchase is the correct supervised target.
Three frozen experiments completed. No policy is promoted and the breakthrough
is not established.

## Internal mechanism screen

On the same3350 fixed whole-game OOF decisions, augmenting the existing
state-only classifier with a memory of similar training purchases increases
main-basket top1 exactness from43.9104% to47.2836%. The paired gain is3.3731pp,
95% game-bootstrap interval[2.3284,4.5672]pp:244 corrected and131 broken.
Top3 increases from69.4627% to71.8209%; target NLL falls from2.361116 to2.282992.
The shuffled-memory control is43.2239% top1. The candidate clears the declared
internal gate against both controls. Original state predictions reproduce with
zero top1/top3 mismatches and zero target-log-probability error.

The method reuses two frozen encoders, queries32 nearest distinct training
games of the same champion, and mixes the retrieved purchase distribution with
the classifier50/50. It is an adaptation of the semi-parametric retrieval idea
in [Khandelwal et al.](https://arxiv.org/abs/1911.00172), not a new theorem or a
claim that language-model gains transfer automatically to League.

This population is an already exposed internal subset of FIT:4001 games in
the ledger,651 without an eligible fixed focal episode,3350 scored. Each
query's whole game is excluded from its encoder training and retrieval memory.
The screen has no native legality decoder and the reference is NOT M2.

Evidence: `data/runs/purchase-memory-20261001-v1/experiment/report.json`,
SHA256 e6a7c38f099e5312cc10bacc81477217fbd8071df73f7daedc04e6e54c662ba9.
Audit: `data/reports/purchase_memory_audit_v1.json`. It independently checks
105967 neighbor receipts, the mixture target probabilities and all reported
metrics/paired intervals. Maximum recomputation logp error3.56e-15.
It does not independently replay the encoder or nearest-neighbor search.

## Direct M2 comparison

The separately declared adaptation combines P_M2 with the mean of the two
FIT-only memories, still50/50, and orders only the original legal M2 options.
On the same859 exposed16.19 development games:

- M2 main top1:454/859 =52.8522%.
- M2 plus memory:464/859 =54.0163%.
- Gain:+1.1641pp, paired95% interval[-0.3492,2.5611]pp;26 corrected,16 broken.
- Shuffled-memory control:458/859 =53.3178%. Real minus shuffled:+0.6985pp,
  interval[-0.6985,1.9790]pp.
- Complete-basket top1:354/859 to360/859; +0.6985pp, interval[-0.5821,1.9790]pp.
- Main top3 stays591/859 because candidate membership is unchanged.

The declared M2 gate FAILS: the gain is below3pp and both primary intervals
include zero. The internal positive result cannot rescue this failed practical
comparison. There were zero feature fallbacks; all859 forbidden-feature
mutation checks passed. Exact original M2 first/full/top3 results reproduce.
All existing candidate objects remain unchanged and retain their bound native18
and native19 feasibility receipts, conditional on the estimated wallet.

Evidence: `data/runs/purchase-memory-m2-20261001-v1/experiment/report.json`.
The completed audit `data/reports/purchase_memory_m2_audit_v1.json` passed.
It independently recomputes retrieved-label aggregation,
the1718 new candidate orderings, actual multiset credits, movement counts and
paired intervals; it does not replay the encoder/search or native legality.

## Full-catalogue candidate test

The follow-up `purchase_memory_candidates_protocol_v1.md` kept exactly the
same distribution and weights, then allowed the unchanged source-native M2
decoder to search all 9,333 groups through its original top-128 legal walk.
All previously restricted candidate scores reproduce. No encoder, nearest
neighbors, weights or observed labels were adjusted.

On the same 859 exposed development games, first main exactness remains
464/859 (54.0163%), versus M2's 454/859. Gain +1.1641pp, paired 95% interval
[-0.3492, 2.5611]pp; 27 corrected and 17 broken. Against the restricted memory
reranker, it corrects one and breaks one: no net first-choice gain.

Displayed main top-3 coverage declines from 591/859 (68.8009%) to 582/859
(67.7532%): -1.0477pp, interval [-2.3283, 0.2328]pp; 12 corrected and 21 broken.
Candidate membership changes on 362 games. The full shuffled control scores
455/859 first exact and 573/859 top-3. Complete-basket first exactness remains
360/859 versus M2's 354/859; it cannot rescue the failed primary.

All 2,516 memory options are feasible in both pinned 16.18 and native 16.19.
Original M2 options replay with zero differences; all 2,577 arm/context
target-field mutation checks pass. The displayed candidate result fails the
declared gate, including its no-mean-top3-loss guard. This closes the fixed
recipe; do not tune it on these scores.

Evidence: `data/runs/purchase-memory-candidates-20261001-v1/evaluation/report.json`,
SHA256 f71c8ea6fa848895d896e8738338fd742af7ced78faf3b390707c5184f489a0a.
Independent audit `data/reports/purchase_memory_candidates_audit_v1.json`
passed, SHA256 c6e71a675b90bf3e4234fd3010f2a700cc47bbf6d1d3aa28cb0f42255619ab01.
It recomputes every observed target multiset, accuracy, movement count, paired
interval and gate, and checks immutable source decoder/legality receipts.
It does not independently rerun the native economic engine.

Source-owned adapters exchange versioned files only. The new
`build_tracker/scripts/decode_purchase_memory_v1.py` invokes the unchanged
MixturePolicy.options and purchase_sequence in separate pinned/native processes;
it edits no frozen runtime, model or database. Source artifacts are under
`build_tracker/data/exports/winprob-memory/purchase-memory-candidates-20261001-v1`.
Distribution assembly took 14.6s, pinned decoding 23.1s and native19 audit 4.9s,
CPU1 with no GPU computation or retraining.

## Decision and limits

Retain the positive mechanism evidence and the unresolved M2 comparison.
Do not retune weights/neighbors/checkpoints on these859 scores, reinterpret
the internal population as an M2 benchmark, or promote the candidate.
Neither experiment reads confirmation/test payloads or outcomes. No new
training, collection, source runtime import, frozen source edit or index edit.

Both restricted reranking and full native candidate generation have now been
tested. Neither establishes a useful M2 gain. The positive internal result is
conditional on the weaker state-only classifier and cannot be advertised as an
M2 improvement. This does not disprove every retrieval representation/learner;
it rejects the tested fixed recipe as a practical replacement. The user's goal
remains unmet, and no confirmation games have been opened.

Measured worker times:80.9s internal and90.6s M2 comparison, excluding audit
and setup. CPU1 with bounded CUDA inference; no optimizer steps. The internal
eight synthetic tests pass (game exclusion, vote deduplication, ties, normalized
mixture, empty-memory fallback, shuffle isolation and null-gate behavior).
