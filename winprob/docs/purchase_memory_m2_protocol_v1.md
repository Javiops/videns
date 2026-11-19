# Purchase memory + M2: fixed displayed-reranking comparison

2026-10-01, declared after the internal3350-game screen passed and before any
new859-game prediction/target comparison. Scope: exposed development, not
confirmation. User criterion remains the actual observed purchase.

Freeze the internal memory recipe: two state-only checkpoints, normalized
penultimate128 activations, same-champion32 nearest DISTINCT training games,
distance temperature=max(last_distance/4,1e-6), no tuning or new training.
Each encoder queries its own original opposite-fold memory. Both memories
contain FIT only; all859 query games must be outside both checkpoint training
sets and all memory game IDs. Reuse all392448 admitted FIT states across the
two partitions. Check current reservations and exact artifact digests.

For each development query, compute R0 and R1 and set R=(R0+R1)/2. The candidate
distribution is 0.5 P_M2 +0.5 R. Rank ONLY M2's original up-to-three legal main
options using these masses; ties preserve original order. No option is added,
removed or altered. Thus this comparison concerns first-choice accuracy; its
top3 set is unchanged. It does not test new candidate generation or an unrestricted
memory decoder. An empty memory for one encoder contributes P_M2 for that half;
unavailable public features retain the exact original M2 ordering.

Use a negative control with the same independently shuffled FIT labels as the
internal screen (within champion/patch/three-minute bin, seed20261001+fold).
It keeps neighbor identities and weights unchanged. No learned gate, weight
search, threshold search, nearest-neighbor count search or checkpoint selection.

Inputs: original859 contexts20260930-v1, M2 reference20260930-v3, native19
audit20260930-v3, frozen full/main catalogue and prior BC export solely to
validate the original reference lineage. Existing winprob ordinary input
validators verify these original manifests; their historical protocol bindings
are preserved. Bind THIS new protocol separately; do not relabel the prior BC
experiment or its results. Source native code/models/data remain unchanged.

Persist complete new option orders and neighbor receipts BEFORE opening the
separate observed-target file. Reuse the existing count-preserving target
validator/evaluator semantics: primary ward-stripped main-basket top1;
secondary strict full-basket and typed-main exactness. Unknown catalogue targets
remain misses, and feature fallbacks remain in the859-game denominator.
Reproduce original M2 main/complete results exactly:454/859 and354/859.
Verify zero candidate-object changes and native18/native19 feasibility using
the bound prior receipts. Test forbidden-feature mutations for every compiled
query. No labels, outcomes, future fields or historical inferred roles may
change query features.

Use2000 paired whole-game bootstrap draws, seed20261001. Declare useful
development evidence only if main top1 improves at least3pp over M2 with95%
lower bound>0, also beats the shuffled control with95% lower bound>0, and all
integrity/legality gates pass. Publish corrected/broken counts and all secondary
costs. Secondary gains cannot rescue a failed primary. No retuning this recipe
on the859 scores. Prior exposure means even a positive result needs separate
confirmation and source-native integration review before practical deployment.

Resources: same bounded streaming CUDA/CPU1 guards as purchase_memory_protocol_v1,
20-minute worker/30-minute launcher, immutable new run. No source runtime imports,
raw events, rewards, sealed payloads, promotion, shared databases or index edits.
