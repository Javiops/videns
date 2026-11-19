# Recipe-transition residual: implementation and feasibility verified

2026-10-01. No real purchase model has been fitted and no improvement over M2
has been measured. This result justifies constructing the fixed real-data
experiment in `recipe_transition_probe_protocol_v1.md`.

The own convex likelihood/receipt module passes11synthetic tests: normalized
coarsened mass, equivalence to a full-catalogue exponential tilt with grouped
tail, zero-correction identity, target-free inference, OTHER examples included,
finite-difference gradient and Hessian-vector parity, strictly positive ridge
curvature, recovery of a known identifiable correction, option permutation/
padding, repeated counts, consumed-component subtraction and malformed receipts.

Native source probe v2 completed in13.143s in a clean deterministic CUDA process
using unchanged M2/native16.18. Sixteen distinct synthetic states were scored
112times including warmups, repeated batches and mutation controls. All six
actual model-input tensors are identical after ignored-label/future mutation;
both same-input and mutated-input full probabilities/options replay exactly.
Its48displayed candidate receipts and6explicit fixtures were consumed by the
own independent verifier; all34used item-stat records agree with the pinned
own16.18 static artifact.

Example verified through native execution: from Boots1001 and Cloth Armor1029,
buying Steelcaps3047 costs600gold. Its purchased-item descriptors say25armor
and45movement speed, while the net descriptor adds10armor and20movement speed
after subtracting the two consumed components. This does not simulate attack
reduction passives, effective combat stats or victory probability.

Three warmed batches of16synthetic native decisions took0.903646,1.059247 and
1.009661s; median63.10ms/row. Replay of48native receipts took10.61ms total.
Peak Torch CUDA allocation164663296bytes. These synthetic states do not
establish real-corpus throughput or live latency. A separately selected200-row
real FIT cost gate must pass before a full native release.

The own CPU1 synthetic676-parameter fit on8192rows, including2028OTHER labels,
converged in9iterations in0.0631s; max absolute gradient9.23e-8. One objective+
gradient pass took about4.5-5.3ms. Its input arrays occupy7077888bytes. These
are implementation/cost measurements on synthetic inputs, not purchase scores.

Source report:
`build_tracker/data/exports/winprob-recipe-transition/recipe-native-probe-20261001-v2/report.json`,
SHA256 ad1bf78aeec9929367a3f5e3bb364fb21f3f171ceb82f77c0a8d5b1a669dbe6c.
Own audit: `data/runs/recipe-transition-synthetic-20261001-v1/audit/report.json`,
SHA256 6bd020579b7c9f714becef0774b55a4ce195d1c87c44c011989ad5a3b725f372.

The preserved v1 synthetic source probe failed its exact floating-point
mutation comparison after native scoring. It made no real-data/model fit.
Version2 added deterministic CUDA/CUBLAS configuration, a same-input replay
control and direct model-input tensor parity, keeping all semantic fixtures
and M2 code/weights unchanged. All v2 exact checks passed. Nondeterministic GPU
computation is the supported explanation; the particular kernel responsible
was not isolated. Do not describe v1 as a successful complete probe. Both
scheduled tasks have ended and unregistered; no source process was stopped.

The real-data selection is now published using identity/time metadata only:
3518original16.17 FIT games,16states each=56288states;28176/28112by fold.
The2900original focal decisions are1449/1451by fold;618original focal games
remain unavailable. Reproduced the old focal selection from its original
hash rules. The M2 train-ID digest matches its6354-game package declaration,
with zero overlap. No purchase/action arrays, source trajectory payloads,
databases, outcomes or sealed payloads were opened by the selector.

Selection manifest:
`data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json`,
SHA256 979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6.
Selection data SHA256
97b7842e8811b3b4ed3baf4cc2918185c98061d8997589b52ba11459213b4191.
Consumed real protocol SHA256
e87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e.

Before publication, code inspection corrected one source-contract assumption:
the FIT exporter deliberately removed reference_compatibility. The real
protocol therefore requires an indexed, digest-bound source role sidecar from
the same admitted games. No role is fabricated, no mixed export imported and
no final-role field enters the new residual. At the end of that stage, the
sidecar and real M2 prediction release had not been implemented. All completed code,
protocols and artifacts are now consumed versions and must be preserved.

Execution update2026-10-03: all source exports, six real correction fits and
the exact functional audit have completed. The comparison is negative:
1481/2900net versus1483/2900M2 main-first hits. This feasibility result is not
evidence of predictive benefit. The fixed recipe is closed; full results and
the numerical stopping repair are recorded in `recipe_transition_result_v1.md`.
