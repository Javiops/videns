# Frozen BC20 and M2: limited but real prediction diversity

2026-09-30. Descriptive analysis of the existing 158 exposed revision pairs
from 144 development games. This does not score a mixture, select a mixing
coefficient, fit a selector, or inspect new games. The numerical artifact is
[revision_complementarity_v1.json](../data/reports/revision_complementarity_v1.json).
Its input is the unchanged completed IQL challenge prediction file, SHA256
`dce9fa2bc0c01afe0ce11d6a566ba9c1478fc97a8c135d9c047be2d180da0bbb`.

The models agreed on 138 of 158 pairs. Eighty pairs were correct under both
models (50.6410% of evaluation weight); 58 were wrong under both (36.5385%).
BC20 alone was correct on 14 pairs (8.9744% weight), while M2 alone was correct
on six (3.8462%). The 20 disagreements occurred in 20 distinct games and
represented 12.8205% of evaluation weight. BC20 was correct on 70% of these
disagreements, an exposed descriptive observation, not a learned selection
rule or a separately confirmed subgroup effect.

A label-aware oracle choosing whichever model was right would score
63.4615%. This **unattainable descriptive ceiling** is only 3.8462 percentage
points above BC20 and 8.9744 points above M2. On this particular challenge,
even perfect selection between the two cannot deliver a five-point gain over
BC20. A probability pool that preserves the models' common pair ordering also
cannot repair the 58 shared errors. This says nothing conclusive about an
ordinary-purchase population with a different sampling scheme.

On BC20-only-correct pairs, median B-minus-A log odds were −0.4779 for M2 and
+0.5458 for BC20. On M2-only-correct pairs they were +1.0915 and −1.4000.
The magnitudes come from different learned distributions; they are not
comparably calibrated confidence scores. These summaries provide no justified
confidence threshold or mixture coefficient. In particular, pair odds alone
do not determine a linear mixture's odds without each model's individual
candidate masses.

The existing paired whole-game interval for BC20−M2 is still
−0.6330 to +10.5599 pp. Diversity therefore makes a fixed mixture a reasonable
secondary diagnostic if specified before the next evaluation, but it is not
evidence of a useful mixture. The primary ordinary-behavior test should keep
its independent population, frozen candidate, strong reference and full
denominator—including unsupported observed targets. A top-three reranker
can only reorder candidates already proposed by M2; its ceiling and coverage
must be reported separately from a full-catalogue actor.

For a source-safe scoring boundary, the own-project scorer should consume
only certified prestate, native patch and opaque row identity, and emit all
13,166 full-action masses plus their 9,333 exact main-group sums. Bind the
unchanged BC20 checkpoint, feature catalogue/statics, code snapshot, runtime
versions and row order. Keep labels outside the compiler/model input, prove
full-distribution label/future-field mutation invariance on every row, and
leave native legality/display semantics with the source-owned producer.
Unknown targets stay in evaluation; they must not determine which contexts
receive predictions. This is an interface recommendation pending the final
ordinary-evaluation bundle contract, not a newly run scorer or protocol change.
