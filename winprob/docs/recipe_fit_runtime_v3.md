# Recipe fit: sustained CPU timing, unchanged scientific recipe

2026-10-01. Both v1 and v2 consumer attempts stopped in preflight BEFORE
load_targets or any real coefficient fit. All input/receipt/zero-prior checks
passed. Their feature digests agree; no purchase score has been inspected.

The v2 maximum single-pass measurement was0.1058672s, while its other eleven
measurements were0.0364251-0.0567836s. Extrapolating that one pass to every
function evaluation projected2466.732s, beyond80% of2636.492s remaining. These
measurements establish timing variability; its exact OS scheduling cause was
not isolated. Raising time limits repeatedly is not the measurement fix.

Keep the2700second (45minute) hard worker budget introduced in runtime_v2,
all resource guards, the six fits, fixed300iterations/maxls50/lambda.01, the
same input hashes, controls, splits, standardizers, metrics and success gates.
Change ONLY the timing estimator: for each of the six fold/arm geometries,
run two untimed objective/gradient warmups followed by20consecutive timed
passes. Time the whole20-pass block, divide by20, and use the MAXIMUM of those
six block means in the unchanged1.5*time*300*8*6+180 projection. All passes use
the synthetic category0 target. No fastest-run selection or target-dependent
timing. The same80%remaining-budget gate and hard limit still apply.

The new v3 execution must reproduce v1's complete locked feature digests before
any real target access. Preserve v1/v2 code, input locks, preflights and failure
logs. The statistical protocol stays byte-identical at
e87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e.
This operational change has not been informed by actual labels or accuracy.
