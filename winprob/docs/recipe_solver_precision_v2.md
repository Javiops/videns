# Recipe correction: stronger numerical stopping, same estimator

Declared2026-10-03 after reproducing the first failed training solve, before
any held-out prediction or purchase-accuracy comparison.

The diagnostic uses the same raw arm, evaluation fold0,28112opposite-fold states,
game weights, training-only standardizer and lambda.01. L-BFGS-B reports success
after79iterations because relative objective change reached ftol1e-12, but the
maximum absolute gradient is2.115549009e-6, above the required1e-6gate. Objective
0.8797179570121145 includes the ridge penalty; this is not an accuracy result.
Three directional finite-difference checks have errors at most6.512e-11,
supporting correctness of the implemented gradient on these actual inputs.
No evaluation prediction or metric was computed. Diagnostic artifact:
`data/runs/recipe-solver-diagnostic-20261003-v1/diagnostic/report.json`.

Change only the solver's function-change tolerance from ftol1e-12 to ftol0.
Keep L-BFGS-B, clean zero initialization, gtol1e-7, maxiter300, maxls50, lambda.01,
the identical objective/gradient kernel, data, controls, standardizers and all
statistical gates. The original core remains untouched; a new solver wrapper
uses its same objective. Retain the convergence/gradient acceptance gate and
record full solver settings and evaluation counts in every checkpoint.

This is numerical refinement of a fixed strictly convex penalized likelihood:
there is one exact optimum, independent of the route used to approach it.
Neither new features nor a new statistical/hyperparameter model is selected.
The amendment supersedes only the ftol setting in the original protocol;
the consumed original protocol remains byte-identical. The45minute worker and
sustained timing method from runtime_v3 remain. A new v4 execution must reproduce
the same original feature digests. All failed runs and unqualified diagnostic
coefficients are preserved; diagnostic coefficients are NOT used as a warm start.

The numerical convergence gate must pass for all six models before prediction.
If it still fails, publish the full numerical receipt before stopping; do not
read held-out accuracy or relax the required gradient after seeing a score.
