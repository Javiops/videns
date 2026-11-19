# Winprob research contract

Read README.md and docs/protocol_v1.md. Work only inside this sibling project
unless the user requests a specific integration. Never import code from other
videns.gg projects; exchanges are versioned artifacts with digests.

- No sealed test payload reads, even for unsupervised features or calibration.
  Admit inputs using explicit IDs and all existing reservation metadata.
- No future interpolation. A decision-time feature must be measurable at its
  cutoff. Offline omniscient inputs are explicitly marked and never live-safe.
- Report measured numbers with metric, population, game count, and report path.
  A development result is not confirmatory evidence.
- Fit, calibrate, select and confirm on disjoint games. Keep all perspectives,
  states and labels from a game together. Bootstrap by game.
- Prediction and observational association do not establish causal action value.
  A learned imitation policy is not a known historical logging propensity.
- Never change frozen code, data or scheduled jobs in sibling projects.
- No GPU work during a game, no heavy work beside the benchmark/reconstruction,
  no long work attached to the agent shell, no automatic promotion.
- Use streaming input, one CPU thread, and a RAM gate for experiments. Never
  commit data, models, caches, secrets, or raw player identifiers.

Goal completion requires verified practical benefit, not implementation alone.
