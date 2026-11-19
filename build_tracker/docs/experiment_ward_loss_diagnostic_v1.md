# Pre-registration: ward-term CE decomposition + gradient diagnostic (v1)

2026-09-22. Owner's assignment, recorded before the run. Run dir:
`data/experiments/16.18-ward-loss-diagnostic-v1/`. NO training, no weight
update, no VAL or test read. This is a READING, not a verdict.

## What it measures and why

The proposed loss contrast would replace the current exact-basket CE with
`main + 0.25 * ward`. The current loss decomposes exactly, by the chain
rule over the ward-stripped grouping of the catalogue:

```
-log p(b*) = [logZ - logZ_g*] + [logZ_g* - s_b*]
              main purchase       ward | main purchase
```

Two readings on a fixed FIT subset:

1. **Size of each term** (mean nats), overall and split by whether the true
   basket contains a ward, plus the share of rows whose target group is a
   singleton (those contribute exactly zero ward loss).
2. **Gradient magnitude and alignment on the SHARED parameters**: `|g_main|`,
   `|g_ward|`, `cos(g_main, g_ward)`, and how far the update direction
   moves at w=0.25 (`|g_weighted - g_total| / |g_total|`, and the cosine
   between them).

## Declared reading rule, written before the numbers

The owner's correction is adopted verbatim: **the size of the ward term is
not a verdict on the contrast.** A term worth 0.05 nats scaled by 0.25
changes the loss by 0.0375 nats, which says nothing about how much the
gradients move. The gradient reading is what carries information about
interference, and it is still only a local property:

- `cos(g_main, g_ward) < 0` — the objectives pull against each other on
  shared parameters; the contrast has a mechanism to help.
- `cos(g_main, g_ward) > 0` — they reinforce; downweighting the ward term
  discards gradient the main task was using.
- `cos ~ 0` — orthogonal; the contrast would mostly be about capacity, and
  this diagnostic cannot resolve it.

**None of these three outcomes decides the experiment on its own.** A
gradient cosine at a frozen point does not predict a 16-epoch trajectory.
The reading is published so the loss contrast is designed with it in hand;
the contrast stays pending the owner's call.

## Measurement points (both frozen, both published)

| point | loss | why |
|-------|------|-----|
| `Bf_ep16` | plain full-catalogue CE | the parent any twin arms would start from; the phase-2 recipe's own loss |
| `econ_ft_ep8` | corrected scores (base + econ on the fixed top-128 domain) | the current dev baseline, the state the model is actually in |

## Method

- Fixed FIT subset built identically to the phase-2 training curves:
  `torch.randperm(n_fit, generator=manual_seed(16))[:10000]`, so the total
  CE is directly comparable to `fixed_subset_full_ce` in those curves.
- Grouping is the decoder's own map (wards 2055 / 772043 stripped; every
  basket that strips to empty joins group 0 with SAVE).
- `eval()` mode: dropout off, so the reading is reproducible. Declared as a
  difference from training gradients, which carry dropout noise.
- Gradients accumulated over microbatches of 32 with the mean-CE weighting,
  via `torch.autograd.grad`. No optimizer is constructed; no parameter is
  written.
- The decomposition residual `|CE - (main + ward)|` is reported as an
  arithmetic self-check and must be ~1e-6.

---

## READING (2026-09-22 15:45, fixed 10.000-row FIT subset, 79 s)

`data/experiments/16.18-ward-loss-diagnostic-v1/ward_loss_diagnostic_full.json`.

| frozen point | CE | = main | + ward\|main | ward share | cos(g_main, g_ward) | \|g_ward\|/\|g_main\| |
|---|---|---|---|---|---|---|
| `Bf_ep16` | 1.70479 | 1.39503 | 0.30976 | 18.2% | **−0.287** | 0.244 |
| `econ_ft_ep8` | 1.63764 | 1.33391 | 0.30374 | 18.5% | **−0.122** | 0.262 |

- Decomposition residual 5.2e−09 / 4.2e−09: exact, as the chain rule
  requires.
- `Bf_ep16`'s total CE on this subset is **1.70479**, the same value as the
  twin econ fine-tune's verified step-0 parent reproduction — independent
  confirmation that the subset and the loss are the same objects.
- The ward term is **not** small: 0.31 nats, ~18% of the loss. It
  concentrates where expected — 0.99 nats on ward-containing targets vs
  0.14 on the rest — and 764 of 10.000 rows have a singleton target group
  and contribute exactly zero ward loss.
- **The cosine is negative at both points.** Under the reading rule written
  before the run, that is the sign under which the contrast has a mechanism
  to help: the two objectives pull partly against each other on shared
  parameters. At w=0.25 the RAW GRADIENT vector moves 19% (cosine 0.98 with
  the unweighted one). That is a property of the gradient, NOT of the Adam
  step the trainer would actually take: Adam rescales per-coordinate by its
  running moments, so this figure does not translate into a 19% change in
  the update applied to the weights.

**This does not decide the contrast**, per the declared rule. A gradient
cosine at a frozen point does not predict a 16-epoch trajectory, and it was
measured with dropout off. The loss contrast stays pending the owner's
call, now designed with this reading in hand.
