# DeepSeek math review (deepseek-flash)

Source-only review. 18 files, 283370 chars sent. No data, split or run history was included, so nothing here is a verified finding about this project's numbers.

---

# Audit

Everything below is derived from the source you pasted, plus arithmetic on the numbers quoted inside those files. Where a claim needs a number nobody has computed, I say so and name the statistic. Confidence is marked per point.

---

## Part 1 — what is implemented

### 1.1 The estimand, and what the objective actually fits

**The intended estimand.** For a state $s$ (shopping player + 10-player board + inventory + clock) the target is a multiset $y:\mathcal I \to \mathbb N$ with finite support. `app/purchase_multiset.py` makes this explicit and lossless (`label_counts`, `expanded_labels`), and `decision_kind`/`save_kind` split the space into build/ward/no-buy. Write the population object as
$$
\pi(y \mid s) = P_\text{high-elo}\big(\text{retained purchases}=y \mid s,\ \text{a shopping decision occurred}\big).
$$
Note the conditioning event. The visit set is defined by `reconstruct.reconstruct_game` as item events inside a 10 s idle window, so rows exist only where something was *detected*. Manual recalls are invisible (`CLAUDE.md`, `docs/history.md` §"What still cannot be demonstrated"). So $\pi$ is a **selection-conditioned** distribution and is not the marginal behaviour of a shopper, and the no-buy class is a subsample of no-buy behaviour (post-death only). This is stated in the docs; the consequence is that no intercept/prior term in the fitted model is a behavioural rate. **[source, high confidence]**

**What is fitted.** In `train_prefix.ShopDataset._fill_row`:

```python
for item_id, count in counts.items():
    ... self.targets[i, idx] = count
self.is_save[i] = int(save_kind == "no_buy_death" or any(x == SAVE_ITEM ...))
```

with the batch presenting

```python
"targets": (self.targets[idx] > 0).float(),   # binary presence
"counts":  self.targets[idx].long(),          # copies
```

and the loss

$$
\mathcal L = \underbrace{\mathrm{BCE}_{\mathbf w}\!\big(\sigma(z),\, \mathbb 1[y_j>0]\big)}_{\text{presence, pos-weighted}} \;+\; \lambda_c\!\!\sum_{j:\,y_j>0}\!\!\mathrm{CE}\big(q_j,\; y_j-1\big) \;+\;\lambda_s\,\mathrm{BCE}(s_\theta,\cdot) \;+\;\lambda_t\,\mathrm{CE}(t_\theta,\cdot).
$$

The mask enters as $\sigma(z)\!\to\!\sigma(z)\mathbb 1[j\in\mathcal A(s)]$. So the fitted object is

$$
\tilde P(y\mid s)=\prod_j \hat p_j(s)^{\mathbb 1[y_j>0]}(1-\hat p_j(s))^{\mathbb 1[y_j=0]}\cdot\prod_{j:y_j>0}\hat q_j(y_j\mid s),\qquad \hat p_j = P(y_j>0\mid s,\; j\in\mathcal A(s)),
$$

i.e. **per-item marginals under a wallet-conditioned support, times a per-item copy distribution conditional on purchase**, trained under a re-weighted risk. That is a well-defined, proper-maximisable quantity. It is not $\pi(y\mid s)$:

| | object |
|---|---|
| Estimand | joint $\pi(y\mid s)$ over multisets |
| Objective | product-of-marginals $\times$ positive-conditioned count, at a pos-weighted risk, on a mask-conditioned support |
| Metric (`eval_policy.first_action_exact`) | the *decoded* greedy feasible prefix equals $y$ exactly |

None of the three is a function of the other two. **[source, high confidence]**

**A defect in the target construction.** The legacy fallback

```python
if not hit and not counts and row.get('label_id'):
    self.targets[i, self.label_index.get(int(row.get('label_id') or 0), 0)] = 1
```

maps a failed lookup to index `0`, which is `sorted(label_counts)[0]` — the single most frequent purchase label. Any malformed row manufactured by a partially-updated export becomes a false positive on the modal item. Should be a no-op. **[source, high confidence, low expected frequency]**

### 1.2 The factorisation: what it gets wrong, and the size of the error

**What is lost.** The likelihood has no term coupling $j$ and $k$. Substitution (two items competing for the same slot / the same 1,300 g) and complementarity (a component whose value depends on what else you own) are absent from the objective. They are not absent from the *pipeline* — they are handled by the decoder: the running wallet, `try_purchase`, `is_blocked`, and the recipe filter `allowed` in `predict_options`. So the coupled part of the joint is supplied by a deterministic feasibility pass over an item ranking, not by the model.

Three consequences I can derive:

1. **The basket size distribution is not fitted.** `predict_options` emits the greedy feasible prefix above `self.threshold`. The number of items, the total spend, and the probability of the empty basket are all functions of (a) the ranking, (b) `threshold`, (c) the mask width, (d) item prices. Nothing in the loss is a function of size. `train_prefix.score_pred_baskets` *reports* `avg size predicted vs actual`, and `miss_profile_v1` *reports* the gold gap distribution — both are diagnostics that measure a quantity the model is not trained on.
2. **The empty basket is derived, not modelled.** With the SAVE sentinel as a pseudo-item in the same sigmoid head, $P(\text{no buy})$ is $P(\text{SAVE is the argmax above }\tau)$ — a max-of-many-thresholds statistic whose value moves with the number of legal items in $\mathcal A(s)$ and with the relative scaling of the SAVE label's pos-weight against everything else. The code has four separate patches for this one structural gap: `caps[SAVE]=lo` in `POSW_CUSTOM`, the separate `save_head`, `save_floor = threshold*0.55`, and the `override` bar in the decoder. `docs/history.md` records that "a hand-shaped `pos_weight` bump … save pinned low" moved save behaviour — which is confirmation that the no-buy propensity is set by a weighting constant, not estimated. **[source, high confidence]**
3. **The composition is ranking-order-dependent and the ranking is pos-weighted per label.** Since $w_j$ varies by label, the cross-label *ranking* that the decoder walks is a label-dependently distorted ordering of the marginals (see 1.3). So the composed basket is ordered by an object that is not the marginal probability.

**How badly — this needs a measurement, and the statistic is easy.** The three quantities below are computable from data you already have and none is in any report:

- **(M1) The factorisation gap in log score.** The factorised model's exact log score on the recorded multiset is available in closed form from the two heads and the mask:
  $$
  \log \hat P(y\mid s)=\sum_{j\in y}\log\hat p_j+\sum_{j\notin y}\log(1-\hat p_j)+\sum_{j\in y}\log \hat q_j(y_j),
  $$
  with the analytic $\log w_j$ correction of 1.3 applied. `app/conditional_policy.py` is a genuine *joint* model (its table keys are multisets: `tables[key][action] += 1` with `action = tuple(expanded_labels(row))`), so it gives $\log \hat P_\text{joint}(y\mid s)$ on the same rows. The mean paired difference is a direct measurement of the dependence the factorisation discards. Both policies already run under `--compare-conditional` and `outcome_capture`; only the log score is missing from `score_visit`.
- **(M2) The residual covariance matrix** $\mathrm{Cov}(y_j-\hat p_j,\,y_k-\hat p_k)$ restricted to item pairs that co-occur, on held-out rows. Its off-diagonal is the dependence the model fails to represent; it also tells you whether the pairs are predominantly substitutive (negative) or complementary (positive), which is exactly what a pairwise energy term would fit.
- **(M3) The size calibration**: predicted basket-size pmf vs realised, and the same for total spend. `miss_profile_v1` computes the realised cost; nothing computes the predicted distribution of it.

**Prior expectation from your own ledger.** The conditional frequency table scores **20.8%** on the displayed-policy metric and the transformer **20.4%** (`docs/history.md`, "The transformer is clearly worth its complexity"). A backoff histogram over (champ, role, inventory, 250 g gold band, 2 min bucket) is a *joint* estimator with enormous variance in high-dimensional cells. If a joint estimator with that bias profile ties a 10-token transformer on exact match, then exact match is nearly saturated by the marginal structure, and the ceiling for any model in this family is close to the table's rate. That caps how much the joint can be worth *on that metric* — and is an argument that the metric, not the model, is the binding constraint. **[source, high confidence; the inference is not a proof, because both models use the same covariates]**

### 1.3 The loss

**The positive weight is exactly invertible, and the code does not invert it.** For a per-label head with pos_weight $w_j$, the optimum of the weighted BCE satisfies

$$
\frac{\sigma(z^\star_j)}{1-\sigma(z^\star_j)}=w_j\frac{p_j}{1-p_j}
\quad\Longrightarrow\quad
\sigma\!\big(z^\star_j-\log w_j\big)=p_j .
$$

So a pos-weighted head carries an *exact, per-label* affine correction back to the calibrated marginal. Some consequences:

- The scores are a **label-dependent** monotone distortion of the marginals. Cross-label ranking — which is precisely what `torch.argsort(first_probs)` and the `0.8` threshold operate on — is therefore not a probability ranking. Two items with $p=0.02,\ w=24$ and $p=0.05,\ w=8$ have equal scores. `train_prefix`'s own artifact config says `"item_score_semantics": "pos-weighted ranking score; not calibrated probability"` — correct, and this is the reason.
- The correction costs nothing: `splice_save` already applies a *scalar* `SAVE_TILT = log(POS_WEIGHT_CAP)`. It should be `log(w_j)` per label and applied to the item head, not only to the save head. `w_j` is not saved in the artifact (only `pos_weight_cap` and `posw_sched`), so this needs a one-line addition to the checkpoint payload. **[source, high confidence — this is an exact algebraic identity, not an approximation, when the head is unconstrained]**
- Consequence of not doing it: the decoder's decision boundary at $\tau=0.8$ is a tuned boundary on a distorted scale, and the optimal $\tau$ is coupled to the pos_weight schedule. `docs/history.md` records that the whole architecture/pos_weight/threshold/blend/graft combination was chosen while looking at one split, and that `BASKET_THRESHOLD` was "swept 0.5–0.8" — with the winner at the **top edge of the grid**. An optimum at a sweep boundary is the signature of a monotone nuisance direction, not of an interior fit. **[source, medium confidence — the grid endpoint winning could also mean the grid was chosen badly]**

**The two losses are not commensurable, and the BCE value is mask-dependent.** `nn.BCEWithLogitsLoss` defaults to `mean` over all elements, including the masked ones (whose logits are $-10^4$, so their contribution is $\approx 0$ but they still occupy the denominator, $|\text{batch}|\times|\mathcal L|$). Tightening the mask therefore *lowers* the printed BCE without improving fit. The `COUNT_LOSS_W = 0.25` weight is relative to a corpus-size-and-mask-dependent denominator. Practical consequences: (i) epoch loss is not comparable across two runs with different masks — which is exactly what the posterior-vs-point comparison is; (ii) the relative weighting of the two heads is not invariant to vocabulary size or mask density. **[source, high confidence]**

**The count head is trained on a different population from the one that uses it.** `count_loss_fn(out["counts"][bought], counts[bought]-1)` fits $q_j(c\mid c\ge 1, s)$ over *every* bought item. At decode, `predict_options` consults `want[idx]` only for items that already cleared $\tau$, passed `is_blocked`, passed `try_purchase`, and were inside the recipe filter — a selected subset. The two populations differ by an event the model controls. Whether this biases counts up or down is not derivable; the measurement is the calibration of $\hat q_j$ restricted to the decode-admitted subset. **[source for the mismatch; needs measurement for the sign and size]**

**`decode_copy_counts(..., "count-tail-075-v1")` thresholds an uncalibrated softmax at 0.75.** `P(C\ge k)\ge 0.75` on a softmax that never saw a temperature fit or a reliability curve. The docstring says "not a player-facing confidence" — true, but it is used as a *decision* threshold, so it needs the calibration check, not just the disclaimer. **[source, high confidence]**

**Two degenerate counts.** `MAX_COPIES=3` refuses more (good, `_fill_row` raises rather than clipping). But `predict_baskets` and `calibrate_threshold` both hardcode `budget = dataset.budget[r] + GOLD_DRIFT`, i.e. **+500 g**, ignoring the trained mask (`gold_x + c_{0.65}(\tau)`) entirely. Under the posterior regime the mask at $\tau\approx29$ s is roughly $0.5153\cdot29+k_{0.65}\sqrt{29}$, tens of gold — so the training-log decoder is spending against a budget an order of magnitude wider than the one the model was trained under. That is a concrete, derivable reason the training-log basket numbers are not admissible, additional to the ones the banner already lists. **[source, high confidence]**

### 1.4 The mask

**The mask is a conditioning set, not a constraint, and there are four different ones.**

| regime | wallet used for the mask | where |
|---|---|---|
| training, posterior off | `gold + GOLD_DRIFT` (500) | `_fill_row` |
| training, posterior on | `gold_est + c_{0.65}(\tau)` | `_fill_row` |
| training-log decode | `dataset.budget + 500` (**not** `allowed`) | `predict_baskets`, `calibrate_threshold` |
| `eval_policy` decode | `gold_est` , `budget_slack=0` | `score_visit` |
| live | `gold_exact` | `app/live.py` |

Because the mask is applied inside a **BCE**, not a softmax, items outside $\mathcal A(s)$ contribute neither positive nor negative gradient. So the fitted object is $P(y_j>0\mid s,\ j\in\mathcal A(s))$, and the serving object is a renormalised restriction of that to $\mathcal A_{\text{tight}}(s)$. This is not "train unconstrained, constrain at decode"; it is **train conditional on a favourable event, serve conditional on a stricter one**. Since $\mathcal A_{\text{tight}}\subseteq\mathcal A_{\text{wide}}$ on 26.4% of recorded purchases, the served distribution is a renormalisation of a likelihood fitted on a support that includes the answer. The two errors partially cancel (the conditioned probability is an *over*-estimate of the marginal; the tighter decode removes the over-scored expensive items), which is why the pipeline produces plausible numbers at all — but a cancellation you have not identified is a coincidence, not a correction, and it will not hold under a regime change (i.e. live).

**The force-unmask makes the training constraint tautological.**

```python
for item_id in row.get("label_ids") or [row.get("label_id")]:
    ... self.legal[i, idx] = True
```

After this, the training constraint is $y\in\mathcal A(s,y)$ — a function of the label. A constraint that can never be violated carries no information and no regularisation. It also overrides non-budget legality (the boots-uniqueness and completed-item rules), which is harmless only if `reconstruct` is internally consistent. The honest version is $y\in\mathcal A(s,g)$ with $g$ integrated out; see 1.5 and Part 2. **[source, high confidence]**

**Consequences, derivable:**

1. No gradient ever teaches the model that an unaffordable item is a negative. Items that are illegal at serving may have been *unseen* at training (they were masked the whole time, or force-unmasked as positives), so their scores are whatever the trunk generalises to. Nothing measures that.
2. The mask is per-item. Basket-level feasibility is an order-dependent search (`basket_is_legal`, `conditional_policy.purchase_sequence`) because a combine frees a slot. So $\mathcal A(s)$ strictly contains the feasible basket set $\mathcal F(s,g)$, and the decode re-restricts. The model's support is a relaxation of the true one, fitted on the relaxation.
3. The knobs $(\tau, q, \text{slack})$ are three free parameters all operating on the same feasibility boundary and all tuned against a 0/1 metric. The observed 14-point swing (`const:0` vs `const:500`, model frozen) is the diagnostic: **a change of that size in the headline metric from a non-model parameter says the displayed basket is dominated by the constraint and the threshold, not by $\hat\pi$.** That is the strongest internal evidence in the repo that the model is not estimating the joint. **[source, high confidence]**

### 1.5 Gold uncertainty: is a drawn wallet + widened mask a valid approximation?

Split the mechanism into its three parts; they have different statuses.

**(a) Per-row draw of $g$ — valid, with three caveats.** A single draw per row over 2M rows is an unbiased Monte-Carlo estimate of the risk
$$
\mathbb E_{y,g}\big[\mathcal L(\theta; y, g)\big],\qquad g\sim\pi(g\mid s),
$$
and the stochastic gradient is unbiased. So the *objective* is right. Caveats: (i) the quantile function is clamped at $q\le0.02$ and $q\ge0.98$, so the tails are point masses — the injected error is under-dispersed by construction; (ii) `max(0.0, ...)` puts a point mass at zero; (iii) the draws are independent across rows, so the *serial* correlation of the estimator's error within a game is destroyed — the marginal is right, the joint across rows is not. Nothing in the likelihood needs that joint, but the features are shared across rows in a game, so the model is trained to be robust to a smeared version of a quantity that is in fact piecewise-constant within a session. Minor. **[source, high confidence]**

**(b) Mask at a fixed 65th percentile — not marginalisation.** This is a plug-in decision under uncertainty at a chosen quantile, and $q$ was chosen by a sweep on TRAIN (`rate:2` ↔ q 0.60–0.70). It is a third conditioning event, distinct from both the tight decode and the exact live wallet. The design justification ("a lucky draw cannot widen its own candidate set") buys protection against the model learning "high wallet ⇒ everything legal", but that is the *truth*, and a coherent marginalisation would teach it. So (b) trades correctness for conservatism, and it is not what "marginalising over a posterior" means. **[source, high confidence]**

**(c) The posterior itself: where the derivation bites.** The fitted object is $c = \text{currentGold} - \text{gold\_est}$ and the fit is `design = [dts, sqrt(dts)]` on three anchor ages. I checked the design numerically, because "extrapolating below the support" is the obvious objection and here it is *not* the problem:

- Taking $u=\tau/60\in\{1,2,3\}$ and columns $(u,\sqrt u)$, $(X'X)^{-1}$ has leverage 0.764 / 0.374 / 0.871 at the three anchors and **0.823 at $u = 29/60$**. So the extrapolated prediction at a median visit is about as well determined *as a function value* as the in-sample points. The no-intercept specification (both basis functions vanish at $\tau=0$) is what rescues it. **[source, arithmetic]**
- But the *decomposition* is weakly identified: $X'X$ has eigenvalues 19.87 and 0.128, condition number ≈ 155, and the near-null direction is $\tilde a:\tilde k \approx -0.545:0.838$ in $(u,\sqrt u)$ units, i.e. a unit change in $k_q$ is compensated by $\approx -0.084$ in $a_q$. So "bias grows linearly with staleness, spread as $\sqrt\tau$" is **not** separately supported: many $(a_q,k_q)$ pairs fit the three points equally well. Note also that the structural argument (drift + diffusion from a sum of increments) predicts $a_q$ *shared* across quantiles with $k_q\propto z_q$; the code rejects a shared drift on the basis of "58–75% error below $q=0.20$", which is a comparison of a 2-parameter fit to a 1-parameter fit on **three points, one residual degree of freedom**. The per-quantile drift is absorbing model misfit. **[source, arithmetic + medium confidence on the interpretation]**
- The fit is unweighted least squares on three *sample percentiles* with unequal $n$, which is neither efficient nor able to produce coefficient standard errors — and **`posterior.json` reports no uncertainty on $(a_q,k_q)$ at all**, while every downstream consumer (`gold_correction`, `parse_arm`, the mask) treats them as known. That is a plug-in-posterior error: the wallet posterior ignores the uncertainty about the wallet posterior. With 1 residual df, that uncertainty is not small. **[source, high confidence]**
- The diagnostic itself found spread scaling with **game phase** (IQR 182→218→247 by minute bucket) and found that kill/bounty events do not explain the spread (quiet windows IQR 203 on 83% of the mass). The first says the posterior is missing a covariate. The second says something more important: if the $\sqrt\tau$ term were income arriving as a stream, restricting to quiet windows should shrink the spread. It does not. **So the $\sqrt\tau$ scaling is more plausibly the walking estimator's own error accumulation than true income variance**, which means (i) it will not be fixed by a better income model (the docs half-suspect this: "which is why the Phase A income variants could not help"), and (ii) it is a *measurement* error of a known function of the state, so the right treatment is a state-augmented estimator or a robustness formulation, not a behavioural prior over income. **[source, medium-high confidence; this is an inference from your reported numbers, not a derivation]**
- The population mismatch: the fit uses the **no-purchase stratum**, and it is applied to rows where a purchase occurred. Purchases select on high $g$, and the error is $g-e$, so selection on the outcome correlates with the error. The shortfall scan's 26.4%/median 83 g is the visible signature. The docs correctly refuse to *fit* on that label-selected sample; they do not correct for the selection in the posterior that is *applied* to those rows. The clean fix is in Part 2. **[source, high confidence]**

**What breaks, then.** (i) Extrapolation *variance* is not the problem; the structural form and the missing phase/uncertainty are. (ii) The mask-at-a-quantile is not the marginal. (iii) The feature-level marginalisation and the mask-level plug-in can disagree about the same row, and the trained mask is 8× tighter than the training-log decoder's. (iv) The transfer argument for live — *"a visit one second after a frame already has a width of ~7 g, so the exact-wallet end live serves is inside the training range"* — is at best "within ~7 g of the boundary": every offline row has strictly positive width (only `gold_exact` rows get 0, and `gold_exact` is not in `stream_rows`' key list at all), so width $=0$ is at the edge of the support, not inside it. More importantly the *support* mismatch is in the same direction as the `const:0` decode that cost 14 points on train: training positives are always affordable under a $q_{0.65}$ mask; live masks at the exact wallet. The pre-registration's live-OOD claim is directionally right; the mechanism it names (a range vs a point) is not the dominant one. **[source, high confidence on the code path; medium on the magnitude]**

### 1.6 Evaluation

**What is right, and it is worth saying because it is rare.** The bootstrap unit is the game (`keys[rng.randrange(len(keys))]` over `match_id`), which matches the clustering of visits; the point estimate and the interval are computed over the *same* filtered population (`_defined` then `_mean`); `paired_diff` requires identical scored games and visit counts; and `TEST_USAGE_LOG` records every test read. Those are the correct choices.

**What is wrong or overstated.**

1. **The "paired" bootstrap is partially paired for prediction-dependent metrics.** `shared = sorted(set(pa) | set(pb))` and then `pa.get(game, ())` / `pb.get(game, ())`. For `save_card_precision` (defined only where a save card was shown) and `ward_only_recall`, the two arms define the metric on different visits, so `diff` is a difference of two ratio estimators over *different* populations, resampled together. That is the right conservative choice — but the label should be "partially paired", and the estimand is "the difference in each policy's own-population accuracy", not a paired effect. For the prediction-independent metrics (`first_action_exact`, `first_basket_*`) the populations coincide and the pairing is real. **[source, high confidence]**
2. **The intervals have their own Monte Carlo error.** `_ci` is a nearest-rank percentile over 400 replicates (indices 9 and 389), so each endpoint is roughly a 10-observation order statistic, and no BCa/bias correction. For ratio statistics with skewed sampling distributions (precision, recall) percentile intervals are biased low. 400 is thin for a 2.5% tail. **[source, high confidence]**
3. **`paired_diff` uses one RNG stream per metric with the same seed**, so all metrics share the same game resamples (common random numbers — good) but the intervals are mutually dependent, which matters for any max-over-metrics claim. Not documented. **[source]**
4. **Exact match is not a proper score and is not aligned with the objective.** For a target with high conditional entropy, 0-1 loss is the least efficient consistent-with-nothing rule: a model can improve its likelihood and lose exact match. The repo has direct evidence of the map failing: the pos_weight schedule change "landed between H24 and the simple 8:24 on every metric"; the transformer vs conditional table tie at 20.4/20.8. And the metric's expectation moves 14 points with a *decode* parameter. So it is (i) insensitive, (ii) nearly saturated by a 5-way backoff histogram, (iii) not proper, (iv) partly a function of the mask. It is defensible as a *product* metric for the lead card; it is not a defensible model-selection criterion. **[source, high confidence]**
5. **`action_family_match` scores against a future-derived set.** `with_targets` attaches the next completed final from the same game (deliberate lookahead; it is a label, not a feature). `family = components_of(target_id) | {target_id}`, and the metric is `any(i in family for i in pred)`. So the primary pair includes a "was the advice in the right recipe tree of an event that had not happened yet" statistic. Legitimate as a product metric, but it is neither the estimand nor the training objective, and promoting on it while promoting on `first_action_exact` is promoting on two different hypotheses. **[source, high confidence]**
6. **`any_of_three_exact` is a function of a decoder heuristic.** Options 2 and 3 come from `greedy(banned)` with the lead item banned — a diversity rule with no fitted quantity behind it. So the metric the product actually exposes is partly measuring the ban rule. **[source]**
7. **The promotion gate is evaluated against a biased wallet.** "zero unaffordable displayed baskets" is checked at `budget_slack=0` against `gold_est`, whose median error is −75 g with a 29.5% over-credit rate. The correct gate is a *probability*: $P\big(C(\hat y) \le g \mid s\big) \ge 1-\epsilon$ integrating the fitted posterior — computable today from `posterior.json`'s quantile grid, and not computed. **[source, high confidence]**

### 1.7 Multiple comparisons

Let the family be the $K$ arms compared against a reference on one window, with paired differences $D_k$. Each `paired_vs_const0` interval is a per-comparison 95% statement. The quantity the reader wants is
$$
P\Big(\max_k |T_k| > c \mid H_0\Big),\quad T_k = \hat D_k/\widehat{\mathrm{se}}_k .
$$
Under exchangeability and equicorrelation $\rho$, this lies between $\alpha$ ($\rho\to1$) and $1-(1-\alpha)^K$ (independence) — and here $\rho$ is high, because every arm scores the *same visits with the same frozen model*, differing only in a scalar. So the inflation is much less than $K$-fold but strictly above $\alpha$, and it is not zero.

Scale of the family: `mask_slack_sweep_v1.py` reports `paired_vs_const0` for **every metric × every arm** (8 metrics × 12 arms = 96 intervals in phase 1/2, plus 8 quantile arms in 2b), with no correction and no statement of family size. `eval_basket_composition.py` declares two primaries and requires *both* to exclude zero, giving a joint level of $\approx 1-0.95^2 = 9.75\%$ under independence, which is not the 5% the word "significant" implies. On top of that, `docs/history.md` records a sequence of candidate experiments reusing the same validation window ("Architecture, pos_weight, threshold, blend and graft recipe had all been chosen while looking at it"). A per-experiment single-comparison rule does not control the across-experiment error when the experiments share the window.

**What would control it:**

- One pre-registered primary metric and one pre-registered primary comparison per generation window; everything else labelled exploratory with no promotion rights. The protocols partially do this; the *ledger* does not.
- For a genuine selection family (the sweep), the correct shape is exactly what is implemented — select on TRAIN, confirm once on validation — **provided** the confirmation is one comparison at level $\alpha$/ (number of experiments already spent on this window), i.e. a fixed-sequence gate with pre-allocated alpha, or Holm over the pre-registered family.
- For the arms themselves, a **studentised maximum bootstrap**: resample games, recompute all $K$ arm differences and their standard errors on each replicate, take $\max_k|T_k|$, and use its empirical 95th percentile as the critical value. This is cheap, uses the joint game-level structure, and does not require an independence assumption. `paired_diff` already has the guts of it.
- The real backstop is the frozen test set read once under `--confirm-final` and appended to `TEST_USAGE_LOG`. Treat every validation-confirmed win as provisional until that read, and count the comparisons spent.

**[source for the family sizes and the uncorrected reports; the FWER math is standard, high confidence]**

---

## Part 2 — what the model and the distribution should be

### 2.0 The likelihood I would use

Let $s$ be the state, $g$ the true wallet, $\pi(g\mid s)$ a label-free posterior on $g$, and $\mathcal F(s,g)$ the set of multisets with a legal purchase sequence at wallet $g$ (monotone non-decreasing in $g$; `conditional_policy.purchase_sequence` is already the membership oracle). The predictive I would fit and report is

$$
\boxed{\;p_\theta(y\mid s)=\int_{\mathcal G} \frac{\exp f_\theta(y,s)}{Z_\theta(s,g)}\;\mathbb 1\big[y\in\mathcal F(s,g)\big]\;\pi(g\mid s)\;dg\;},\qquad Z_\theta(s,g)=\!\!\sum_{y'\in\mathcal F(s,g)}\!\!\exp f_\theta(y',s).
$$

Three things about this, in order of practical importance:

**(1) The constraint is inside the normaliser, so there is no training/serving support mismatch.** $\theta$ is fit where it is served. The renormalisation gap of 1.4 disappears by construction.

**(2) The empty multiset is in the support** (it is feasible at every $g$), so $p_\theta(\emptyset\mid s)$ is a fitted number, not a threshold artefact.

**(3) $g$ is integrated, not plugged in.** Feasibility is monotone in $g$, so the integrand vanishes below
$$
g_{\min}(y,s)=\min_{\sigma\ \text{legal order of }y}\ \sum_t \mathrm{cost}(\sigma_t)\ \ (-\ \text{sell refunds}),
$$
and the integral is a censored marginal. This is the mechanism by which a recorded purchase is *also an observation about the wallet* — and it is the one thing the current pipeline throws away when it force-unmasks.

### 2.1 The three factorisations, costed

| | normaliser | estimation cost | bias |
|---|---|---|---|
| **AR over an item sequence + STOP** | exact over legal sequences, so exact over multisets via the pushforward $p(y)=\sum_{\sigma\mapsto y}p(\sigma)$ | low: your `PrefixModel` already takes a query + inventory; `SequentialCompositionPolicy._compose` is already the greedy decode of this model, re-querying at the updated state — so the machinery exists and is a wrapper instead of a likelihood | bias is the *order artefact*: a multiset with several legal orders has mass spread over them, and the model is trained on one. The legal-order DP gives a deterministic canonical order, so the artefact is small; if you want it gone, train with random order and average |
| **EBM over $\mathcal F(s,g)$ with restricted $Z$** | exact by definition | high: $Z$ needs enumeration or MCMC. Tractable only with a truncated candidate set (top-$M$ by a proposal), which replaces an implicit bias with an explicit one | none in principle; the truncated-support bias is at least *stated*. The right home for pairwise terms $\theta_{jk}y_jy_k$ — which are exactly the objects (M2) would measure. No order artefact |
| **Two-stage count-then-composition** | exact: $p(n\mid s)$ proper over $0..N$, $\mathrm{Comp}_n$ proper over $n$-multisets | lowest | the composition model must *condition on $n$*; the current code does not condition on anything, it thresholds marginals |

**Recommendation.** These are not three options; the third is the second factorised, and the first is the third with the length implicit. I would build:

$$
p_\theta(y\mid s)=\underbrace{\pi_0(s)}_{p(\emptyset)}\mathbb 1[y=\emptyset]+\big(1-\pi_0(s)\big)\sum_{n=1}^{N}\rho_n(s)\;g_\theta(y\mid s,n),
$$
with $\rho$ a proper pmf over basket size (this is the *basket* length, not the per-item copy count the current `count_head` produces), $g_\theta$ an autoregressive model over the $n$ items conditioned on length, decoded on the canonical legal purchase order, and the whole thing normalised *on the feasible set* — i.e. the autoregressive next-step distribution renormalised over the next purchases that `try_purchase` admits at the running state. That gives: an exact joint, a first-class no-buy, a fitted size distribution, the constraint natively in the likelihood, and a per-basket log score for evaluation. The per-item copy head survives inside $g_\theta$ (each of the $n$ slots can carry a copy count, as `SequentialCompositionPolicy` already does).

The EBM is the right *target* if and only if the residual covariance (M2) turns out to be large and structurally pairwise. Fit (M2) first; it tells you whether the pairwise terms are worth the normaliser.

### 2.2 Where the wallet constraint belongs

Not at decode, and not as a widened mask. Three placements, all inside the model:

1. **In the normaliser** (above): the feasible set is the support, so the model's probabilities always sum to one on what can actually be bought, at training and serving alike.
2. **In the autoregressive transition**: at each composition step the next-purchase distribution is masked by `try_purchase` at the *running* wallet and inventory and renormalised. This is a prefix constraint, so it is exact — and it is the same function `SequentialCompositionPolicy._compose` and `basket_is_legal` already call, so there is no second implementation.
3. **Nothing else.** No `GOLD_DRIFT`, no `gold_est + c_q`, no `budget_slack`. The slack parameter family (q, slack, τ) disappears, which removes three tuned constants whose joint optimum is currently being fitted against a 0/1 metric.

### 2.3 How gold uncertainty should be handled

**Fit.** Replace the three-point per-quantile LS with a fit that produces standard errors and uses every window: quantile regression of $c$ on $(\tau,\sqrt\tau)$ with pinball loss, or a parametric location-scale MLE, both with $\tau$ *and game phase* as covariates (the diagnostic already shows phase matters). Report coefficient SEs. Better still, export `gold_gap.frame_ts` — the anchor timestamp exists in the reconstruction (`debug['frame_series']`) and the export just drops it — so $\tau$ is exact rather than `ts mod 60000`; the 0.45% cadence error and the phase assumption both go away, and it is a rebuild rather than a research programme.

**Know the limit.** Truth exists only at frames, so the error at $\tau<60$ s is not measurable from frame data alone. The $\sqrt\tau$ shape is derivable from the estimator's structure (a walk whose error is a sum of increments over $[0,\tau]$), not from the three points; but the *magnitude* below 60 s and the residual shape are an extrapolation. The honest statement is that the shape is a modelling assumption and the coefficients are estimated at $\tau\ge60$; the telemetry is what would close it.

**Use.** Two changes, both cheap:

- **Training: integrate, with censoring.** Because feasibility is monotone in $g$,
  $$
  \int \mathbb 1[y\in\mathcal F(s,g)]\,P_\theta(y\mid s,g)\,\pi(g\mid s)\,dg \;=\; \bar\Pi\big(g_{\min}(y,s)\mid s\big)\,P_\theta(y\mid s)
  $$
  when $P_\theta$ depends on $g$ only through the indicator. That is a **per-row scalar weight** equal to the posterior probability that the wallet admitted the recorded basket, computable today from the quantile grid. It replaces the force-unmask (which asserts that probability is 1) and it is exactly the quantity `censored_coverage_check` already computes as a diagnostic and never uses. Note `basket_shortfall_shape_v1` prices the basket by the first legal sequence found, not the minimum-cost one, so $g_{\min}$ is over-estimated and the weight is conservative — fine, but say so.
  Separately, the mask *for the negative labels* should be softened the same way: an item is a real negative in proportion to how likely the player could have afforded it. A single wide mask weights all negatives equally regardless of the wallet's uncertainty, which is the same error in the other direction.
- **Serving: decide under the posterior.** Display the basket that maximises the posterior-predictive $p_\theta(y\mid s)$ subject to
  $$
  \Pi\big(C(y)\mid s\big) \;\ge\; 1-\epsilon ,
  $$
  i.e. affordable with high probability, not with certainty under a point estimate. That also fixes the promotion gate: "zero unaffordable displayed baskets" should become "$P(\text{affordable})\ge1-\epsilon$ for every displayed option", which is a computable quantity rather than an assertion about a biased estimate.

**And the caveat that the diagnostic forces.** Spread that survives conditioning on quiet windows is not income variance; it is the estimator's own drift accumulation. Then the correct response is a better estimator (state-augmented, or a direct measurement from shop telemetry) or a decision rule that is *robust* to a known error process — not a behavioural prior on income. The posterior is currently doing duty as both.

### 2.4 The save outcome

Represent it as a **mass at the empty multiset**, not as a pseudo-item and not as a side head. In the factorisation above, $\pi_0(s)=p(\emptyset\mid s)$ is a first-class output of a proper pmf, fitted by the same likelihood as everything else, with no pos_weight schedule, no `save_floor`, no `override` bar, and no separate `save_head` whose target is a *different event* from the one the item head is supposed to represent. That is the whole fix, and it deletes four patches.

Two constraints on how it is reported, both already known to the project and both still worth restating: (a) the observability boundary means $\pi_0$ estimates $P(\text{no recorded purchase}\mid\text{a visit occurred})$, which is a strict subset of "the player chose to save". It must be labelled that way wherever it is displayed, and it must never be regularised toward the empirical class balance — which is exactly what `pos_weight` does. (b) Because the visit set is selected on detected events, $\pi_0$'s intercept is a sampling artefact; do not let any calibration procedure re-estimate it from the corpus, or you will calibrate a selection probability.

### 2.5 Scoring rule, calibration, primary metric

**Proper score, primary.** Mean log score of the basket distribution on held-out visits,
$$
\overline{\mathrm{LS}} = \frac{1}{|\mathcal V|}\sum_{v\in\mathcal V}\log p_\theta(y_v\mid s_v),
$$
with the under-the-factorisation form $\log\hat\pi_0$ (or $\log(1-\hat\pi_0)+\log\hat\rho_n+\log g_\theta(y\mid s,n)$) spelled out, and a **game-clustered paired bootstrap** on the difference. This is proper, it uses the shape of the distribution rather than the mode, it detects miscalibration, and it has far more power than exact match at the same family-wise error — which also relieves the multiple-comparisons pressure in 1.7. **This is what I would promote on.**

**Secondary, product-aligned.** A cardinal decision metric on the *displayed options*: for each displayed basket, a gold-distance-weighted multiset loss
$$
\ell(\hat y,y)=\sum_j \mathrm{cost}_j\,\big|(\hat y_j-y_j)\big|\quad\text{or}\quad 1-\frac{2\sum_j\min(\hat y_j,y_j)}{\sum_j \hat y_j+\sum_j y_j},
$$
averaged over visits, with the same paired bootstrap; plus `any_of_three_exact` as the tiebreaker, labelled as a decoder-heuristic metric.

**Diagnostics only.** `first_action_exact` demoted, and *with its ceiling reported*: the plug-in conditional mode's held-out exact rate. The conditional policy's 20.8% is a crude estimate of that ceiling, and reporting a metric next to its empirical ceiling is what makes 20.4% vs 20.8% readable.

**Calibration procedure.**

1. **Per-item reliability on held-out data, after the analytic correction** $z_j\mapsto z_j-\log w_j$ (exact under the identity in 1.3). Report the Brier decomposition — reliability, resolution, uncertainty. The resolution term is the one that matters here: it says how much signal the item head actually has, and it is currently unmeasured.
2. **Condition the reliability curve on the mask regime**, because the score's meaning depends on $\mathcal A(s)$. Calibrating pooled across regimes will hide the conditioning effect of 1.4.
3. **Count head:** fit a temperature on held-out count logits, then report the reliability of $P(C\ge k)$. `count-tail-075-v1`'s 0.75 threshold is meaningless until this exists.
4. **Joint:** the discrete randomised PIT is available once the model emits a proper pmf. A cheap surrogate that works without a total order on multisets: the rank histogram of the **basket gold cost** and of the **basket size** — predicted distribution vs realised. That is the diagnostic that would have caught the size miscalibration in 1.2 immediately.
5. **Sharpness next to every calibration number.** A constant is perfectly calibrated.

---

## What I could not determine from the source

- The magnitude and sign of the factorisation error: needs (M1) and (M2).
- The count head's post-selection bias: needs the calibration of $\hat q_j$ on the decode-admitted subset.
- Whether the posterior's $(a_q,k_q)$ are within sampling error: needs standard errors on the fit, or a re-fit by quantile regression.
- Whether the width-$0$ serving regime costs anything: needs live shop telemetry, and the pre-registration says so.
- The direction of the mask/force-unmask cancellation: derivable in sign for the two components, not for their sum. `const:500` vs `const:0` on train (−0.1424) is the only measured point, and it is on train.
- Whether the conditional table's tie with the transformer is a ceiling for this covariate set: it is evidence, not proof, because both models see the same inputs. The decisive experiment is (M1) on the paired bootstrap, and it is cheap.

## Two things I would fix before anything else, because they are one line each and they invalidate comparisons

1. **Save the per-label `pos_weight` vector in the artifact and apply $z_j-\log w_j$ at inference.** Without it, every reported score is on a label-dependently distorted scale, the threshold $\tau$ is coupled to a nuisance parameter, and no calibration statement about the item head is possible. (Exact identity; `train_prefix.py` artifact payload + `app/predictor.py._probs`.)
2. **Make `predict_baskets` and `calibrate_threshold` use the same budget the mask used**, instead of a hardcoded `+ GOLD_DRIFT`. As written, the training-log decoder spends against a budget an order of magnitude wider than the mask under the posterior regime, which is an additional and currently unstated reason those numbers are inadmissible.