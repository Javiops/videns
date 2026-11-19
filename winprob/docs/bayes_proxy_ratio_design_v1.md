# Closed-purchase evidence transferred as a probability ratio

2026-10-03, after closing the static route correction. Mathematical and
input-feasibility stage only; no new forecast, purchase fit or score yet.
The intended endpoint remains the player's actual counted purchase, compared
with unchanged native M2. The previous+0.069pp route result is not rescued.

## Different information and its availability

The hypothesis is that a player's CLOSED past choices reveal which causal
forecaster fits their subsequent behavior. Account for the probability of
those choices in their own PRE-purchase states, rather than counting raw items
or feeding future completed items to a model. This does not identify a private
intention, preference, reward or optimal build.

Directly using native M2's historical parent forecasts would carry uncertified
final-role metadata into the new history channel. Verified code dependencies:
train_prefix.py query includes role and selects a same-role opponent; the
economic features also use role. We do not certify those forecasts as causal.
Native M2 remains the frozen CURRENT reference, with its declared limitation.
No new history feature may depend on those final-role fields.

Candidate causal proxies already exist within this project: original OOF
state-only HistoryProbeActor and its frozen purchase-memory predictor. Both
use the previously audited public-state allowlist and exclude the query's
entire game from learned weights and retrieval memory. Preserve the old k32,
one neighbor per distinct training game, same champion, cosine geometry,
temperature rule and.5state/.5retrieval mixture. No encoder/retrieval retuning
or repeat of the old raw M2/memory averaging recipe is proposed here.

Before any real comparison, retain exact checkpoint/training-game provenance,
declare the forecast/retrieval batches and controls, and validate new queries
including historical ones. A past forecast must not consume its own action;
that action becomes evidence only after its real visit closes. Synthetic SAVE
nodes depend on future qualification and must be excluded BEFORE history
adjacency or counting. Ward-only real purchases remain. No current/future
action, outcome, synthetic SAVE existence, or replaced older visit may leak.

## Bayesian proxy mixture and an anchored correction

For causal proxy forecasts S_t(a) and M_t(a), start pi=(.5,.5). For each real
past visit s finalized strictly before the query, update

    w_k(H) proportional to pi_k product_s P_{s,k}(observed_main_s).

At the query, B_prior=.5S+.5M and B_post=w_S S+w_M M. Transfer only the
relative factor to the normalized frozen M2 distribution p0:

    r(a) = B_post(a)/B_prior(a)
    q(a) = p0(a) r(a) / sum_b p0(b) r(b).

No history, or no posterior change, gives EXACT p0. No new action support is
created. Stable log-space arithmetic handles long prefixes; impossible proxy
observations fail, without epsilon support or silently dropped history.

If the two B distributions were the true P(A|O,H) and P(A|O), their ratio
would be a conditional likelihood factor. Here they are finite approximations.
Transfer to M2 is a predictive hypothesis, not an exact Bayesian identity for
M2 and not an established improvement. It avoids replacing M2's absolute
probabilities with those of a weaker proxy, but may still make predictions worse.

Sequential Bayesian expert aggregation has a pathwise cumulative log-loss
bound against each included expert:

    L_Bayes(1:T) <= L_expert_k(1:T) - log(pi_k).

This bound applies to the PROXY forecaster that actually issues those sequential
predictions and updates with their observed outcomes. It does NOT transfer
through the ratio to q, does not protect every focal decision, does not ensure
top1/top3 accuracy, and does not survive arbitrary forecast/update substitution.
Tests must include a counterexample to any claimed M2-bound transfer. Expert
aggregation under log loss is discussed by
[Orseau, Lattimore and Legg, Soft-Bayes](https://proceedings.mlr.press/v76/orseau17a.html).
Our basic posterior update is ordinary Bayes, not an implementation of that
paper's full Prod/tracking algorithm; no League benefit follows from the paper.

With positive prior weights, B_post/B_prior lies between the minimum and maximum
w_k/pi_k wherever the denominator is positive. If S=M on a query, q=p0 regardless
of history. If all historical observed-likelihood pairs agree, w=pi and q=p0.
These are limiting cases and tests, not useful-accuracy evidence.

## Immediate authorized feasibility work

Implement/test the normalized posterior and ratio transfer on synthetic data.
Then census ONLY past-prefix availability for the original2900FIT16.17focals,
using original admission, episode/time metadata and the already certified
visit-end/source-event sidecars. Publish the fixed query ledger before opening
action-kind values. Access full-action IDs only at indices strictly preceding
each focal in its own episode, to distinguish real purchases from synthetic
SAVE. No current/future action value, main-action label, forecast, outcome,
raw archive, database,859development or1911confirmation read in this census.
The whole action file's digest is verified, but only past scalar values are
interpreted; report this distinction honestly.

Report number of histories, prefix-count distribution, ties/overlaps, excluded
synthetic nodes and total causal proxy forecast queries needed. Do not select
or drop evaluation games based on history. This selects no effect threshold,
posterior clipping, future label, personal-intent target or real scoring recipe.
After feasibility, a separate fixed predictive protocol with strong M2/control
comparisons, native decoding, uncertainty and failure rules is required.
