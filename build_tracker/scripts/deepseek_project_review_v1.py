r"""Send a source bundle of this project to DeepSeek and ask for a
strengths/weaknesses review.

What it sends is deliberately narrow: governance docs, all of app/, and the
pipeline scripts. It never reads .env, data/ or anything untracked-by-intent,
and it prints the exact file list and byte count before the call so the
outbound payload is auditable.

Note what this review can and cannot be. A reader of the source can judge
structure, invariants and failure handling. It cannot check any number this
project has produced, because the corpus, the split and the run history are
not in the bundle. Treat the output as hypotheses to verify, not findings.

    .venv\Scripts\python.exe scripts\deepseek_project_review_v1.py
    .venv\Scripts\python.exe scripts\deepseek_project_review_v1.py --budget 250000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
ROOT = Path(__file__).resolve().parent.parent

# Priority order. Everything past --budget is dropped from the tail and
# reported, so a trim is visible rather than silent.
PRIORITY: list[str] = [
    "CLAUDE.md",
    "README.md",
    "docs/history.md",
    "app/reconstruct.py",
    "app/shop_econ.py",
    "app/predictor.py",
    "app/db.py",
    "app/live.py",
    "app/advisor.py",
    "app/config.py",
    "scripts/promote_served_model.py",
    "scripts/status.py",
    "scripts/run_tests.py",
    "scripts/baseline.py",
    "scripts/eval_policy.py",
    "app/*.py",
    "scripts/train_prefix.py",
]

MATH_PRIORITY: list[str] = [
    "CLAUDE.md",
    "docs/history.md",
    "docs/experiment_gold_posterior_training_v1.md",
    "docs/experiment_mask_slack_v1.md",
    "app/purchase_multiset.py",
    "app/shop_econ.py",
    "app/predictor.py",
    "app/conditional_policy.py",
    "app/footwear_policy.py",
    "scripts/eval_policy.py",
    "scripts/basket_composition_policy.py",
    "scripts/eval_basket_composition.py",
    "scripts/gold_posterior_fit_v1.py",
    "scripts/gold_error_shape_v1.py",
    "scripts/budget_shortfall_shape_v1.py",
    "scripts/mask_slack_sweep_v1.py",
    "scripts/miss_profile_v1.py",
    "scripts/train_prefix.py",
]

SYSTEM = (
    "You are a blunt senior engineer reviewing an unfamiliar codebase. "
    "You have the source only: no data, no run history, no ability to execute. "
    "Say plainly when a judgement would need evidence you do not have, rather "
    "than guessing. Do not flatter the code."
)

MATH_SYSTEM = (
    "You are a statistician and machine-learning researcher reviewing the "
    "probabilistic modelling of an unfamiliar codebase. You have the source "
    "only: no data, no fitted parameters, no run history, no ability to "
    "execute. Reason about the estimand, the likelihood, the independence "
    "assumptions, the constraint handling, the calibration and the inference "
    "procedure. Write mathematics where mathematics is clearer than prose. "
    "State plainly when a judgement needs a measurement you cannot make. "
    "Do not flatter the design."
)

MATH_PROMPT = r"""This project estimates what high-elo League of Legends players
historically bought at an explicit in-game shop decision, given a reconstructed
game state. Read "optimal model" strictly as: the best-specified probabilistic
model of the OBSERVED PURCHASING BEHAVIOUR. It is imitation, not control. It
does not optimise win rate, and any answer framed as reinforcement learning or
outcome optimisation is off-target.

The structure of the problem, as the code has it:
- A decision is a state s (player inventory, gold, level, CS, KDA, plus a
  10-player board snapshot) mapping to a target that is a MULTISET of item ids
  (`item-counts-v1`), because a player can buy two of the same component. The
  empty multiset is a legitimate, first-class outcome: deliberately buying
  nothing and saving gold.
- Purchases are constrained by a wallet. Affordability is recursive: owning
  part of a recipe discounts the remainder (`shop_econ.combine_cost`). Slot
  capacity and uniqueness rules further restrict which multisets are legal.
- Gold at decision time is NOT observed exactly offline. Participant frames
  arrive every 60 s, so the wallet is estimated from a stale frame; there is a
  posterior over the true gold, and a "mask slack" mechanism that loosens the
  legality mask to accommodate that uncertainty.

Check my mathematics, and then tell me what the model and the distribution
should actually be.

Part 1 — audit what is implemented:
- The estimand. What quantity is actually being estimated, and does the
  training objective estimate that quantity or a different one?
- The factorisation of the basket distribution. Independent per-item Bernoullis
  plus a count head is a strong assumption: items in a basket substitute and
  complement each other, and the wallet couples them. What does this
  factorisation get wrong, and how badly?
- The loss: weighted BCE, the positive weight, the count head, and how the two
  combine. Is the weighting defensible, and what does it do to calibration?
- Constraint handling. Legality is applied as a mask at decode time rather than
  in the likelihood. What are the consequences of training an unconstrained
  model and constraining it only at decode?
- Gold uncertainty. Marginalising a decision over a posterior on the wallet is
  not the same as widening a hard mask by a slack term. Is the implemented
  approach a valid approximation, and when does it break?
- Evaluation and inference: the bootstrap and its resampling unit, the paired
  comparison, the confidence intervals, and exact-match as the headline metric.
  Is exact-match aligned with the training objective, and is it the right
  primary metric for a multiset target?
- Multiple comparisons. `docs/history.md` records many rejected candidates.
  What does that sequence of tests do to the error rate of the surviving one,
  and what procedure would control it?

Part 2 — prescribe the target design:
- Write down the likelihood you would actually use for a budget-constrained
  multiset, and say why. Consider at least: autoregressive decoding over an
  item sequence, a conditional/energy-based model over legal baskets with a
  normaliser restricted to the feasible set, and a two-stage
  count-then-composition model. Give the trade-offs in estimation cost and bias.
- Say how the wallet constraint belongs in the model rather than after it.
- Say how gold uncertainty should be handled properly.
- Say how the no-buy/save outcome should be represented so it is not a
  degenerate special case.
- Name the proper scoring rule and the calibration procedure you would report,
  and the primary metric you would promote on.

For each point: the file and the specific mechanism, the mathematical
consequence, and how confident you are. Separate what you can derive from the
source from what needs a measurement.
"""


PROMPT = """This is a behavioural-imitation advisor for League of Legends. At an
explicit in-game shop decision it shows what high-elo players historically
bought in similar reconstructed game states. It deliberately does NOT optimise
outcomes and does not claim any purchase is best.

Review the whole project and pinpoint its strengths and its weaknesses.

Cover at least:
- the reconstruction layer, which replays match timelines into shop visits
- the leakage surface: what the model is allowed to see at decision time
- the train/validate/promote path and whether its guarantees actually hold
- the live serving path
- structural and maintainability concerns

For each point give: the file and the specific mechanism, why it matters, and
how confident you are. Separate what you can see in the code from what you
would have to run or measure to confirm. Rank weaknesses by how much damage
they could do, not by how easy they are to fix.
"""

FOCUS = {
    "all": (PRIORITY, SYSTEM, PROMPT, "reports/deepseek_project_review_v1.md"),
    "math": (MATH_PRIORITY, MATH_SYSTEM, MATH_PROMPT,
             "reports/deepseek_math_review_v1.md"),
}


def collect(priority: list[str], budget: int) -> tuple[list[tuple[str, str]], list[str]]:
    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    dropped: list[str] = []
    total = 0
    for pattern in priority:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            p = Path(path)
            rel = p.relative_to(ROOT).as_posix()
            if rel in seen or "__pycache__" in rel:
                continue
            seen.add(rel)
            text = p.read_text(encoding="utf-8", errors="replace")
            if total + len(text) > budget:
                dropped.append(rel)
                continue
            kept.append((rel, text))
            total += len(text)
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400_000,
                    help="max characters of source to send (default: 400000)")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--max-tokens", type=int, default=64000,
                    help="completion budget. deepseek-flash reasons inside it before "
                         "emitting text: a 125k-token bundle burned 16000 tokens on "
                         "reasoning alone and returned nothing (default: 64000)")
    ap.add_argument("--focus", choices=sorted(FOCUS), default="all",
                    help="'all' reviews the project; 'math' audits the probabilistic "
                         "modelling and asks what the distribution should be")
    ap.add_argument("--out", help="override the per-focus default report path")
    ap.add_argument("--dry-run", action="store_true", help="print the bundle manifest and exit")
    args = ap.parse_args()

    priority, system, prompt, default_out = FOCUS[args.focus]
    out_rel = args.out or default_out
    kept, dropped = collect(priority, args.budget)
    chars = sum(len(t) for _, t in kept)
    print(f"bundling {len(kept)} files, {chars} chars (~{chars // 4} tokens)")
    for rel, text in kept:
        print(f"  {len(text):>7}  {rel}")
    if dropped:
        print(f"dropped past --budget {args.budget}:")
        for rel in dropped:
            print(f"           {rel}")
    if args.dry_run:
        return 0
    sys.stdout.flush()

    load_dotenv()
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY not set.", file=sys.stderr)
        return 2

    bundle = "\n\n".join(f"=== {rel} ===\n{text}" for rel, text in kept)
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=1800.0)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{prompt}\n\n{bundle}"},
        ],
        max_tokens=args.max_tokens,
        stream=False,
    )
    content = resp.choices[0].message.content or ""
    print(f"usage: {resp.usage}", file=sys.stderr)
    if not content.strip():
        rt = resp.usage.completion_tokens_details.reasoning_tokens
        print(f"empty content: {rt} reasoning tokens spent, none left for output. "
              f"Raise --max-tokens or lower --budget.", file=sys.stderr)
        return 1

    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# DeepSeek {args.focus} review ({args.model})\n\n"
        f"Source-only review. {len(kept)} files, {chars} chars sent. "
        f"No data, split or run history was included, so nothing here is a "
        f"verified finding about this project's numbers.\n\n---\n\n"
    )
    out.write_text(header + content, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
