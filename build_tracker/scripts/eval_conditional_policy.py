"""Compare a transformer with a frozen TRAIN-only conditional basket policy.

Uses the common displayed-policy traversal, validation populations and paired
game bootstrap in eval_policy. Outputs belong to the isolated experiment run.

    ... scripts/eval_conditional_policy.py --artifact candidate.pt --split val --games 0
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from app.conditional_policy import action_of, purchase_sequence
from app.shop_econ import SAVE_ITEM


def legal(action: tuple[int, ...], row: dict, dragon) -> bool:
    """Legacy regression adapter; production comparison uses ordered entries."""
    if action == (SAVE_ITEM,):
        return True
    budget = row.get("gold_est") if row.get("gold_est") is not None else row.get("gold") or 0
    return purchase_sequence(action, row, float(budget), dragon) is not None


def main() -> None:
    from eval_policy import main as evaluate
    if "--compare-conditional" not in sys.argv:
        sys.argv.append("--compare-conditional")
    evaluate()


if __name__ == "__main__":
    main()
