"""Fixed train-only multiset lookup policy for paired displayed-policy evaluation.

The hierarchy and support threshold are predeclared, not tuned on validation.
The only learned objects are counts of complete retained purchase multisets.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from app.dataset_artifacts import sha256
from app.purchase_multiset import expanded_labels
from app.shop_econ import SAVE_ITEM, try_purchase
from app.footwear_policy import footwear_arguments, validate_footwear_state

CONFIG = {
    "policy": "conditional-multiset-backoff-v1",
    "gold_x": True,
    "gold_input": "prequential-v3",
    "gold_band": 250,
    "minute_band": 2,
    "min_support_l1": 5,
    "max_options": 3,
    "hierarchy": ["champion_role_inventory_gold_minute", "champion_role_inventory",
                  "champion_role", "champion", "global"],
    "tie_break": "lexicographic action IDs",
    "no_legal_action": "no options; never invent a SAVE label",
}


def require_explicit_slot_state(row: dict) -> None:
    """A comparative run must observe capacity, never infer legacy completion."""
    for player in [row, *(row.get("others") or [])]:
        if player.get("slot_state_version") != "role-slots-v1":
            raise ValueError("Comparison requires explicit role-slots-v1 on shopper and board")
        if type(player.get("bot_quest_complete")) is not bool:
            raise ValueError("Comparison requires boolean bot_quest_complete on shopper and board")
    hidden = row.get("unmodeled_regular_slots")
    if type(hidden) is not int or hidden < 0:
        raise ValueError("Comparison requires explicit nonnegative unmodeled_regular_slots")
    if row.get("role_slot_inventory_unknown"):
        raise ValueError("Comparison cannot use unknown role-slot inventory")
    validate_footwear_state(row)


def keys_for(row: dict) -> list[tuple]:
    champ, role = row.get("champion") or "?", row.get("role") or "?"
    inventory = tuple(sorted(int(i) for i in row.get("inventory") or [] if i))
    gold = row.get("gold_est") if row.get("gold_est") is not None else row.get("gold")
    gold_band = int(gold or 0) // CONFIG["gold_band"]
    minute_band = int((row.get("ts") or 0) / 60000 / CONFIG["minute_band"])
    return [("l1", champ, role, inventory, gold_band, minute_band),
            ("l2", champ, role, inventory), ("l3", champ, role), ("l4", champ), ("l5",)]


def action_of(row: dict) -> tuple[int, ...]:
    labels = tuple(expanded_labels(row))
    if row.get("save_kind") == "no_buy_death":
        if labels != (SAVE_ITEM,):
            raise ValueError("No-buy label must contain only SAVE_ITEM")
        return labels
    if SAVE_ITEM in labels and labels != (SAVE_ITEM,):
        raise ValueError("SAVE_ITEM cannot share a purchase basket")
    return labels


def purchase_sequence(action: tuple[int, ...], row: dict, budget: float, dragon):
    """Find a legal ordered sequence, retaining every copy at its actual cost.

    Target multisets have no order. A combine can release a slot needed by the
    next purchase, and newly displayed components must never discount a later
    retained entry. Return None when no order satisfies all constraints.
    """
    if not action or SAVE_ITEM in action:
        return None
    target = Counter(action)
    footwear = footwear_arguments(row,dragon,required_version=getattr(dragon,'footwear_state_version',None))

    @lru_cache(maxsize=None)
    def search(remaining, inventory, wallet):
        if not remaining:
            return ()
        pending = Counter(dict(remaining))
        protected = target - pending
        for item_id in sorted(pending, key=lambda i: (dragon.classify(i).get("depth", 0), i), reverse=True):
            purchase = try_purchase(item_id, list(inventory), wallet, dragon,
                role=row.get("role") or "", bot_quest_complete=row.get("bot_quest_complete"),
                unmodeled_regular_slots=row.get("unmodeled_regular_slots") or 0,
                role_slot_inventory_unknown=bool(row.get("role_slot_inventory_unknown")),
                protected=protected, **footwear)
            if purchase is None:
                continue
            cost, owned = purchase
            rest = pending.copy()
            rest[item_id] -= 1
            if not rest[item_id]:
                del rest[item_id]
            tail = search(tuple(sorted(rest.items())), tuple(sorted(owned.elements())), wallet - cost)
            if tail is not None:
                return ((item_id, cost),) + tail
        return None

    sequence = search(tuple(sorted(target.items())), tuple(sorted(row.get("inventory") or [])), float(budget))
    if sequence is None:
        return None
    return [{"item_id": item_id, "cost": cost} for item_id, cost in sequence]


class ConditionalPolicy:
    """Predictor-compatible policy; fitting accepts the frozen TRAIN file only."""
    def __init__(self, dragon, tables, evidence):
        self.dragon = dragon
        # Stable frequency ties are independent of input row ordering.
        self.tables = {key: tuple(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))
                       for key, counts in tables.items()}
        self.config = dict(CONFIG)
        self.evidence = evidence
        self.provenance = evidence["provenance"]
        self.artifact = Path("conditional_policy.json")
        self.artifact_sha256 = evidence["table_sha256"]
        self.trained_at = "fitted from frozen train in this evaluation"
        self.threshold = "frequency rank"
        self.level_used = Counter()
        self.no_legal_action = 0

    @classmethod
    def fit(cls, train_path, row_factory, dragon, manifest, generation, assignments):
        train_path = Path(train_path)
        if train_path.name != "visits_train.jsonl":
            raise ValueError("Conditional policy may fit frozen TRAIN only")
        expected = generation["files"]["visits_train.jsonl"]["sha256"]
        train_hash = sha256(train_path)
        if train_hash != expected:
            raise ValueError("Conditional TRAIN fingerprint changed")
        if dragon.signature != manifest.get("static_data_sha256") or dragon.version != manifest.get("ddragon_version"):
            raise ValueError("Conditional policy requires the export's pinned static data")
        support, n_train, seen = Counter(), 0, set()
        for row in row_factory(train_path):
            require_explicit_slot_state(row)
            validate_footwear_state(row,required_version=getattr(dragon,'footwear_state_version',None))
            if row.get("gold_est_version") != CONFIG["gold_input"] or row.get("gold_est") is None:
                raise ValueError(f"Conditional comparison requires causal {CONFIG['gold_input']} gold")
            if assignments.get(row.get("match_id")) != "train":
                raise ValueError("Non-training game reached conditional fitting")
            support[keys_for(row)[0]] += 1
            n_train += 1
            seen.add(row["match_id"])
        if not n_train:
            raise ValueError("Cannot fit conditional policy on empty TRAIN")
        keep = {key for key, count in support.items() if count >= CONFIG["min_support_l1"]}
        del support
        tables = defaultdict(Counter)
        action_rows = 0
        for row in row_factory(train_path):
            action = action_of(row)
            if not action:
                continue
            action_rows += 1
            for key in keys_for(row):
                if key[0] != "l1" or key in keep:
                    tables[key][action] += 1
        if sha256(train_path) != train_hash:
            raise ValueError("Conditional TRAIN changed during fitting")
        digest = hashlib.sha256()
        for key in sorted(tables):
            record = [key, [[action, count] for action, count in sorted(tables[key].items())]]
            digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")
        evidence = {
            "schema_version": 1, "config": dict(CONFIG), "table_sha256": digest.hexdigest(),
            "train_sha256": train_hash, "train_rows": n_train, "train_action_rows": action_rows,
            "train_games": len(seen), "table_buckets": len(tables),
            "provenance": {
                "training_split": "train", "selection_split": "validation",
                "export_fingerprint": manifest["export_fingerprint"],
                "eval_version": manifest["eval_version"], "patch": manifest["patch"],
                "ddragon_version": dragon.version, "static_data_sha256": dragon.signature,
                "gold_input": CONFIG["gold_input"], "purpose": generation["purpose"],
                "probe": True, "promotion_allowed": False,
            },
        }
        if manifest.get('footwear_runtime'):
            evidence['provenance'].update(footwear_runtime=manifest['footwear_runtime'],
                footwear_state_version=manifest['footwear_runtime']['footwear_state_version'],
                source_target_version='full-source-completions-v1')
        return cls(dragon, tables, evidence)

    def predict_options(self, row, n_options=3, budget_slack=0, allow_save=False):
        require_explicit_slot_state(row)
        validate_footwear_state(row,required_version=getattr(self.dragon,'footwear_state_version',None))
        if budget_slack != 0:
            raise ValueError("Conditional displayed policy supports exact supplied budgets only")
        budget = row.get("gold_est") if row.get("gold_est") is not None else row.get("gold")
        if budget is None:
            raise ValueError("Conditional displayed policy requires a budget")
        for key in keys_for(row):
            options = []
            for action, _count in self.tables.get(key, ()):
                if action == (SAVE_ITEM,):
                    if not allow_save:
                        continue
                    option = {"kind": "save", "items": []}
                else:
                    items = purchase_sequence(action, row, budget, self.dragon)
                    if items is None:
                        continue
                    option = {"kind": "buy", "items": items}
                options.append(option)
                if len(options) >= min(n_options, CONFIG["max_options"]):
                    break
            if options:
                self.level_used[key[0]] += 1
                return options
        self.level_used["none"] += 1
        self.no_legal_action += 1
        return []
