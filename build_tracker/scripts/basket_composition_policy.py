"""Stage-2 candidate generators for docs/experiment_basket_composition_v1.md.

Decode-time interventions over a FROZEN checkpoint: the wrapped policy calls
the unmodified Predictor and changes only how the lead BUY basket is composed.
Every other card — including every SAVE card, its position and the primary
action kind on SAVE-led decisions — is byte-identical to the original policy,
so save/ward behavior can only move through the lead BUY contents.

`sequential-v1` (intervention A): compose the lead basket the way a shopper
buys — take the model's top-ranked purchase at the CURRENT state, pay its
recursive recipe cost, update inventory and remaining budget, ask the model
again, and stop when the model's lead turns to SAVE, nothing legal remains,
or the artifact's basket cap is reached. Every step is validated through
`shop_econ.try_purchase` with the same arguments the independent evaluator
replay uses, so a composed basket is legal and affordable by construction
(the evaluator still re-checks independently).

Scores remain internal ranking values; nothing here is player-facing
confidence. No thresholds are introduced; the only constant is the artifact's
existing basket capacity.
"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

MAX_COMPOSE_STEPS = 6  # physical regular-slot bound; the artifact cap also applies


def _multiset(option):
    return Counter(int(item['item_id']) for item in option.get('items') or [])


class SequentialCompositionPolicy:
    """Wraps a frozen Predictor; only the lead BUY card's contents may differ."""

    version = 'sequential-v1'

    def __init__(self, predictor, dragon):
        self.predictor = predictor
        self.dragon = dragon
        self.config = predictor.config
        self.label_ids = getattr(predictor, 'label_ids', None)

    def _compose(self, row, budget, first_options=None):
        from app.shop_econ import try_purchase
        from app.footwear_policy import footwear_arguments
        from eval_policy import serving_row
        footwear = footwear_arguments(row, self.dragon,
            required_version=getattr(self.dragon, 'footwear_state_version', None))
        inventory = list(row.get('inventory') or [])
        wallet = float(budget)
        protected = Counter()
        composed = []
        cap = int(self.config.get('max_basket_items') or MAX_COMPOSE_STEPS)
        state = dict(serving_row(row, wallet))
        for step in range(min(cap, MAX_COMPOSE_STEPS)):
            # Step 0's state equals the caller's row, so the already computed
            # original options are reused verbatim — identical result, one
            # forward pass saved per composed decision.
            if step == 0 and first_options is not None:
                options = first_options
            else:
                options = self.predictor.predict_options(state, budget_slack=0, allow_save=True)
            lead = options[0] if options else None
            if not lead or lead.get('kind') != 'buy' or not lead.get('items'):
                break
            item_id = int(lead['items'][0]['item_id'])
            purchase = try_purchase(item_id, inventory, wallet, self.dragon,
                role=row.get('role') or '',
                bot_quest_complete=row.get('bot_quest_complete'),
                unmodeled_regular_slots=row.get('unmodeled_regular_slots') or 0,
                role_slot_inventory_unknown=bool(row.get('role_slot_inventory_unknown')),
                protected=protected, **footwear)
            if purchase is None:
                break
            cost, owned = purchase
            composed.append({'item_id': item_id, 'cost': cost})
            protected[item_id] += 1
            inventory = list(owned.elements())
            wallet -= cost
            state = dict(serving_row(row, wallet))
            state['inventory'] = inventory
        return composed

    def predict_options(self, row, n_options=3, budget_slack=None, allow_save=False):
        options = self.predictor.predict_options(row, n_options=n_options,
                                                 budget_slack=budget_slack, allow_save=allow_save)
        if not options or options[0].get('kind') != 'buy':
            return options  # SAVE-led and empty decisions are untouched
        budget = row.get('gold_est', row.get('gold'))
        if budget is None:
            return options
        reusable = options if (budget_slack == 0 and allow_save) else None
        composed = self._compose(row, budget, first_options=reusable)
        if not composed:
            return options
        original_lead = _multiset(options[0])
        if Counter(int(i['item_id']) for i in composed) == original_lead:
            return options  # identical composition: keep the original objects
        new_lead = {**options[0], 'items': composed}
        return [new_lead, *options[1:]]


def build(version, predictor, dragon):
    if version == 'sequential-v1':
        return SequentialCompositionPolicy(predictor, dragon)
    if version == 'count-completion-v1':
        raise NotImplementedError(
            'count-completion-v1 is a named protocol alternative; implement it only if the '
            'frozen addendum selects it after the stage-1 diagnostic')
    raise ValueError(f'Unknown composition policy: {version}')
