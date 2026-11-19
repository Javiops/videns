"""Exact, synthetic identification checks; not a production OPE estimator.

All arithmetic uses fractions so the falsification examples have no simulation
or rounding uncertainty. Passing these checks establishes algebra only, not
identification, overlap, calibration, or policy benefit in real game data.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping


Number = int | float | Fraction
ACTIONS = ("A", "B")


def _fraction(value: Number) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not probability measurements")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Probabilities must be finite")
        return Fraction(str(value))
    if not isinstance(value, (int, Fraction)):
        raise TypeError("Use int, finite float, or Fraction")
    return Fraction(value)


def _probability(value: Number) -> Fraction:
    result = _fraction(value)
    if not 0 <= result <= 1:
        raise ValueError("A probability must be in [0, 1]")
    return result


def _distribution(values: Mapping[str, Number]) -> dict[str, Fraction]:
    if not values:
        raise ValueError("An action distribution cannot be empty")
    result = {key: _probability(value) for key, value in values.items()}
    if sum(result.values(), Fraction(0)) != 1:
        raise ValueError("Synthetic action distributions must sum exactly to one")
    return result


@dataclass(frozen=True)
class ContrastScore:
    direct: Fraction
    residual: Fraction
    contrast_weight: Fraction

    @property
    def total(self) -> Fraction:
        return self.direct + self.residual


def dr_contrast_score(
    *,
    action: str,
    reward: Number,
    target: Mapping[str, Number],
    reference: Mapping[str, Number],
    q_values: Mapping[str, Number],
    logging: Mapping[str, Number],
) -> ContrastScore:
    """Paired one-decision target-minus-reference DR score on one observation.

    The context has already selected each mapping. Policies and nuisance
    functions are fixed independently of evaluation observations. ``logging``
    represents the supplied historical propensity, not an imitation ranking.
    This function deliberately rejects unsupported policies and never clips.
    """
    target_p = _distribution(target)
    reference_p = _distribution(reference)
    logging_p = _distribution(logging)
    keys = set(target_p)
    if keys != set(reference_p) or keys != set(logging_p) or keys != set(q_values):
        raise ValueError("Policies, propensities and Q must share the action space")
    if action not in keys:
        raise ValueError("Observed action is outside the declared action space")
    q = {key: _probability(value) for key, value in q_values.items()}
    observed_reward = _probability(reward)
    for key in keys:
        if logging_p[key] == 0 and (target_p[key] > 0 or reference_p[key] > 0):
            raise ValueError("Policy action has no supplied historical support")
    if logging_p[action] == 0:
        raise ValueError("Observed action has zero supplied logging probability")
    direct = sum(
        ((target_p[key] - reference_p[key]) * q[key] for key in keys), Fraction(0)
    )
    weight = (target_p[action] - reference_p[action]) / logging_p[action]
    return ContrastScore(direct, weight * (observed_reward - q[action]), weight)


@dataclass(frozen=True)
class PopulationAtom:
    context: str
    action: str
    outcome: int
    mass: Fraction


TRUE_LOGGING = {
    "strong": {"A": Fraction(4, 5), "B": Fraction(1, 5)},
    "weak": {"A": Fraction(1, 5), "B": Fraction(4, 5)},
}


def synthetic_population(effect: bool = False) -> tuple[list[PopulationAtom], dict]:
    """Enumerate P(skill, action, binary outcome), with known causal means.

    Equal strong/weak populations choose A with probabilities .8/.2. Without
    an action effect, their win probabilities are .8/.2 for either action.
    The positive-control variant has a .2 A-minus-B effect in both strata.
    """
    if effect:
        q = {"strong": {"A": Fraction(9, 10), "B": Fraction(7, 10)},
             "weak": {"A": Fraction(3, 10), "B": Fraction(1, 10)}}
    else:
        q = {"strong": {key: Fraction(4, 5) for key in ACTIONS},
             "weak": {key: Fraction(1, 5) for key in ACTIONS}}
    atoms = []
    for context in ("strong", "weak"):
        for action in ACTIONS:
            for outcome in (0, 1):
                outcome_mass = q[context][action] if outcome else 1 - q[context][action]
                mass = Fraction(1, 2) * TRUE_LOGGING[context][action] * outcome_mass
                atoms.append(PopulationAtom(context, action, outcome, mass))
    return atoms, q


def expected_dr_contrast(
    atoms: list[PopulationAtom],
    *,
    target: Mapping[str, Mapping[str, Number]],
    reference: Mapping[str, Mapping[str, Number]],
    q: Mapping[str, Mapping[str, Number]],
    logging: Mapping[str, Mapping[str, Number]],
) -> Fraction:
    """Exact expectation across a declared finite synthetic population."""
    if not atoms or sum((atom.mass for atom in atoms), Fraction(0)) != 1:
        raise ValueError("Population atom masses must sum exactly to one")
    if any(atom.mass <= 0 for atom in atoms):
        raise ValueError("Population atom masses must be positive")
    result = Fraction(0)
    for atom in atoms:
        score = dr_contrast_score(
            action=atom.action, reward=atom.outcome,
            target=target[atom.context], reference=reference[atom.context],
            q_values=q[atom.context], logging=logging[atom.context],
        )
        result += atom.mass * score.total
    return result


def absorbing_value(
    *, nonterminal_probability: Number | None = None, terminal_outcome: int | None = None,
) -> Fraction:
    """Use a known terminal result directly, including games ending before h."""
    if terminal_outcome is not None:
        if isinstance(terminal_outcome, bool) or terminal_outcome not in (0, 1):
            raise ValueError("Terminal outcome must be binary 0 or 1")
        return Fraction(terminal_outcome)
    if nonterminal_probability is None:
        raise ValueError("A nonterminal state needs an estimated probability")
    return _probability(nonterminal_probability)


def _record(value: Fraction) -> dict:
    return {"value": float(value), "exact_fraction": str(value)}


def identification_report() -> dict:
    """Run exact negative/positive controls and return machine-readable evidence."""
    atoms, true_q = synthetic_population()
    target = {x: {"A": 1, "B": 0} for x in true_q}
    reference = {x: {"A": 0, "B": 1} for x in true_q}
    wrong_g = {x: {key: Fraction(1, 2) for key in ACTIONS} for x in true_q}
    wrong_q = {x: {key: Fraction(1, 2) for key in ACTIONS} for x in true_q}
    overall_win = sum((atom.mass * atom.outcome for atom in atoms), Fraction(0))
    added = {}
    for action in ACTIONS:
        action_mass = sum((atom.mass for atom in atoms if atom.action == action), Fraction(0))
        wins = sum((atom.mass * atom.outcome for atom in atoms if atom.action == action), Fraction(0))
        added[action] = wins / action_mass - Fraction(1, 2)

    observed_cases = {
        "correct_g_wrong_q": (TRUE_LOGGING, wrong_q),
        "wrong_g_correct_q": (wrong_g, true_q),
        "correct_g_correct_q": (TRUE_LOGGING, true_q),
        "wrong_g_wrong_q": (wrong_g, wrong_q),
    }
    estimates = {
        name: expected_dr_contrast(atoms, target=target, reference=reference, q=q, logging=g)
        for name, (g, q) in observed_cases.items()
    }
    hidden_atoms = [PopulationAtom("hidden", a.action, a.outcome, a.mass) for a in atoms]
    hidden_q = {"hidden": {"A": Fraction(17, 25), "B": Fraction(8, 25)}}
    estimates["hidden_skill_correct_observational_g_and_q"] = expected_dr_contrast(
        hidden_atoms, target={"hidden": {"A": 1, "B": 0}},
        reference={"hidden": {"A": 0, "B": 1}}, q=hidden_q,
        logging={"hidden": {"A": Fraction(1, 2), "B": Fraction(1, 2)}},
    )
    effect_atoms, effect_q = synthetic_population(effect=True)
    estimates["positive_control_correct_g_wrong_q"] = expected_dr_contrast(
        effect_atoms, target=target, reference=reference, q=wrong_q, logging=TRUE_LOGGING,
    )
    estimates["positive_control_wrong_g_correct_q"] = expected_dr_contrast(
        effect_atoms, target=target, reference=reference, q=effect_q, logging=wrong_g,
    )
    expected = {
        "correct_g_wrong_q": Fraction(0), "wrong_g_correct_q": Fraction(0),
        "correct_g_correct_q": Fraction(0), "wrong_g_wrong_q": Fraction(9, 25),
        "hidden_skill_correct_observational_g_and_q": Fraction(9, 25),
        "positive_control_correct_g_wrong_q": Fraction(1, 5),
        "positive_control_wrong_g_correct_q": Fraction(1, 5),
    }
    checks = {
        "constant_judge_is_marginally_calibrated": overall_win == Fraction(1, 2),
        "spurious_wp_added_A": added["A"] == Fraction(9, 50),
        "spurious_wp_added_B": added["B"] == Fraction(-9, 50),
        **{name: estimates[name] == value for name, value in expected.items()},
    }
    return {
        "schema": "synthetic-identification-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "Exact algebra on an enumerated synthetic population; no game-data or causal-benefit claim",
        "population": {"kind": "synthetic_enumerated", "atoms": len(atoms), "game_count": 0},
        "sampling_uncertainty": "None: exact population expectations; empirical interval coverage not tested",
        "judge": {"constant_prediction": _record(Fraction(1, 2)),
                  "population_win_rate": _record(overall_win), "marginal_calibration_error": _record(Fraction(0))},
        "wp_added_by_action": {key: _record(value) for key, value in added.items()},
        "zero_effect_causal_truth": _record(Fraction(0)),
        "positive_control_causal_truth": _record(Fraction(1, 5)),
        "paired_dr_target_A_minus_reference_B": {key: _record(value) for key, value in estimates.items()},
        "checks": checks,
    }
