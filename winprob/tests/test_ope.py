from fractions import Fraction as F
import unittest

from winprob.ope import (
    PopulationAtom, TRUE_LOGGING, absorbing_value, dr_contrast_score,
    expected_dr_contrast, identification_report, synthetic_population,
)


class IdentificationTests(unittest.TestCase):
    def setUp(self):
        self.atoms, self.q = synthetic_population()
        self.target = {x: {"A": 1, "B": 0} for x in self.q}
        self.reference = {x: {"A": 0, "B": 1} for x in self.q}
        self.wrong = {x: {"A": F(1, 2), "B": F(1, 2)} for x in self.q}

    def estimate(self, *, q, logging, atoms=None, target=None, reference=None):
        return expected_dr_contrast(
            self.atoms if atoms is None else atoms,
            target=self.target if target is None else target,
            reference=self.reference if reference is None else reference,
            q=q, logging=logging,
        )

    def test_calibration_can_coexist_with_spurious_eighteen_point_value(self):
        report = identification_report()
        self.assertEqual(report["judge"]["marginal_calibration_error"]["exact_fraction"], "0")
        self.assertEqual(report["wp_added_by_action"]["A"]["exact_fraction"], "9/50")
        self.assertEqual(report["wp_added_by_action"]["B"]["exact_fraction"], "-9/50")
        self.assertEqual(report["zero_effect_causal_truth"]["exact_fraction"], "0")
        self.assertEqual(report["population"]["game_count"], 0)
        self.assertEqual(report["status"], "passed")

    def test_correct_propensity_recovers_null_with_wrong_outcome_model(self):
        self.assertEqual(self.estimate(q=self.wrong, logging=TRUE_LOGGING), 0)

    def test_correct_outcome_model_recovers_null_with_wrong_propensity(self):
        self.assertEqual(self.estimate(q=self.q, logging=self.wrong), 0)

    def test_both_wrong_can_create_false_gain(self):
        self.assertEqual(self.estimate(q=self.wrong, logging=self.wrong), F(9, 25))

    def test_hidden_confounder_bias_survives_correct_observational_nuisances(self):
        hidden = [PopulationAtom("hidden", a.action, a.outcome, a.mass) for a in self.atoms]
        estimate = self.estimate(
            atoms=hidden, target={"hidden": {"A": 1, "B": 0}},
            reference={"hidden": {"A": 0, "B": 1}},
            q={"hidden": {"A": F(17, 25), "B": F(8, 25)}},
            logging={"hidden": {"A": F(1, 2), "B": F(1, 2)}},
        )
        self.assertEqual(estimate, F(9, 25))

    def test_positive_control_recovers_nonzero_truth_with_either_correct_nuisance(self):
        atoms, q = synthetic_population(effect=True)
        self.assertEqual(self.estimate(atoms=atoms, q=self.wrong, logging=TRUE_LOGGING), F(1, 5))
        self.assertEqual(self.estimate(atoms=atoms, q=q, logging=self.wrong), F(1, 5))

    def test_paired_identity_is_exact_zero_even_with_wrong_nuisances(self):
        for atom in self.atoms:
            score = dr_contrast_score(
                action=atom.action, reward=atom.outcome, target=self.target[atom.context],
                reference=self.target[atom.context], q_values=self.wrong[atom.context],
                logging=self.wrong[atom.context],
            )
            self.assertEqual(score.total, 0)

    def test_pair_reversal_reverses_every_contribution(self):
        kwargs = dict(action="A", reward=1, q_values={"A": F(2, 3), "B": F(1, 3)},
                      logging={"A": F(4, 5), "B": F(1, 5)})
        forward = dr_contrast_score(target={"A": 1, "B": 0}, reference={"A": 0, "B": 1}, **kwargs)
        reverse = dr_contrast_score(target={"A": 0, "B": 1}, reference={"A": 1, "B": 0}, **kwargs)
        self.assertEqual(forward.total, -reverse.total)
        self.assertEqual(forward.total, F(3, 4))

    def test_top_three_membership_does_not_make_zero_propensity_supported(self):
        with self.assertRaisesRegex(ValueError, "no supplied historical support"):
            dr_contrast_score(
                action="A", reward=1, target={"A": 0, "B": 0, "C": 1},
                reference={"A": 1, "B": 0, "C": 0},
                q_values={"A": F(1, 2), "B": F(1, 2), "C": F(3, 4)},
                logging={"A": F(4, 5), "B": F(1, 5), "C": 0},
            )

    def test_invalid_mass_and_probability_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "masses must sum"):
            self.estimate(atoms=self.atoms[:-1], q=self.q, logging=TRUE_LOGGING)
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                absorbing_value(nonterminal_probability=bad)

    def test_non_normalized_logging_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum exactly"):
            dr_contrast_score(action="A", reward=1, target={"A": 1, "B": 0},
                              reference={"A": 0, "B": 1}, q_values={"A": 1, "B": 0},
                              logging={"A": F(1, 4), "B": F(1, 4)})

    def test_terminal_result_overrides_unsaturated_prediction(self):
        self.assertEqual(absorbing_value(nonterminal_probability=F(3, 5), terminal_outcome=0), 0)
        self.assertEqual(absorbing_value(nonterminal_probability=F(3, 5), terminal_outcome=1), 1)
        self.assertEqual(absorbing_value(nonterminal_probability=F(3, 5)), F(3, 5))
        with self.assertRaises(ValueError):
            absorbing_value()
        with self.assertRaises(ValueError):
            absorbing_value(terminal_outcome=2)


if __name__ == "__main__":
    unittest.main()
