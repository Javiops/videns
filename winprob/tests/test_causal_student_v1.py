from dataclasses import replace
import unittest
import torch
from winprob.iql_models import PublicBoardBatch
from winprob.causal_student_v1 import CausalStudent, purchase_objective


class CausalStudentTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(71)
        b = 2
        board = PublicBoardBatch(torch.randint(1, 9, (b, 10)), torch.randint(1, 12, (b, 10, 3)),
            torch.ones(b, 10, 3), torch.randn(b, 10, 22), torch.ones(b, 10, 1),
            torch.tensor([[False]+[True]*4+[False]*5]*b), torch.tensor([[False]*5+[True]*5]*b), torch.randn(b, 37))
        return CausalStudent(8, 12), board, torch.tensor([[1, 2], [2, 3]])

    def test_player_permutation_symmetry_preserves_buyer(self):
        model, board, runes = self.fixture(); model.eval()
        order = torch.tensor([0, 4, 1, 3, 2, 9, 5, 8, 6, 7])
        changed = replace(board, **{name: getattr(board, name)[:, order] for name in board.__dataclass_fields__ if name != 'global_public'})
        torch.testing.assert_close(model.state(board, runes), model.state(changed, runes), atol=2e-6, rtol=2e-6)

    def test_item_order_counts_and_empty_identity(self):
        model, _, _ = self.fixture(); model.eval()
        ids = torch.tensor([[1, 2, 0], [0, 0, 0]]); counts = torch.tensor([[2., 1., 0.], [0., 0., 0.]])
        attrs = torch.zeros(2, 23); empty = torch.tensor([False, True])
        a = model.actions(ids, counts, attrs, empty)
        torch.testing.assert_close(a, model.actions(ids[:, [1, 0, 2]], counts[:, [1, 0, 2]], attrs, empty), atol=0, rtol=0)
        torch.testing.assert_close(a[1], model.empty_main, atol=0, rtol=0)
        wrong = counts.clone(); wrong[0, 0] = 1
        self.assertFalse(torch.equal(a[0], model.actions(ids, wrong, attrs, empty)[0]))
        with self.assertRaises(ValueError): model.actions(ids, counts, attrs, ~empty)

    def test_deep_checkpoint_and_chunk_gradient_equivalence(self):
        model, _, _ = self.fixture(); model.double().train()
        s = torch.randn(2, 64, dtype=torch.float64, requires_grad=True)
        a = torch.randn(7, 64, dtype=torch.float64, requires_grad=True)
        left = model.scores(s, a, chunk=7, checkpoint_deep=False)
        g1 = torch.autograd.grad(left.square().sum(), (s, a, *model.interaction.parameters()))
        right = model.scores(s, a, chunk=3, checkpoint_deep=True)
        g2 = torch.autograd.grad(right.square().sum(), (s, a, *model.interaction.parameters()))
        torch.testing.assert_close(left, right, atol=1e-14, rtol=1e-14)
        for x, y in zip(g1, g2): torch.testing.assert_close(x, y, atol=2e-13, rtol=2e-13)

    def test_observed_label_and_teacher_gradient_are_separate(self):
        logits = torch.tensor([[.2, -.4, 1.], [.3, .8, -.2]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 2]); teacher = torch.tensor([[.1, .2, .7], [.6, .3, .1]], dtype=torch.float64)
        loss, parts = purchase_objective(logits, labels, teacher)
        gradient, = torch.autograd.grad(loss, (logits,))
        expected = (2*logits.softmax(1)-torch.nn.functional.one_hot(labels, 3)-teacher)/2
        torch.testing.assert_close(gradient, expected, atol=1e-15, rtol=1e-15)
        self.assertGreaterEqual(parts['teacher_KL'].item(), 0)
        hard, _ = purchase_objective(logits, labels)
        same, _ = purchase_objective(logits, labels, torch.nn.functional.one_hot(labels, 3).to(torch.float64))
        torch.testing.assert_close(same, 2*hard, atol=1e-15, rtol=1e-15)

    def test_inference_signature_has_no_teacher_or_target(self):
        import inspect
        params = inspect.signature(CausalStudent.forward).parameters
        self.assertNotIn('teacher', params); self.assertNotIn('observed_main', params)
        logits = torch.ones(2, 3, requires_grad=True)
        with self.assertRaises(ValueError): purchase_objective(logits, torch.tensor([0, 1]), torch.full((2, 3), 1/3, requires_grad=True))


if __name__ == '__main__': unittest.main()
