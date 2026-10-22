"""Independent tiny checks for the transition-only comparison."""
import unittest
import numpy as np

from .transition_baselines import PROFILES, Transitions, grouped_rows, paired, topk


class TransitionChecks(unittest.TestCase):
    def model(self):
        model = Transitions.__new__(Transitions)
        model.keys = np.arange(1, 13)
        model.frequency = np.arange(12, 0, -1)
        model.marginal = np.full(12, 1 / 12)
        model.group_sizes = np.ones(12, int)
        model.group_sizes[0] = 2  # SID 1 is collided; SID 2 is a singleton.
        model.tie_order = np.arange(12)
        model.sid_rows = {1: (np.array([2, 3]), np.array([2., 1.])), 2: (np.array([2]), np.array([1.]))}
        model.item_rows = {0: (np.array([2, 3]), np.array([2., 1.]))}
        return model

    def test_topk_exact_with_boundary_ties(self):
        rng = np.random.default_rng(7)
        scores = rng.integers(-1, 3, (30, 50)).astype(np.float32)
        tie_order = rng.permutation(50)
        actual = topk(scores, tie_order)
        expected = tie_order[np.argsort(-scores[:, tie_order], axis=1, kind="stable")[:, :10]]
        self.assertTrue(np.array_equal(actual, expected))

    def test_item_zero_and_collision_destinations_are_aggregated(self):
        rows = grouped_rows(np.array([0, 0, 1]), np.array([2, 2, 3]), np.array([1, 2, 4]), 12)
        self.assertEqual(rows[0][1].tolist(), [3.])
        model = self.model()
        p = next(p for p in PROFILES if p["name"] == "item_transition_last1")
        score, support = model.scores(p, np.array([[-1, 1]]), np.array([[-1, 0]]))
        self.assertTrue(support[0])
        self.assertAlmostEqual(float(score[0, 2]), 2 / 3, places=6)

    def test_dedup_recency_collision_filter_and_smoothing(self):
        model = self.model()
        p = next(p for p in PROFILES if p["name"] == "sid_lift_recent3")
        score, _ = model.scores(p, np.array([[1, 1, 2]]), np.array([[0, 0, 1]]))
        expected = .4 * (np.log1p(1 / (20 / 12)) + .5 * np.log1p(2 / (20 / 12)))
        self.assertAlmostEqual(float(score[0, 2]), expected, places=6)
        self.assertEqual(float(score[0, 0]), 0.)
        self.assertEqual(float(score[0, 1]), -1000.)
        p = dict(kind="sid_probability", history=1, smoothing=20, filter_seen=False)
        score, _ = model.scores(p, np.array([[1]]), np.array([[0]]))
        self.assertAlmostEqual(float(score.sum()), 1., places=6)
        self.assertAlmostEqual(float(score[0, 2]), (2 + 20 / 12) / 23, places=6)

    def test_complementarity_contingency_and_union_are_distinct(self):
        result = paired(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.array([1, 0, 1, 1]))
        self.assertEqual(result["contingency"], dict(both=1, tiger_only=1, statistics_only=1, neither=1))
        self.assertEqual(result["oracle_union_hits"], 3)
        self.assertEqual(result["combined_hits_by_group"]["neither"], 1)


if __name__ == "__main__":
    unittest.main()
