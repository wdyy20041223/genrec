"""Small synthetic checks of the offline analysis, never model results."""
import itertools
import unittest

import numpy as np

from .cascade_analysis import deviation, oracle_bounds


class CascadeTests(unittest.TestCase):
    def test_first_deviation_uses_only_current_prefix(self):
        self.assertEqual(deviation(np.array([[12], [18]]), [12]).tolist(), [0, 1])
        self.assertEqual(deviation(np.array([[12, 402, 704], [12, 403, 541], [55, 460, 601]]),
                                   [12, 402, 704]).tolist(), [0, 2, 1])

    def test_oracle_matches_exhaustive_tie_orderings(self):
        # 14 strict competitors, Gold plus three score ties, one lower value.
        values = np.array(list(range(14, 0, -1)) + [0, 0, 0, 0, -1], dtype=float)
        gold = 17
        foreign = np.zeros(len(values), dtype=bool)
        foreign[[0, 1, 2, 14]] = True
        for amount in (1, 3, 5, "all"):
            remove = set(np.flatnonzero(foreign)) if amount == "all" else set(
                sorted(np.flatnonzero(foreign & (values > values[gold])), key=lambda i: -values[i])[:amount])
            survivors = [i for i in range(len(values)) if i not in remove]
            greater = [i for i in survivors if values[i] > values[gold]]
            tied = [i for i in survivors if values[i] == values[gold]]
            outcomes = []
            for order in itertools.permutations(tied):
                outcomes.append(gold in (greater + list(order))[:15])
            bounds = oracle_bounds(14, 4, 3, 1, amount)
            self.assertEqual(bounds["guaranteed"], all(outcomes))
            self.assertEqual(bounds["possible"], any(outcomes))

    def test_no_op_cannot_rescue_originally_pruned_tied_gold(self):
        bounds = oracle_bounds(14, 4, 0, 0, 1)
        self.assertFalse(bounds["guaranteed"])
        self.assertFalse(bounds["possible"])

    def test_wrong_tokens_from_gold_parent_are_not_removed(self):
        # Sixteen higher scores belong to the correct parent, so deleting every
        # foreign-parent candidate still cannot bring Gold into Top-15.
        bounds = oracle_bounds(20, 1, 4, 0, "all")
        self.assertFalse(bounds["possible"])


if __name__ == "__main__":
    unittest.main()
