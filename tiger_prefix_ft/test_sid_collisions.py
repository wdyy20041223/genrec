import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from .analyze_sid_collisions import analyze_collisions, load_sid_file


class CollisionStatisticsTests(unittest.TestCase):
    def test_three_code_collision_counts_include_every_member(self):
        sids = np.array([[1, 2, 3]] * 3 + [[1, 2, 4]] * 2 + [[7, 8, 9]])
        summary, groups, members, distribution, tokens = analyze_collisions(sids)
        self.assertEqual(summary["total_items"], 6)
        self.assertEqual(summary["unique_semantic_prefixes"], 3)
        self.assertEqual(summary["collision_groups"], 2)
        self.assertEqual(summary["colliding_items"], 5)
        self.assertEqual(summary["non_colliding_items"], 1)
        self.assertEqual(summary["excess_items_beyond_one_per_prefix"], 3)
        self.assertEqual(summary["collision_pairs"], 4)
        self.assertEqual(summary["max_group_size"], 3)
        self.assertEqual(groups.item_count.tolist(), [3, 2])
        self.assertEqual(members.item_index.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(distribution.group_size.tolist(), [1, 2, 3])
        self.assertEqual(distribution.item_count.tolist(), [1, 2, 3])
        self.assertTrue(members.sid_3.isna().all())
        self.assertTrue(groups.fourth_tokens.isna().all())
        self.assertIsNone(summary["remaining_collision_groups_after_fourth"])
        self.assertEqual(len(tokens), 0)

    def test_fourth_codes_do_not_prevent_semantic_collisions(self):
        sids = np.array([[4, 7, 8, token] for token in (0, 2, 3, 4)])
        summary, groups, members, _, tokens = analyze_collisions(sids)
        self.assertEqual(summary["collision_groups"], 1)
        self.assertEqual(summary["colliding_items"], 4)
        self.assertEqual(summary["collision_pairs"], 6)
        self.assertEqual(json.loads(groups.iloc[0].fourth_tokens), [0, 2, 3, 4])
        self.assertEqual(members.sid_3.tolist(), [0, 2, 3, 4])
        self.assertEqual(summary["remaining_collision_groups_after_fourth"], 0)
        self.assertEqual(tokens.item_count.tolist(), [1, 1, 1, 1])
        self.assertEqual(tokens.collision_group_count.tolist(), [1, 1, 1, 1])

    def test_repeated_fourth_codes_are_reported(self):
        sids = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 2], [1, 1, 1, 3], [2, 2, 2, 0]])
        summary, groups, _, _, tokens = analyze_collisions(sids)
        self.assertEqual(summary["remaining_collision_groups_after_fourth"], 1)
        self.assertEqual(summary["remaining_colliding_items_after_fourth"], 2)
        self.assertEqual(summary["remaining_collision_pairs_after_fourth"], 1)
        self.assertEqual(json.loads(groups.iloc[0].fourth_token_counts), {"0": 2, "2": 1, "3": 1})
        self.assertEqual(groups.iloc[0].repeated_fourth_token_items, 2)
        self.assertEqual(tokens.iloc[0].item_count, 3)
        self.assertEqual(tokens.iloc[0].colliding_item_count, 2)

    def test_no_collisions_and_empty_inputs_have_valid_tables(self):
        summary, groups, members, distribution, _ = analyze_collisions(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]))
        self.assertEqual(summary["collision_groups"], 0)
        self.assertEqual(summary["colliding_item_rate"], 0.0)
        self.assertEqual(summary["max_group_size"], 1)
        self.assertEqual(len(groups), 0)
        self.assertEqual(len(members), 0)
        self.assertEqual(distribution.item_count.tolist(), [3])
        summary, groups, members, distribution, _ = analyze_collisions(np.empty((0, 3), dtype=int))
        self.assertEqual(summary["total_items"], 0)
        self.assertEqual(summary["max_group_size"], 0)
        self.assertIsNone(summary["colliding_item_rate"])
        self.assertEqual(len(distribution), 0)
        self.assertIn("fourth_tokens", groups.columns)
        self.assertIn("sid_3", members.columns)

    def test_same_item_id_cannot_be_counted_twice(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze_collisions(np.array([[0, 0, 0], [0, 0, 0]]), [9, 9])

    def test_invalid_codes_and_widths_are_rejected(self):
        for invalid in (np.array([[0, 0, np.nan]]), np.array([[0, 0, 0.5]]),
                        np.array([[0, 0, -1]]), np.array([[0, 0]]), np.array([[0, 0, 0, 0, 0]])):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    analyze_collisions(invalid)

    def test_loading_npy_csv_and_parquet_preserves_item_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sids = np.array([[4, 7, 8, 0], [4, 7, 8, 2]])
            np.save(root / "ids.npy", sids)
            loaded, indices, labels, _ = load_sid_file(root / "ids.npy")
            self.assertTrue(np.array_equal(loaded, sids))
            self.assertEqual(indices.tolist(), [0, 1])
            self.assertEqual(labels, [None, None])
            frame = pd.DataFrame(sids, columns=["sid_0", "sid_1", "sid_2", "sid_3"])
            frame["item_index"] = [42, 13]
            frame["asin"] = ["0000012345", None]
            frame["sid"] = ["4-7-8-0", "4-7-8-2"]
            frame.to_csv(root / "ids.csv", index=False)
            frame.to_parquet(root / "ids.parquet", index=False)
            for name in ("ids.csv", "ids.parquet"):
                loaded, indices, labels, _ = load_sid_file(root / name)
                self.assertTrue(np.array_equal(loaded, sids))
                self.assertEqual(indices.tolist(), [42, 13])
                self.assertEqual(labels, ["0000012345", None])
                _, groups, members, _, _ = analyze_collisions(loaded, indices, labels)
                self.assertEqual(json.loads(groups.iloc[0].item_indices), [13, 42])
                self.assertEqual(members.sid_3.tolist(), [2, 0])

    def test_inconsistent_sid_list_and_wide_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.parquet"
            pd.DataFrame({"sid_0": [1], "sid_1": [2], "sid_2": [3], "sid": ["1-2-4"]}).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_sid_file(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
