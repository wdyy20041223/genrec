"""Small regression fixtures only; these are not experimental training data."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .negative_sampler import NegativeSampler, ROOT, SIDCatalog
from .prepare_data import CATALOG_SCHEMA, HARD_SCHEMA, TRAIN_SCHEMA, mine_depth


def catalog_rows():
    tokens = [(12, 402, 704), (12, 402, 705), (12, 403, 541),
              (12, 404, 550), (55, 460, 600), (117, 453, 536)]
    return [{"raw_sid": [p[0] - 1, p[1] - 257, p[2] - 513],
             "offset_tokens": list(p), "collision_group_size": 1} for p in tokens]


class PrefixPipelineTests(unittest.TestCase):
    def setUp(self):
        self.catalog = SIDCatalog(catalog_rows())
        self.record = {"sample_id": "train_0000000", "dataset_index": 0,
                       "target_tokens": [12, 402, 704]}
        self.data = {
            "next_token_logprobs": np.full((1, 3, 15, 769), -100, dtype=np.float32),
            "beam_scores_before": np.full((1, 3, 15), -1e9, dtype=np.float32),
            "parent_prefix_tokens_with_start": np.full((1, 3, 15, 3), -1, dtype=np.int16),
            "active_beam_mask": np.zeros((1, 3, 15), dtype=bool),
            "selected_parent_indices": np.zeros((1, 3, 15), dtype=np.int16),
            "selected_token_ids": np.zeros((1, 3, 15), dtype=np.int16),
            "gold_cumulative_logprobs": np.asarray([[-.4, -5, -6.1]], dtype=np.float32),
            "gold_next_token_logprobs": np.full((1, 3, 769), -100, dtype=np.float32),
        }
        self.data["parent_prefix_tokens_with_start"][:, :, :, 0] = 0
        self.data["active_beam_mask"][0, :, 0] = True
        self.data["beam_scores_before"][0, :, 0] = [0, -.4, -1.7]
        self.data["parent_prefix_tokens_with_start"][0, 1, 0] = [0, 12, -1]
        self.data["parent_prefix_tokens_with_start"][0, 2, 0] = [0, 12, 403]
        self.data["next_token_logprobs"][0, 0, 0, [0, 12, 55]] = [10, -.4, -2]
        self.data["next_token_logprobs"][0, 1, 0, 403] = -1.3
        self.data["next_token_logprobs"][0, 2, 0, 541] = -1.6
        self.data["selected_token_ids"][0, 1] = 403
        self.data["selected_token_ids"][0, 2] = 541
        self.data["gold_next_token_logprobs"][0, 1, 403] = -.2
        self.data["gold_next_token_logprobs"][0, 2, 705] = -2

    def test_positive_first_token_and_special_token_are_not_negatives(self):
        rows, _ = mine_depth(self.record, self.data, 0, 1, self.catalog)
        self.assertNotIn([12], [r["negative_tokens"] for r in rows])
        self.assertNotIn([0], [r["negative_tokens"] for r in rows])

    def test_depth_specific_scores_and_beam_precedence(self):
        depth2, _ = mine_depth(self.record, self.data, 0, 2, self.catalog)
        a = next(r for r in depth2 if r["negative_tokens"] == [12, 403])
        depth3, _ = mine_depth(self.record, self.data, 0, 3, self.catalog)
        b = next(r for r in depth3 if r["negative_tokens"] == [12, 403, 541])
        self.assertEqual(a["score"], float(np.float32(-.4) + np.float32(-1.3)))
        self.assertEqual(b["score"], float(np.float32(-1.7) + np.float32(-1.6)))
        self.assertNotEqual(a["score"], b["score"])
        self.assertEqual(a["source"], "beam")
        self.assertTrue(a["survived_beam"])
        self.assertTrue(a["from_gold_parent"])
        self.assertFalse(b["from_gold_parent"])

    def test_inactive_parent_is_never_mined(self):
        self.data["parent_prefix_tokens_with_start"][0, 1, 1] = [0, 55, -1]
        self.data["beam_scores_before"][0, 1, 1] = 10
        self.data["next_token_logprobs"][0, 1, 1, 460] = 100
        rows, _ = mine_depth(self.record, self.data, 0, 2, self.catalog)
        self.assertNotIn([55, 460], [r["negative_tokens"] for r in rows])

    def test_gold_conditional_missing_actual_parent(self):
        rows, _ = mine_depth(self.record, self.data, 0, 3, self.catalog)
        candidate = next(r for r in rows if r["negative_tokens"] == [12, 402, 705])
        self.assertEqual(candidate["source"], "gold_conditional")
        self.assertEqual(candidate["parent_beam_index"], -1)
        self.assertFalse(candidate["survived_beam"])
        self.assertTrue(candidate["from_gold_parent"])

    def test_duplicate_catalog_sid_is_rejected(self):
        with self.assertRaises(ValueError):
            SIDCatalog(catalog_rows() + catalog_rows()[:1])

    def test_sampler_dedup_shortfalls_depth1_and_wrong_user(self):
        parent = ROOT / "out/tiger_prefix_ft"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            path = Path(temporary)
            train = dict(self.record, input_ids=[0, 12, 360, 554], attention_mask=[0, 1, 1, 1],
                         target_sid=[11, 145, 191], target_item_id=0,
                         tensor_shard="train/shard_fixture.npz", tensor_row=0)
            hard, _ = mine_depth(self.record, self.data, 0, 2, self.catalog)
            hard = hard[:1]
            pq.write_table(pa.Table.from_pylist([train], schema=TRAIN_SCHEMA), path / "train_samples.parquet")
            pq.write_table(pa.Table.from_pylist(catalog_rows(), schema=CATALOG_SCHEMA), path / "sid_catalog.parquet")
            pq.write_table(pa.Table.from_pylist(hard, schema=HARD_SCHEMA), path / "hard_negatives.parquet")
            sampler = NegativeSampler(path)
            group = sampler.sample("train_0000000", 2, 20, 20, 20)
            prefixes = [tuple(n["tokens"]) for n in group["negatives"]]
            self.assertEqual(len(prefixes), len(set(prefixes)))
            self.assertEqual(group["actual_counts"], {"easy": 2, "medium": 1, "hard": 1})
            self.assertEqual(sampler.sample("train_0000000", 1, 2, 2, 4)["actual_counts"]["medium"], 0)
            self.assertEqual(sampler.sample("train_0000000", 3, 0, 0, 8)["actual_counts"]["hard"], 0)
            hard[0]["sample_id"] = "train_0000001"
            pq.write_table(pa.Table.from_pylist(hard, schema=HARD_SCHEMA), path / "hard_negatives.parquet")
            with self.assertRaises(ValueError):
                NegativeSampler(path).sample("train_0000000", 2)


if __name__ == "__main__":
    unittest.main()
