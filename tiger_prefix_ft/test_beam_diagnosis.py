"""Synthetic path-integrity checks, separate from the real Validation smoke run."""
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .beam_diagnosis import PrefixRecorder, sample_schema, summarize


def synthetic_paths(corrupt_scores=False):
    targets = torch.tensor([[9, 2, 3], [4, 9, 3], [1, 2, 3], [1, 7, 6]])
    size, beams, vocab = 4, 2, 16
    recorder = PrefixRecorder(targets)
    parents = torch.zeros(size, beams, 1, dtype=torch.long)
    parent_scores = torch.tensor([[0.0, -1e9]]).expand(size, -1).clone()
    for depth in (1, 2, 3):
        if depth == 1:
            logits = torch.full((size, vocab), -100.0)
            logits[:, 1], logits[:, 4] = 5.0, 4.9
        elif depth == 2:
            logits = torch.zeros(size, beams, vocab)
            logits[:, 0] = -100.0
            logits[:, 0, 2], logits[:, 0, 7] = 5.0, 4.9
        else:
            logits = torch.zeros(size, beams, vocab)
            logits[:, 0, 3] = -100.0
            logits[:, 1] = -100.0
            logits[:, 1, 6], logits[:, 1, 7] = 10.0, 9.9
        logp = logits.log_softmax(-1)
        expansions = (logp[:, None] if depth == 1 else logp) + parent_scores[:, :, None]
        scores, indices = expansions.flatten(1).topk(beams, sorted=True)
        observed_scores = scores.clone()
        if corrupt_scores and depth == 2:
            observed_scores[0, 0] += 0.125
        recorder(depth=depth, batch_size=size, num_beams=beams, first_beam_only=depth == 1,
                 parent_tokens=parents.flatten(0, 1), parent_scores=parent_scores,
                 next_token_logprobs=logp.flatten(0, 1) if depth > 1 else logp,
                 selected_flat_indices=indices, selected_cumulative_scores=observed_scores)
        parent_index, token = indices // vocab, indices % vocab
        parents = torch.cat((parents.gather(1, parent_index[:, :, None].expand(-1, -1, depth)), token[:, :, None]), -1)
        parent_scores = scores
    rows = recorder.finish(parents.flatten(0, 1), [f"synthetic_{i}" for i in range(size)], "synthetic")
    return rows, parents


class BeamDiagnosisChecks(unittest.TestCase):
    def test_first_prune_and_missing_future_scores(self):
        rows, _ = synthetic_paths()
        self.assertEqual([r["first_prune_depth"] for r in rows], [1, 2, 3, 0])
        self.assertIsNone(rows[0]["gold_rank_depth1"])
        self.assertIsNotNone(rows[0]["gold_candidate_depth1"])
        self.assertIsNone(rows[0]["gold_candidate_depth2"])
        self.assertIsNone(rows[0]["gold_margin_depth3"])
        self.assertIsNotNone(rows[1]["gold_candidate_depth2"])
        self.assertIsNone(rows[1]["gold_candidate_depth3"])
        self.assertIsNone(rows[2]["gold_score_depth3"])

    def test_early_survival_cannot_be_inferred_from_final_prefixes(self):
        rows, final = synthetic_paths()
        self.assertTrue(rows[2]["survived_depth2"])
        self.assertFalse((final[2, :, 1:3] == torch.tensor([1, 2])).all(-1).any())
        self.assertTrue(rows[3]["survived_depth3"])

    def test_wrong_depth_or_corrupt_score_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "cumulative score"):
            synthetic_paths(corrupt_scores=True)

    def test_top15_survival_is_not_recall10(self):
        row = {"survived_depth1": True, "survived_depth2": True,
               "survived_depth3": True, "first_prune_depth": 0, "gold_rank_depth3": 11}
        result = summarize([row], dict(zip(("Recall@5", "Recall@10", "NDCG@5", "NDCG@10"), [0.0] * 4)))
        self.assertEqual(result["target_in_final_top15_rate"], 1.0)
        self.assertEqual(result["metrics"]["Recall@10"], 0.0)

    def test_nullable_scores_survive_parquet_roundtrip(self):
        rows, _ = synthetic_paths()
        table = pa.Table.from_pylist(rows, schema=sample_schema())
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        self.assertEqual(pq.read_table(pa.BufferReader(sink.getvalue())).to_pylist(), rows)


if __name__ == "__main__":
    unittest.main()
