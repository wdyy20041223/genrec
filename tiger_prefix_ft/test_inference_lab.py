"""Small independent correctness checks for the isolated inference experiments."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from .inference_lab import CandidateRecorder, alternative_search, ranks_for
from .test_sid_trie import ScriptedT5
from .inference_rerank import PREFIX_SPACE, SID_SPACE, make_features, prefix_keys, sid_keys
from .inference_priors import PrefixPrior
from .inference_rerank import fuse_caches, score_candidates, rank_candidates


def reference(tiger, ids, mask, widths, weights, sibling=0., depths=(2,)):
    """Uncached single-user full-prefix decoding, independently sorted in Python."""
    encoded = tiger.model.encoder(input_ids=ids, attention_mask=mask)[0]
    result, components = [], []
    for i in range(len(ids)):
        kept = [([0], [], 0.)]
        for step, width in enumerate(widths):
            candidates = []
            for prefix, prior, score in kept:
                out = tiger.model(encoder_outputs=(encoded[i:i+1],), attention_mask=mask[i:i+1],
                                  decoder_input_ids=torch.tensor([prefix]), use_cache=False)
                lp = out.logits[0, -1].float().log_softmax(-1)
                token_order = sorted(range(len(lp)), key=lambda token: float(lp[token]), reverse=True)
                for rank, token in enumerate(token_order):
                    new_score = score + weights[step] * float(lp[token])
                    penalty = sibling * np.log1p(rank) if step + 1 in depths else 0.
                    candidates.append((prefix + [token], prior + [float(lp[token])], new_score, new_score - penalty))
            kept = [row[:3] for row in sorted(candidates, key=lambda row: row[3], reverse=True)[:width]]
        result.append([row[0] for row in kept])
        components.append([row[1] for row in kept])
    return torch.tensor(result), torch.tensor(components)


class InferenceChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.tiger = Tiger(dict(num_layers=1, num_decoder_layers=1, d_model=16, d_ff=32,
                               d_kv=8, num_heads=2, vocab_size=17, dropout_rate=0., sem_id_dim=3)).eval()
        self.ids = torch.tensor([[0, 5, 6], [3, 7, 8]])
        self.mask = (self.ids != 0).long()

    @torch.inference_mode()
    def test_zero_adjustment_reproduces_original_and_uncached(self):
        for widths in ((3, 5, 4), (4, 4, 4)):
            got, lp, raw, weighted, _ = alternative_search(self.tiger, self.ids, self.mask, widths)
            original = self.tiger.generate(self.ids, self.mask, num_beams=widths[-1], beam_sizes=widths)
            expected, expected_lp = reference(self.tiger, self.ids, self.mask, widths, (1, 1, 1))
            self.assertTrue(torch.equal(got, original))
            self.assertTrue(torch.equal(got.view_as(expected), expected))
            torch.testing.assert_close(lp, expected_lp, atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(raw, weighted)

    @torch.inference_mode()
    def test_weighted_and_sibling_decisions_match_uncached_reference(self):
        for weights, sibling in (((.6, 1, 1.2), 0.), ((1, 1, 1), .4)):
            recorder = CandidateRecorder(16)
            got, lp, _, _, _ = alternative_search(self.tiger, self.ids, self.mask, (3, 5, 4), weights, sibling, recorder=recorder)
            expected, expected_lp = reference(self.tiger, self.ids, self.mask, (3, 5, 4), weights, sibling)
            self.assertTrue(torch.equal(got.view_as(expected), expected))
            torch.testing.assert_close(lp, expected_lp, atol=2e-6, rtol=1e-6)
            pool, pool_lp = reference(self.tiger, self.ids, self.mask, (3, 5, 16), weights, sibling)
            self.assertTrue(np.array_equal(recorder.pool_tokens, pool[:, :, 1:].numpy()))
            np.testing.assert_allclose(recorder.pool_logps, pool_lp.numpy(), atol=2e-6, rtol=1e-6)

    @torch.inference_mode()
    def test_equal_score_pool_preserves_actual_original_order(self):
        model = ScriptedT5().eval()
        model.script.zero_()
        recorder = CandidateRecorder(16)
        model._beam_trace_recorder = recorder
        try:
            original = model.generate(self.ids, max_length=4, num_beams=4,
                                      num_return_sequences=4, beam_sizes=(3, 5, 4))
        finally:
            del model._beam_trace_recorder
        self.assertTrue(np.array_equal(original.view(2, 4, 4)[:, :, 1:].numpy(), recorder.pool_tokens[:, :4]))
        for paths in recorder.pool_tokens:
            self.assertEqual(len({tuple(path) for path in paths}), 16)

    @torch.inference_mode()
    def test_passive_pool_own_conditionals_and_final_width_invariance(self):
        recorder = CandidateRecorder(16)
        self.tiger.model._beam_trace_recorder = recorder
        try:
            output = self.tiger.generate(self.ids, self.mask, num_beams=4, beam_sizes=(3, 5, 4))
        finally:
            del self.tiger.model._beam_trace_recorder
        expected, expected_lp = reference(self.tiger, self.ids, self.mask, (3, 5, 16), (1, 1, 1))
        self.assertTrue(np.array_equal(recorder.pool_tokens, expected[:, :, 1:].numpy()))
        np.testing.assert_allclose(recorder.pool_logps, expected_lp.numpy(), atol=2e-6, rtol=1e-6)
        self.assertTrue(torch.equal(output, expected[:, :4].flatten(0, 1)))
        self.assertTrue(torch.equal(self.tiger.generate(self.ids, self.mask, 8, (3, 5, 8)).view(2, 8, 4)[:, :4], output.view(2, 4, 4)))
        self.assertEqual(ranks_for(recorder.pool_tokens, recorder.pool_tokens[:, 2]).tolist(), [3, 3])

    def test_history_and_train_features_have_no_target_dependency(self):
        candidates = np.array([[[1, 257, 513], [2, 258, 514]]], np.int16)
        histories = np.zeros((1, 50, 3), np.int16)
        histories[0, -1] = candidates[0, 0]
        stats = {"samples": np.array(8), "code1_counts": np.zeros(769, np.int32)}
        stats["code1_counts"][[1, 2]] = [5, 3]
        for name, query, space in (("sid", sid_keys(candidates)[0], SID_SPACE),
                                    ("prefix", prefix_keys(candidates)[0], PREFIX_SPACE)):
            stats[f"{name}_target_keys"] = query
            stats[f"{name}_target_counts"] = np.array([5, 3])
            stats[f"{name}_pair_keys"] = np.array([query[0] * space + query[1]])
            stats[f"{name}_pair_counts"] = np.array([2])
            stats[f"{name}_context_keys"] = query[:1]
            stats[f"{name}_context_counts"] = np.array([2])
        cache = {"tokens": candidates, "histories": histories, "targets": candidates[:, 0]}
        # This is a synthetic boundary check, independent of the private Beauty export.
        catalog = candidates[0].astype(np.int64) - np.array([1, 257, 513])
        with patch("tiger_prefix_ft.inference_rerank.np.load", return_value=catalog):
            first = make_features(cache, stats)
            cache["targets"] = candidates[:, 1]
            second = make_features(cache, stats)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["seen"].tolist(), [[1., 0.]])
        self.assertEqual(first["sid_transition"][0, 0], 0.)
        self.assertGreater(first["sid_transition"][0, 1], 0.)

    def test_prefix_prior_uses_train_projection_and_preserves_collided_history(self):
        candidates = np.array([[1, 257, 513], [2, 258, 514]], np.int64)
        keys = sid_keys(candidates)
        stats = {"samples": np.array(8), "code1_counts": np.zeros(769, np.int32),
                 "sid_pair_keys": np.array([keys[0] * SID_SPACE + keys[1]]), "sid_pair_counts": np.array([2]),
                 "sid_target_keys": keys, "sid_target_counts": np.array([5, 3]),
                 "prefix_target_keys": prefix_keys(candidates), "prefix_target_counts": np.array([5, 3])}
        stats["code1_counts"][[1, 2]] = [5, 3]
        prior = PrefixPrior(stats, candidates, [.1, .2, .4], True)
        prior.prepare(torch.tensor([[1, 257, 513]]), torch.ones(1, 3, dtype=torch.long))
        expected = np.log1p(2 / (20 * (4 / 10)))
        first = prior(1, torch.tensor([[[0]]]), torch.zeros(1, 1, 769))
        self.assertAlmostEqual(float(first[0, 0, 2]), .1 * expected, places=6)
        second = prior(2, torch.tensor([[[0, 2]]]), torch.zeros(1, 1, 769))
        self.assertAlmostEqual(float(second[0, 0, 258]), .2 * expected, places=6)
        last = prior(3, torch.tensor([[[0, 1, 257], [0, 2, 258]]]), torch.zeros(1, 2, 769))
        self.assertEqual(float(last[0, 0, 513]), -1000.)
        self.assertAlmostEqual(float(last[0, 1, 514]), .4 * expected, places=6)
        collision = PrefixPrior(stats, np.concatenate((candidates, candidates[:1])), [.1, .2, .4], True)
        collision.prepare(torch.tensor([[1, 257, 513]]), torch.ones(1, 3, dtype=torch.long))
        last = collision(3, torch.tensor([[[0, 1, 257]]]), torch.zeros(1, 1, 769))
        self.assertEqual(float(last[0, 0, 513]), 0.)
        recent = PrefixPrior(stats, candidates, [0., 0., .4], False, history_items=3)
        recent.prepare(torch.tensor([[1, 257, 513, 2, 258, 514]]), torch.ones(1, 6, dtype=torch.long))
        bonus = recent(3, torch.tensor([[[0, 2, 258]]]), torch.zeros(1, 1, 769))
        self.assertAlmostEqual(float(bonus[0, 0, 514]), .4 * .5 * expected, places=6)
        recent.prepare(torch.tensor([[1, 257, 513, 1, 257, 513]]), torch.ones(1, 6, dtype=torch.long))
        bonus = recent(3, torch.tensor([[[0, 2, 258]]]), torch.zeros(1, 1, 769))
        self.assertAlmostEqual(float(bonus[0, 0, 514]), .4 * expected, places=6)
        zero = PrefixPrior(stats, candidates, [0., 0., 0.], False, history_items=3)
        ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
        mask = torch.ones_like(ids)
        zero.prepare(ids, mask)
        with torch.inference_mode():
            actual = alternative_search(self.tiger, ids, mask, (3, 5, 4), adjuster=zero)[0]
            baseline = self.tiger.generate(ids, mask, 4, (3, 5, 4))
        self.assertTrue(torch.equal(actual, baseline))

    def test_fusion_deduplicates_sid_preserves_source_and_excludes_padding(self):
        def cache(start, scale, name):
            x = np.arange(start, start + 10)
            tokens = np.stack((x, x + 256, x + 512), -1)[None].astype(np.int16)
            lp = np.full((1, 10, 3), -scale, np.float32)
            meta = {"split": "valid", "checkpoint_sha256": "same", "sid_sha256": "same", "data_sha256": "same",
                    "seconds": 1., "peak_gpu_mib": 1., "samples": 1, "widths": [15, 15, 15], "survival_counts": [1, 1, 1]}
            return {"tokens": tokens, "logps": lp, "raw_scores": lp.sum(-1), "search_scores": lp.sum(-1),
                    "depth_ranks": np.array([[1, 1, 1]], np.int16), "sample_ids": np.array(["valid_synthetic"]),
                    "histories": np.zeros((1, 50, 3), np.int16), "targets": np.array([[8, 264, 520]], np.int16),
                    "meta": meta, "source_cache": name}
        a, b = cache(1, 2, "primary"), cache(6, 1, "secondary")
        with tempfile.TemporaryDirectory() as folder:
            c = fuse_caches(a, b, Path(folder) / "union")
        self.assertEqual(c["candidate_mask"].sum(), 15)
        self.assertTrue(np.array_equal(c["tokens"][:, :10], a["tokens"]))
        self.assertTrue(np.array_equal(c["raw_scores"][:, :10], a["raw_scores"]))
        order = rank_candidates(c, score_candidates(c, {"pool": 20}, {}), {"pool": 20})
        self.assertTrue(c["candidate_mask"][0, order[0, :10]].all())
        self.assertEqual(len({tuple(row) for row in c["tokens"][0, order[0, :10]]}), 10)
        self.assertNotIn("metrics", c["meta"])


if __name__ == "__main__":
    unittest.main()
