"""Check per-depth Beam expansion against independent uncached decoding."""
import unittest

import torch

from cotiger.core.modules.sid_trie import SIDTrie
from cotiger.core.modules.t5 import T5Config, T5ForConditionalGeneration
from .beam_diagnosis import PrefixRecorder, paired, summarize
from .test_sid_trie import ScriptedT5


def reference_search(model, ids, mask, widths):
    """Recompute complete decoder prefixes without KV cache or generate()."""
    encoded = model.encoder(input_ids=ids, attention_mask=mask)[0]
    paths, final_scores = [], []
    for row in range(len(ids)):
        kept = [([0], 0.0)]
        for width in widths:
            prefixes = torch.tensor([path for path, _ in kept])
            output = model(encoder_outputs=(encoded[row:row + 1].expand(len(kept), -1, -1),),
                           attention_mask=mask[row:row + 1].expand(len(kept), -1),
                           decoder_input_ids=prefixes, use_cache=False)
            logp = output.logits[:, -1].float().log_softmax(-1)
            cumulative = logp + torch.tensor([score for _, score in kept])[:, None]
            candidates = [(path + [token], float(cumulative[parent, token]))
                          for parent, (path, _) in enumerate(kept) for token in range(logp.shape[-1])]
            kept = sorted(candidates, key=lambda value: value[1], reverse=True)[:width]
        paths.append([path for path, _ in kept])
        final_scores.append([score for _, score in kept])
    return torch.tensor(paths), torch.tensor(final_scores)


class BeamSizeChecks(unittest.TestCase):
    def test_wider_middle_recovers_a_real_search_path(self):
        model = ScriptedT5().eval()
        model.script.fill_(-20)
        model.script[0, 1], model.script[0, 2] = 4, 3
        model.script[1, 3], model.script[1, 4], model.script[1, 5] = 4, 3.9, 3.8
        model.script[2:5] = 0
        model.script[5, 6] = 5
        ids = torch.tensor([[1, 2]])
        target = torch.tensor([[1, 5, 6]])
        observed = {}
        for name, widths in (("fixed", None), ("wider", [2, 4, 2])):
            recorder = PrefixRecorder(target)
            shape_log = []

            def record(**trace):
                recorder(**trace)
                shape_log.append((trace["num_beams"], trace["selected_cumulative_scores"].shape[1]))

            model._beam_trace_recorder = record
            generated = model.generate(ids, max_length=4, num_beams=2, num_return_sequences=2, beam_sizes=widths)
            observed[name] = recorder.finish(generated, ["synthetic_0"], name)[0]
            if widths:
                self.assertEqual(shape_log, [(2, 2), (2, 4), (4, 2)])
                self.assertEqual(generated[0].tolist(), [0, 1, 5, 6])
                self.assertEqual(observed[name]["gold_rank_depth2"], 3)
        self.assertEqual(observed["fixed"]["first_prune_depth"], 2)
        self.assertEqual(observed["wider"]["first_prune_depth"], 0)

    def test_real_kv_cache_resize_matches_full_prefix_reference(self):
        ids = torch.tensor([[0, 5, 6], [7, 8, 9]])
        mask = (ids != 0).long()
        for first_only in (True, False):
            torch.manual_seed(13)
            model = T5ForConditionalGeneration(T5Config(vocab_size=13, d_model=16, d_ff=32,
                d_kv=8, num_heads=2, num_layers=2, num_decoder_layers=2,
                dropout_rate=0, eos_token_id=0, first_beam_only=first_only)).eval()
            with torch.inference_mode():
                for widths, returned in (([2, 4, 2], 2), ([2, 4, 3], 1)):
                    with self.subTest(first_only=first_only, widths=widths):
                        expected, scores = reference_search(model, ids, mask, widths)
                        trace_scores = []
                        model._beam_trace_recorder = lambda **step: trace_scores.append(step["selected_cumulative_scores"].clone())
                        actual = model.generate(ids, mask, max_length=4, num_beams=2,
                                                num_return_sequences=returned, beam_sizes=widths)
                        self.assertTrue(torch.equal(actual.view(2, returned, 4), expected[:, :returned]))
                        torch.testing.assert_close(trace_scores[-1], scores, atol=2e-5, rtol=1e-5)

    def test_constant_width_profile_preserves_existing_generation(self):
        model = ScriptedT5().eval()
        ids = torch.tensor([[1, 2], [2, 1]])
        original = model.generate(ids, max_length=4, num_beams=2, num_return_sequences=2)
        explicit = model.generate(ids, max_length=4, num_beams=2, num_return_sequences=2, beam_sizes=[2, 2, 2])
        self.assertTrue(torch.equal(original, explicit))

    def test_single_first_beam_can_expand(self):
        model = ScriptedT5().eval()
        shapes = []
        model._beam_trace_recorder = lambda **step: shapes.append(
            (step["num_beams"], step["selected_cumulative_scores"].shape[1]))
        output = model.generate(torch.tensor([[1]]), max_length=4, num_beams=1,
                                num_return_sequences=1, beam_sizes=[1, 2, 1])
        self.assertEqual(shapes, [(1, 1), (1, 2), (2, 1)])
        self.assertEqual(output.shape, (1, 4))

    def test_invalid_profiles_and_trie_combination_are_explicit(self):
        model = ScriptedT5().eval()
        ids = torch.tensor([[1]])
        for widths in ([2, 4], [2, 0, 2], [2, 4.5, 2]):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                model.generate(ids, max_length=4, beam_sizes=widths)
        with self.assertRaisesRegex(ValueError, "final Beam width"):
            model.generate(ids, max_length=4, num_return_sequences=3, beam_sizes=[2, 4, 2])
        with self.assertRaisesRegex(ValueError, "Trie to be disabled"):
            model.generate(ids, max_length=4, beam_sizes=[2, 4, 2],
                           sid_trie=SIDTrie.from_semantic_ids([[0, 0, 0]], codebook_size=3))

    def test_top30_survival_is_distinct_from_top15_and_top10(self):
        row = {"sample_id": "synthetic_0", "survived_depth1": True,
               "survived_depth2": True, "survived_depth3": True,
               "first_prune_depth": 0, "gold_rank_depth3": 20}
        result = summarize([row], dict.fromkeys(("Recall@5", "Recall@10", "NDCG@5", "NDCG@10"), 0.0), 30)
        self.assertEqual(result["target_in_final_beam_rate"], 1)
        self.assertEqual(result["target_in_final_top15_rate"], 0)
        before = {**row, "gold_rank_depth3": 11}
        comparison = paired([before], [row])
        self.assertEqual(comparison["net_final_beam_hits"], 0)
        self.assertEqual(comparison["net_top15_hits"], -1)


if __name__ == "__main__":
    unittest.main()
