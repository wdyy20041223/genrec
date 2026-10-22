"""Small catalog and scripted-logit checks for constrained TIGER generation."""
import io
import unittest

import torch
from torch import nn

from cotiger.core.models.tiger import Tiger
from cotiger.core.modules.sid_trie import SIDTrie
from cotiger.core.modules.t5 import T5Config, T5ForConditionalGeneration, _Seq2SeqLMOutput


RAW = [[0, 0, 0], [0, 1, 1], [1, 0, 2], [1, 2, 0], [1, 2, 0]]
PATHS = {(1, 4, 7), (1, 5, 8), (2, 4, 9), (2, 6, 7)}


class DummyEncoder(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        return (torch.zeros(len(input_ids), 1, 8, device=input_ids.device),)


class ScriptedT5(T5ForConditionalGeneration):
    def __init__(self, first_beam_only=True):
        super().__init__(T5Config(vocab_size=10, d_model=8, d_ff=16, d_kv=4, num_heads=2,
            num_layers=1, num_decoder_layers=1, dropout_rate=0, eos_token_id=0,
            first_beam_only=first_beam_only))
        self.encoder = DummyEncoder()
        logits = torch.full((10, 10), -6.0)
        logits[0, 1], logits[0, 2], logits[0, 3] = 3.0, 2.3, 9.0
        logits[1, 4], logits[1, 5], logits[1, 6] = 3.0, 2.7, 8.0
        logits[2, 4], logits[2, 6], logits[2, 5] = 4.0, 3.7, 9.0
        logits[4, 7], logits[4, 9], logits[4, 8] = 1.0, 2.2, 10.0
        logits[5, 8], logits[5, 9] = 0.7, 7.0
        logits[6, 7], logits[6, 8] = 0.2, 6.5
        self.register_buffer("script", logits)

    def forward(self, decoder_input_ids, **kwargs):
        return _Seq2SeqLMOutput(None, self.script[decoder_input_ids[:, -1]][:, None, :], ())


def reference_search(model, beams):
    # Independent Python search, retaining original full-vocabulary log probs.
    logp = model.script.log_softmax(-1)
    kept = [((), 0.0)]
    for depth in (1, 2, 3):
        expanded = {}
        for parent, score in kept:
            for path in PATHS:
                if path[:depth - 1] == parent:
                    prefix = path[:depth]
                    expanded[prefix] = score + float(logp[parent[-1] if parent else 0, prefix[-1]])
        kept = sorted(expanded.items(), key=lambda row: row[1], reverse=True)[:beams]
    return kept


class SIDTrieChecks(unittest.TestCase):
    def setUp(self):
        self.trie = SIDTrie.from_semantic_ids(RAW, codebook_size=3)
        self.ids = torch.tensor([[1, 2], [2, 1]])

    def test_catalog_offsets_collision_dedup_and_roundtrip(self):
        self.assertEqual((self.trie.item_count, self.trie.unique_sid_count), (5, 4))
        self.assertEqual(self.trie.prefix_counts, [2, 4, 4])
        self.assertEqual(self.trie.allowed_tokens([]), [1, 2])
        self.assertEqual(self.trie.allowed_tokens([1, 4]), [7])
        self.assertEqual(self.trie.allowed_tokens([2, 4]), [9])
        self.assertEqual(self.trie.allowed_tokens([1, 6]), [])
        self.assertEqual(self.trie.allowed_tokens([0]), [])
        self.assertEqual(self.trie.allowed_tokens([1, 4, 7]), [])
        buffer = io.BytesIO()
        self.trie.save(buffer)
        buffer.seek(0)
        restored = SIDTrie.load(buffer)
        self.assertTrue(torch.equal(restored.children, self.trie.children))
        self.assertEqual(restored.allowed_tokens([2, 4]), [9])

    def test_beam_paths_and_original_scores_match_independent_search(self):
        for first_only in (True, False):
            with self.subTest(first_beam_only=first_only):
                model = ScriptedT5(first_only).eval()
                scores = []
                model._beam_trace_recorder = lambda **step: scores.append(step["selected_cumulative_scores"].clone())
                output = model.generate(self.ids, max_length=4, num_beams=2, num_return_sequences=2, sid_trie=self.trie)
                expected = reference_search(model, 2)
                for row in output.view(2, 2, 4).tolist():
                    self.assertEqual(row, [[0, *path] for path, _ in expected])
                torch.testing.assert_close(scores[-1], torch.tensor([[score for _, score in expected]]).expand(2, -1))
                self.assertTrue(all(torch.isfinite(step).all() for step in scores))

    def test_greedy_uses_catalog_and_excludes_high_scoring_illegal_tokens(self):
        model = ScriptedT5().eval()
        free = model.generate(self.ids, max_length=4)
        self.assertEqual(free[0, 1].item(), 3)  # Highest-scoring first token is illegal.
        constrained = model.generate(self.ids, max_length=4, sid_trie=self.trie)
        self.assertEqual(constrained.shape, (2, 4))
        for row in constrained[:, 1:].tolist():
            self.assertIn(tuple(row), PATHS)
        self.assertTrue(torch.equal(free, model.generate(self.ids, max_length=4, sid_trie=None)))
        # Fixed-length SID decoding must also ignore an overlapping generic EOS.
        model.config.eos_token_id = 1
        self.assertTrue(torch.equal(constrained, model.generate(self.ids, max_length=4, sid_trie=self.trie)))

    def test_checkpoint_keys_and_tiger_switch(self):
        config = dict(vocab_size=10, sem_id_dim=3, num_layers=1, num_decoder_layers=1,
                      d_model=8, d_ff=16, d_kv=4, num_heads=2)
        model = Tiger(config)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model.set_sid_trie(self.trie)
        self.assertEqual(set(before), set(model.state_dict()))
        model.load_state_dict(before, strict=True)
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        model.set_sid_trie(None)
        self.assertIsNone(model.sid_trie)

    def test_mismatched_catalog_and_insufficient_root_width_are_explicit(self):
        model = ScriptedT5().eval()
        with self.assertRaisesRegex(ValueError, "length/vocabulary"):
            model.generate(self.ids, max_length=3, sid_trie=self.trie)
        with self.assertRaisesRegex(ValueError, "first-level prefixes"):
            model.generate(self.ids, max_length=4, num_beams=3, sid_trie=self.trie)
        for raw in ([[0, -1, 0]], [[0, 3, 0]], [[0.1, 0, 0]]):
            with self.assertRaises(ValueError):
                SIDTrie.from_semantic_ids(raw, codebook_size=3)


if __name__ == "__main__":
    unittest.main()
