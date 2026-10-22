"""Independent correctness checks before real-data acceleration benchmarks."""
import unittest
import numpy as np
import torch

from .inference_fast import CPUCompactPrior, CSRGPUPrior, fast_search
from .inference_lab import alternative_search
from .inference_priors import PrefixPrior
from .inference_rerank import PREFIX_SPACE, SID_SPACE, prefix_keys, sid_keys
from . import test_inference_lab


class FastChecks(unittest.TestCase):
    def test_decoder_all_paths_raw_scores_and_parents_match(self):
        fixture = test_inference_lab.InferenceChecks()
        fixture.setUp()
        ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
        mask = torch.ones_like(ids)
        observed = []
        with torch.inference_mode():
            expected = alternative_search(fixture.tiger, ids, mask, (3, 3, 3))
            actual = fast_search(fixture.tiger, ids, mask, (3, 3, 3), trace=lambda d, p, s: observed.append((p.clone(), s.clone())))
        self.assertTrue(torch.equal(actual, expected[0]))
        self.assertTrue(torch.equal(observed[-1][1], expected[2]))
        for step, values in enumerate(observed):
            self.assertTrue(torch.equal(values[0][:, :, 1:], expected[-1][step]))

    def test_compact_priors_keep_offsets_unseen_zero_collision_and_recency(self):
        tokens = np.array([[1, 257, 513], [2, 258, 514]])
        keys, prefixes = sid_keys(tokens), prefix_keys(tokens)
        stats = dict(samples=np.array(8), code1_counts=np.zeros(769, int),
                     sid_target_keys=keys, sid_target_counts=np.array([5, 3]),
                     prefix_target_keys=prefixes, prefix_target_counts=np.array([5, 3]),
                     sid_pair_keys=np.array([keys[0]*SID_SPACE+keys[1], keys[1]*SID_SPACE+keys[0]]),
                     sid_pair_counts=np.array([2, 1]))
        stats['code1_counts'][[1, 2]] = [5, 3]
        catalog = np.concatenate((tokens, tokens[:1]))
        old = PrefixPrior(stats, catalog, [0., .4, .4], True, 3)
        # Non-multiple-of-three left padding; a repeated SID keeps its original age.
        ids = torch.zeros(2, 16, dtype=torch.long)
        ids[0, -9:] = torch.tensor(np.r_[tokens[0], tokens[1], tokens[0]])
        ids[1, -3:] = torch.tensor(tokens[1])
        mask = (ids > 0).long()
        ids[:, :3] = 17  # Masked garbage must never become history.
        for cls in [CPUCompactPrior, CSRGPUPrior]:
            new = cls(old, 'cpu')
            old.prepare(ids, mask)
            new.prepare(ids, mask)
            for depth, parent in [(2, [[0, 1], [0, 2], [0, 0], [0, 768]]),
                                  (3, [[0, 1, 257], [0, 2, 258], [0, 1, 258], [0, 768, 768]])]:
                parents = torch.tensor([parent, parent])
                probabilities = torch.zeros(2, len(parent), 769)
                a, b = old(depth, parents, probabilities), new(depth, parents, probabilities)
                self.assertTrue(torch.equal(a, b), (cls.__name__, depth, float((a-b).abs().max())))
            self.assertEqual(float(new.tables[1][0, 0]), 0.)
            self.assertGreater(float(new.tables[1][0, 1]), -100.)  # collided group
            self.assertLess(float(new.tables[1][0, 2]), -900.)   # seen singleton


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
