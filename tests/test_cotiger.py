"""Small CPU checks for search equivalence and sparse-prior boundary cases."""
import unittest

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.decoding import CPUCompactPrior, CSRGPUPrior, fast_search
from cotiger.ids import SID_SPACE, prefix_keys, sid_keys
from cotiger.prior import PrefixPrior
from cotiger.reference import reference_search


class CoTigerChecks(unittest.TestCase):
    def test_search_preserves_every_depth_and_raw_score(self):
        torch.manual_seed(19)
        model = Tiger(dict(num_layers=1, num_decoder_layers=1, d_model=16,
                           d_ff=32, d_kv=8, num_heads=2, vocab_size=17,
                           dropout_rate=0., sem_id_dim=3)).eval()
        ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
        mask = torch.ones_like(ids)
        observed = []
        with torch.inference_mode():
            expected = reference_search(model, ids, mask, (3, 3, 3))
            actual = fast_search(model, ids, mask, (3, 3, 3),
                                trace=lambda d, p, s: observed.append((p.clone(), s.clone())))
        self.assertTrue(torch.equal(actual, expected[0]))
        self.assertTrue(torch.equal(observed[-1][1], expected[2]))
        for depth, (paths, _) in enumerate(observed):
            self.assertTrue(torch.equal(paths[:, :, 1:], expected[-1][depth]))

    def test_sparse_priors_preserve_mask_recency_offsets_and_collisions(self):
        tokens = np.array([[1, 257, 513], [2, 258, 514]])
        keys, prefixes = sid_keys(tokens), prefix_keys(tokens)
        stats = dict(samples=np.array(8), code1_counts=np.zeros(769, int),
                     sid_target_keys=keys, sid_target_counts=np.array([5, 3]),
                     prefix_target_keys=prefixes, prefix_target_counts=np.array([5, 3]),
                     sid_pair_keys=np.array([keys[0]*SID_SPACE+keys[1], keys[1]*SID_SPACE+keys[0]]),
                     sid_pair_counts=np.array([2, 1]))
        stats['code1_counts'][[1, 2]] = [5, 3]
        reference = PrefixPrior(stats, np.concatenate((tokens, tokens[:1])), [0., .4, .4], True, 3)
        ids = torch.zeros(2, 16, dtype=torch.long)
        ids[0, -9:] = torch.tensor(np.r_[tokens[0], tokens[1], tokens[0]])
        ids[1, -3:] = torch.tensor(tokens[1])
        mask = (ids > 0).long()
        ids[:, :3] = 17  # Masked garbage must never enter the history.
        for implementation in (CPUCompactPrior, CSRGPUPrior):
            prior = implementation(reference, 'cpu')
            reference.prepare(ids, mask)
            prior.prepare(ids, mask)
            for depth, paths in ((2, [[0, 1], [0, 2], [0, 0], [0, 768]]),
                                 (3, [[0, 1, 257], [0, 2, 258], [0, 1, 258], [0, 768, 768]])):
                parents = torch.tensor([paths, paths])
                probabilities = torch.zeros(2, len(paths), 769)
                self.assertTrue(torch.equal(reference(depth, parents, probabilities),
                                            prior(depth, parents, probabilities)))
            self.assertEqual(float(prior.tables[1][0, 0]), 0.)
            self.assertGreater(float(prior.tables[1][0, 1]), -100.)  # Collision protected.
            self.assertLess(float(prior.tables[1][0, 2]), -900.)  # Seen singleton.


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
