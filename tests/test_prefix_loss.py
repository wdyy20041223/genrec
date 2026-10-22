"""Asset-free checks of CE + Prefix InfoNCE on a tiny T5, not an experiment."""
import unittest
from unittest.mock import patch

import torch

from cotiger.core.models.tiger import Tiger
from tiger_prefix_ft.losses import joint_loss, token_logprobs


class PrefixLossChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = Tiger(dict(num_layers=1, num_decoder_layers=1, d_model=16,
                                d_ff=32, d_kv=8, num_heads=2, vocab_size=769,
                                dropout_rate=0., sem_id_dim=3)).eval()
        ids = torch.tensor([[0, 0, 1, 257, 513], [0, 0, 2, 258, 514], [0, 0, 3, 259, 515]])
        negative = torch.tensor([[[30+i, 300+i, 600+i] for i in range(8)]]*3)
        negative[0,:,1:] = 0
        negative[1,:,2:] = 0
        self.batch = dict(input_ids=ids, attention_mask=ids.ne(0).long(),
                          ce_labels=torch.tensor([[12,402,704], [13,403,705], [14,404,706]]),
                          rank_depth=torch.tensor([1,2,3]), negative_tokens=negative)

    def test_shared_encoder_own_prefix_scores_and_ce_agree(self):
        t5, batch = self.model.model, self.batch
        with patch.object(t5.encoder, 'forward', wraps=t5.encoder.forward) as encoder:
            actual = joint_loss(self.model, batch)
            self.assertEqual(encoder.call_count, 1)
        expected_ce, _ = self.model(batch['input_ids'], batch['attention_mask'], batch['ce_labels'])
        torch.testing.assert_close(actual['loss_ce'], expected_ce)
        encoded = t5.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        for row, depth in enumerate([1,2,3]):
            tokens = batch['negative_tokens'][row:row+1, 0]
            valid = torch.arange(3)[None] < depth
            output = t5(encoder_outputs=(encoded[row:row+1],),
                        attention_mask=batch['attention_mask'][row:row+1],
                        decoder_input_ids=t5._shift_right(tokens),
                        decoder_attention_mask=valid.long(), use_cache=False)
            expected = (token_logprobs(output.logits, tokens)*valid).sum()
            torch.testing.assert_close(actual['negative_scores'][row,0], expected)

    def test_joint_loss_backpropagates_to_encoder_and_decoder(self):
        loss = joint_loss(self.model, self.batch)['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for prefix in ('model.encoder.block', 'model.decoder.block'):
            gradients = [p.grad for name,p in self.model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))


if __name__ == '__main__':
    unittest.main()
