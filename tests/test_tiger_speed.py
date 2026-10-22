"""Small correctness checks for TIGER execution optimizations (no dataset downloads)."""
import unittest
import tempfile
import logging
from unittest.mock import patch

import torch

from cotiger.core.data.amazon import AmazonSeqDataset
from cotiger.core.data.schemas import SeqData
from cotiger.core.models.rqvae import RqVae, _sinkhorn_knopp, QuantizeForwardMode
from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import t5_collate, evaluate, train


class TigerSpeedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = dict(num_layers=1, num_decoder_layers=1, d_model=32,
                           d_ff=64, num_heads=2, d_kv=16, dropout_rate=0.0,
                           vocab_size=769, sem_id_dim=3)
        self.samples = [SeqData(0, [2, 3, 4, 5, 6, 7], [8, 9, 10]),
                        SeqData(1, [12, 13, 14], [15, 16, 17])]

    def test_vectorized_collate_preserves_offset_and_left_truncation(self):
        samples = [SeqData(0, list(range(12)), [1, 2, 3]), self.samples[1]]
        batch = t5_collate(samples, max_history_tokens=7)
        expected = torch.tensor([[5 + 513, 6 + 1, 7 + 257, 8 + 513,
                                  9 + 1, 10 + 257, 11 + 513],
                                 [0, 0, 0, 0, 13, 270, 527]])
        torch.testing.assert_close(batch['input_ids'], expected)
        torch.testing.assert_close(batch['labels'], torch.tensor([[2, 259, 516], [16, 273, 530]]))

    def test_sdpa_logits_loss_gradients_and_cached_beams_match_eager(self):
        eager = Tiger(self.config).eval()
        sdpa = Tiger({**self.config, 'use_sdpa': True}).eval()
        sdpa.load_state_dict(eager.state_dict())
        batch = t5_collate(self.samples, max_history_tokens=12)
        inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'labels')}
        loss_a, logits_a = eager(**inputs)
        loss_b, logits_b = sdpa(**inputs)
        torch.testing.assert_close(logits_a, logits_b, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(loss_a, loss_b, rtol=2e-6, atol=2e-6)
        loss_a.backward()
        loss_b.backward()
        for a, b in zip(eager.parameters(), sdpa.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
        generated_a = eager.generate(batch['input_ids'], batch['attention_mask'], num_beams=15)
        generated_b = sdpa.generate(batch['input_ids'], batch['attention_mask'], num_beams=15)
        torch.testing.assert_close(generated_a, generated_b, rtol=0, atol=0)

    def test_dynamic_padding_preserves_model_logits(self):
        model = Tiger(self.config).eval()
        fixed = t5_collate(self.samples, max_history_tokens=150)
        dynamic = t5_collate(self.samples, max_history_tokens=150,
                             dynamic_padding=True, pad_to_multiple_of=8)
        self.assertEqual(dynamic['input_ids'].shape[1], 8)
        with torch.no_grad():
            outputs = [model(b['input_ids'], b['attention_mask'], b['labels']) for b in (fixed, dynamic)]
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=3e-6, atol=3e-6)

    def test_aligned_padding_keeps_history_limit(self):
        sample = SeqData(0, [1, 2, 3] * 55, [4, 5, 6])
        batch = t5_collate([sample], max_history_tokens=150,
                           dynamic_padding=True, pad_to_multiple_of=8)
        self.assertEqual(batch['input_ids'].shape[1], 152)
        self.assertEqual(batch['attention_mask'].sum().item(), 150)

    def test_first_beam_reuse_matches_reference_decode(self):
        reference = Tiger({**self.config, 'first_beam_only': False}).eval()
        optimized = Tiger({**self.config, 'first_beam_only': True}).eval()
        optimized.load_state_dict(reference.state_dict())
        batch = t5_collate(self.samples, max_history_tokens=12)
        old = reference.generate(batch['input_ids'], batch['attention_mask'], num_beams=15)
        new = optimized.generate(batch['input_ids'], batch['attention_mask'], num_beams=15)
        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_split_reuse_preserves_targets_and_original_samples(self):
        data = AmazonSeqDataset.__new__(AmazonSeqDataset)
        data.sequences = [list(range(5)), list(range(5, 12))]
        data.user_ids = ['first', 'second']
        data.sem_ids_list = [[i, i, i] for i in range(12)]
        data.item_embeddings = torch.zeros(12, 8)
        data._max_seq_len = 50
        data.train_test_split = 'train'
        data._generate_samples()
        original_samples = list(data.samples)
        valid = data.with_split('valid')
        test = data.with_split('test')
        self.assertIs(data.sem_ids_list, valid.sem_ids_list)
        self.assertIs(data.sequences, test.sequences)
        self.assertEqual(data.samples, original_samples)
        self.assertEqual([s['target'] for s in valid.samples], [3, 10])
        self.assertEqual([s['target'] for s in test.samples], [4, 11])

    def test_unique_statistic_matches_quadratic_formula(self):
        model = RqVae(8, 4, [8], 4, codebook_kmeans_init=False, n_layers=3,
                      n_cat_features=0).eval()
        data = torch.randn(12, 8)
        with torch.no_grad():
            ids = model.get_semantic_ids(data).sem_ids
            old = (~torch.triu((ids[:, None] == ids[None, :]).all(-1), diagonal=1)).all(1).sum() / len(ids)
            output = model(data, 0.2, compute_stats=True)
            no_stats = model(data, 0.2, compute_stats=False)
        torch.testing.assert_close(output.p_unique_ids, old)
        torch.testing.assert_close(output.loss, no_stats.loss, rtol=0, atol=0)

    def test_sinkhorn_accepts_fp64_cost_with_fp32_marginals(self):
        cost = torch.rand(8, 4, dtype=torch.float64)
        result = _sinkhorn_knopp(cost, torch.full((8,), 1/8), torch.full((4,), 1/4), eps=0.1, max_iter=100)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result.sum(0), torch.full((4,), 1/4, dtype=torch.float64), atol=1e-5, rtol=1e-5)

    def test_small_kmeans_initialization_uses_quantization_path(self):
        model = RqVae(8, 4, [8], 4, codebook_kmeans_init=True,
                      codebook_mode=QuantizeForwardMode.STE,
                      codebook_last_layer_mode=QuantizeForwardMode.SINKHORN,
                      n_layers=3, n_cat_features=0).train()
        with torch.no_grad():
            output = model.get_semantic_ids(torch.randn(32, 8), 0.2)
        self.assertTrue(all(layer.kmeans_initted for layer in model.layers))
        self.assertTrue(torch.isfinite(output.quantize_loss).all())
        self.assertEqual(output.sem_ids.shape, (32, 3))
        self.assertTrue(((output.sem_ids >= 0) & (output.sem_ids < 4)).all())

    def test_metrics_are_invariant_to_incomplete_eval_batch(self):
        class FakeModel(torch.nn.Module):
            def generate(self, input_ids, attention_mask=None, num_beams=10):
                result = torch.zeros(len(input_ids), num_beams, 4, dtype=torch.long)
                for row, user in enumerate(input_ids[:, 0].tolist()):
                    if user < 3:
                        result[row, 0, 1:] = torch.tensor([user, 257, 513])
                return result.reshape(-1, 4)
        def batch(users):
            return {'input_ids': torch.tensor(users)[:, None],
                    'attention_mask': torch.ones(len(users), 1, dtype=torch.long),
                    'labels': torch.tensor([[u, 257, 513] for u in users])}
        model = FakeModel()
        two_batches = evaluate(model, [batch([1, 2]), batch([3])], torch.device('cpu'), 10)
        one_batch = evaluate(model, [batch([1, 2, 3])], torch.device('cpu'), 10)
        self.assertEqual(two_batches, one_batch)
        self.assertAlmostEqual(two_batches['Recall@10'], 2/3)
        self.assertAlmostEqual(two_batches['NDCG@10'], 2/3)
        self.assertTrue(model.training)

    def test_periodic_test_runs_independently_and_final_test_uses_best(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, train_test_split='train', **kwargs):
                self.train_test_split = train_test_split

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return SeqData(index, [0, 1, 2] * 2, [1, 2, 3])

        evaluated_splits = []
        evaluated_weights = []

        def fake_evaluate(model, loader, **kwargs):
            evaluated_splits.append(loader.dataset.train_test_split)
            evaluated_weights.append(next(model.parameters()).detach().clone())
            return {key: 0.0 for key in ('Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10')}

        # CPU-only: six updates to a tiny network, with constant validation
        # scores so epoch 1 remains best while epoch 5 still triggers testing.
        with tempfile.TemporaryDirectory() as output:
            try:
                with patch('cotiger.core.trainers.tiger_trainer.evaluate', side_effect=fake_evaluate), \
                     patch('cotiger.core.trainers.tiger_trainer.torch.cuda.is_available', return_value=False):
                    result = train(dataset=TinyDataset, save_dir_root=output, epochs=6,
                                   batch_size=2, infer_batch_size=2, wandb_logging=False,
                                   num_workers=0, num_layers=1, num_decoder_layers=1,
                                   d_model=8, d_ff=16, num_heads=2, d_kv=4,
                                   codebook_size=4, max_seq_len=2, dropout_rate=0,
                                   beam_size=10, eval_valid_every_n_epochs=3,
                                   eval_test_every_epoch=5)
            finally:
                logger = logging.getLogger('tiger')
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
        self.assertEqual([entry['epoch'] for entry in result['evaluation_history'] if entry['split'] == 'valid'], [1, 3, 6])
        self.assertEqual([entry['epoch'] for entry in result['evaluation_history'] if entry['split'] == 'test'], [5])
        self.assertEqual(result['best_epoch_zero_based'], 0)
        self.assertEqual(result['optimizer_updates'], 6)
        self.assertEqual(evaluated_splits, ['valid', 'valid', 'test', 'valid', 'test'])
        torch.testing.assert_close(evaluated_weights[0], evaluated_weights[-1], rtol=0, atol=0)
        self.assertFalse(torch.equal(evaluated_weights[-2], evaluated_weights[-1]))


if __name__ == '__main__':
    unittest.main()
