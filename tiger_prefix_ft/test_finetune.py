"""Tiny real-checkpoint checks; no local epoch training."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from cotiger.core.models.tiger import Tiger
from .finetune import FixedDataset, collate_fixed
from .losses import joint_loss, token_logprobs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out/cloud_reproduction/tiger_beauty_fast_20261012_194804"


class JointLossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        required = [BASE / 'analysis/model_config.json', BASE / 'run/tiger/best_model.pt',
                    ROOT / 'out/tiger_prefix_ft/beauty_20261013/data/train_k8_h4.parquet']
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest('Optional real-checkpoint tests require original Beauty weights and fixed H4 data')
        cls.config = json.loads((BASE / "analysis/model_config.json").read_text())
        cls.state = torch.load(BASE / "run/tiger/best_model.pt", map_location="cpu", weights_only=True)
        dataset = FixedDataset(ROOT / "out/tiger_prefix_ft/beauty_20261013/data/train_k8_h4.parquet", limit=64)
        rows = {}
        for row in dataset.rows:
            rows.setdefault(row["rank_depth"], row)
        assert set(rows) == {1, 2, 3}
        cls.batch = collate_fixed([rows[depth] for depth in (1, 2, 3)])

    def setUp(self):
        self.model = Tiger(self.config)
        self.model.load_state_dict(self.state, strict=True)
        self.model.eval()

    def test_ce_matches_original_teacher_forcing_and_positive_prefix(self):
        with torch.no_grad():
            value = joint_loss(self.model, self.batch)
            original = self.model.model(input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"], labels=self.batch["ce_labels"])
            self.assertTrue(torch.allclose(value["loss_ce"], original.loss, atol=1e-6))
            scores = token_logprobs(original.logits, self.batch["ce_labels"])
            expected = torch.stack([scores[i, :int(depth)].sum() for i, depth in enumerate(self.batch["rank_depth"])])
            self.assertTrue(torch.allclose(value["positive_scores"], expected, atol=1e-5))
            self.assertTrue(torch.allclose(value["loss"], value["loss_ce"] + .1 * value["loss_nce"]))

    def test_negative_scores_match_each_own_prefix(self):
        with torch.no_grad():
            value = joint_loss(self.model, self.batch)
            expected = torch.empty_like(value["negative_scores"])
            for user, depth in enumerate(self.batch["rank_depth"]):
                for negative in range(8):
                    tokens = self.batch["negative_tokens"][user, negative, :int(depth)][None, :]
                    out = self.model.model(input_ids=self.batch["input_ids"][user:user + 1],
                        attention_mask=self.batch["attention_mask"][user:user + 1], labels=tokens)
                    expected[user, negative] = token_logprobs(out.logits, tokens).sum()
            self.assertTrue(torch.allclose(value["negative_scores"], expected, atol=2e-5))
            # A real Depth-2 Easy prefix starts differently from the gold parent.
            tokens = self.batch["negative_tokens"][1, 0, :2][None, :]
            wrong_input = self.model.model._shift_right(tokens)
            wrong_input[0, 1] = self.batch["ce_labels"][1, 0]
            wrong = self.model.model(input_ids=self.batch["input_ids"][1:2],
                attention_mask=self.batch["attention_mask"][1:2], decoder_input_ids=wrong_input)
            own = self.model.model(input_ids=self.batch["input_ids"][1:2],
                attention_mask=self.batch["attention_mask"][1:2], labels=tokens)
            self.assertGreater(float((wrong.logits[:, 1] - own.logits[:, 1]).abs().max()), 1e-4)

    def test_encoder_once_and_negative_decoder_vectorized(self):
        with patch.object(self.model.model.encoder, "forward", wraps=self.model.model.encoder.forward) as encode:
            with patch.object(self.model.model.decoder, "forward", wraps=self.model.model.decoder.forward) as decode:
                with torch.no_grad():
                    joint_loss(self.model, self.batch)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(decode.call_count, 2)
                self.assertEqual(decode.call_args_list[1].kwargs["input_ids"].shape, (24, 3))

    def test_future_padding_does_not_change_prefix_scores(self):
        changed = {key: tensor.clone() for key, tensor in self.batch.items()}
        for user, depth in enumerate(changed["rank_depth"]):
            changed["negative_tokens"][user, :, int(depth):] = 600
            changed["ce_labels"][user, int(depth):] = 650
        with torch.no_grad():
            original = joint_loss(self.model, self.batch)
            modified = joint_loss(self.model, changed)
            self.assertTrue(torch.allclose(original["positive_scores"], modified["positive_scores"], atol=1e-6))
            self.assertTrue(torch.allclose(original["negative_scores"], modified["negative_scores"], atol=1e-6))

    def test_nce_encoder_gradient_and_joint_parameter_update(self):
        value = joint_loss(self.model, self.batch)
        value["loss_nce"].backward()
        parameter = self.model.model.encoder.block[0].layer[0].SelfAttention.q.weight
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.model.zero_grad(set_to_none=True)
        before = parameter.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        value = joint_loss(self.model, self.batch)
        self.assertTrue(torch.isfinite(value["loss"]))
        value["loss"].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_ce_only_skips_negative_decoder(self):
        with patch.object(self.model.model.decoder, "forward", wraps=self.model.model.decoder.forward) as decode:
            with torch.no_grad():
                value = joint_loss(self.model, self.batch, loss_mode="ce_only")
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(float(value["loss_nce"]), 0)
            self.assertIsNone(value["negative_scores"])
            self.assertEqual(float(value["loss"]), float(value["loss_ce"]))


if __name__ == "__main__":
    unittest.main()
