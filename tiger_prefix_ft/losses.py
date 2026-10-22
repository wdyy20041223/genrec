"""Teacher-forced CE and Mixed Prefix InfoNCE for the original TIGER T5."""
import torch
import torch.nn.functional as F


def token_logprobs(logits, tokens):
    return F.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def joint_loss(tiger, batch, lambda_nce=0.1, temperature=1.0, loss_mode="mixed"):
    if temperature <= 0 or lambda_nce < 0:
        raise ValueError("temperature must be positive and lambda_nce nonnegative")
    if loss_mode not in ("mixed", "ce_only"):
        raise ValueError("loss_mode must be mixed or ce_only")
    t5 = tiger.model
    labels = batch["ce_labels"]
    batch_size, sid_length = labels.shape
    encoder_outputs = t5.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ce = t5(encoder_outputs=encoder_outputs, attention_mask=batch["attention_mask"], labels=labels, use_cache=False)
    valid = torch.arange(sid_length, device=labels.device)[None, :] < batch["rank_depth"][:, None]
    positive = (token_logprobs(ce.logits, labels) * valid).sum(-1)
    if loss_mode == "ce_only":
        return {"loss": ce.loss, "loss_ce": ce.loss, "loss_nce": ce.loss.new_zeros(()),
                "positive_scores": positive, "negative_scores": None}

    negatives = batch["negative_tokens"]
    count = negatives.shape[1]
    assert count == 8 and negatives.shape == (batch_size, count, sid_length)
    flat_tokens = negatives.reshape(batch_size * count, sid_length)
    negative_valid = valid.repeat_interleave(count, dim=0)
    # repeat_interleave preserves the autograd route back to the single encoder.
    expanded_hidden = encoder_outputs[0].repeat_interleave(count, dim=0)
    expanded_mask = batch["attention_mask"].repeat_interleave(count, dim=0)
    negative_output = t5(
        encoder_outputs=(expanded_hidden,), attention_mask=expanded_mask,
        decoder_input_ids=t5._shift_right(flat_tokens),
        decoder_attention_mask=negative_valid.long(), use_cache=False,
    )
    negative = (token_logprobs(negative_output.logits, flat_tokens) * negative_valid).sum(-1).reshape(batch_size, count)
    scores = torch.cat((positive[:, None], negative), dim=1) / temperature
    nce = F.cross_entropy(scores, torch.zeros(batch_size, dtype=torch.long, device=scores.device))
    return {"loss": ce.loss + lambda_nce * nce, "loss_ce": ce.loss, "loss_nce": nce,
            "positive_scores": positive, "negative_scores": negative}
