"""
TIGER trainer (HF T5-based).

Pipeline:
- Flat offset-encoded codebook tokens (token = code + codebook_idx * codebook_size + 1).
- HF T5ForConditionalGeneration via cotiger.core.models.tiger.Tiger.
- Plain Adam (default) or cosine schedule with linear warmup.
- Early stopping on validation Recall@10.
"""

import os
import json
import math
import time
import numpy as np
import gin
import torch
import wandb

from cotiger.core.models.tiger import Tiger
from cotiger.core.modules.utils import parse_config, setup_logger
from cotiger.core.data.schemas import SeqData
from cotiger.core.trainers.trainer_utils import setup_wandb, set_seed
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from functools import partial
from tqdm import tqdm
from typing import List, Dict


def t5_collate(
    batch: List[SeqData],
    codebook_size: int = 256,
    sem_id_dim: int = 3,
    max_history_tokens: int = 60,  # max_seq_len * sem_id_dim (e.g. 20*3=60)
    dynamic_padding: bool = False,
    pad_to_multiple_of: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Convert SeqData to flat T5 format with offset encoding.

    Offset encoding: token = code + codebook_idx * codebook_size + 1
    (0 is reserved for padding)
    """
    B = len(batch)
    padded_length = max_history_tokens
    if dynamic_padding:
        padded_length = min(max(len(s.item_ids) for s in batch), max_history_tokens)
        # Truncate real history to max_history_tokens, then allow a few extra
        # padding tokens for aligned SDPA kernels (150 -> 152, still 50 items).
        padded_length = math.ceil(max(padded_length, 1) / pad_to_multiple_of) * pad_to_multiple_of
    input_ids = np.zeros((B, padded_length), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids)
    raw_targets = np.asarray([s.target_ids for s in batch], dtype=np.int64)

    for i, sample in enumerate(batch):
        raw_ids = sample.item_ids  # flat list: [c0, c1, c2, c0, c1, c2, ...]
        L = min(len(raw_ids), max_history_tokens)
        positions = np.arange(len(raw_ids) - L, len(raw_ids))
        offset_ids = np.asarray(raw_ids[-L:], dtype=np.int64) + (positions % sem_id_dim) * codebook_size + 1
        start = padded_length - L
        input_ids[i, start:] = offset_ids
        attention_mask[i, start:] = 1
    labels = raw_targets + np.arange(sem_id_dim) * codebook_size + 1

    return {
        'input_ids': torch.from_numpy(input_ids),
        'attention_mask': torch.from_numpy(attention_mask),
        'labels': torch.from_numpy(labels),
        'raw_targets': torch.from_numpy(raw_targets),
    }


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate position index of ground truth in predictions (vectorized)."""
    labels_expanded = labels.unsqueeze(1).expand_as(preds)  # (B, maxk, seq_len)
    match = (preds == labels_expanded).all(dim=-1)  # (B, maxk)
    first_match_mask = match.int().cumsum(dim=1) == 1
    return match & first_match_mask


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float().mean().item()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device).float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, 0.0)
    return dcg[:, :k].sum(dim=1).mean().item()


def autocast_context(device, amp, mixed_precision_type):
    dtype = torch.bfloat16 if mixed_precision_type == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=amp and device.type == "cuda")


@torch.inference_mode()
def evaluate(model, loader, device, beam_size, amp=False, mixed_precision_type="fp16", desc="Eval"):
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    n_samples = 0
    try:
        with autocast_context(device, amp, mixed_precision_type):
            for batch in tqdm(loader, desc=desc, mininterval=1.0):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                B_cur = input_ids.size(0)
                preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
                preds = preds[:, 1:].reshape(B_cur, beam_size, -1)
                pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
                discounts = 1.0 / torch.log2(torch.arange(2, beam_size + 2, device=device).float())
                for offset, k in enumerate((5, 10)):
                    totals[offset] += pos_index[:, :k].sum()
                    totals[offset + 2] += (pos_index[:, :k] * discounts[:k]).sum()
                n_samples += B_cur
        if not n_samples:
            raise ValueError("Evaluation dataset is empty.")
        # Average by users, not by batches; increasing infer_batch_size must not
        # change the weight of the final incomplete batch.
        values = (totals / n_samples).tolist()
        return dict(zip(('Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10'), values))
    finally:
        model.train(was_training)


@gin.configurable
def train(
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    dataset_folder="dataset/amazon",
    save_dir_root="out/tiger/",
    dataset=None,
    wandb_logging=True,
    wandb_project="tiger_training",
    wandb_log_interval=10,
    # Model config
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    feed_forward_proj='relu',
    # Data config
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    # Scheduler config
    lr_schedule="none",  # "none" or "cosine"
    num_warmup_steps=0,
    # Eval config
    beam_size=30,
    early_stop_patience=10,
    eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1,
    amp=False,
    mixed_precision_type="fp16",
    use_sdpa=False,
    first_beam_only=True,
    dynamic_padding=False,
    pad_to_multiple_of=8,
    num_workers=4,
    seed=42,
    max_train_samples=None,
    max_eval_samples=None,
):
    started_at = time.perf_counter()
    set_seed(seed)
    logger = setup_logger(save_dir_root, name="tiger")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs < 1 or min(eval_test_every_epoch, eval_valid_every_n_epochs, pad_to_multiple_of) < 1:
        raise ValueError("Epochs, evaluation intervals and padding multiple must be positive.")
    if beam_size < 10:
        raise ValueError("Recall@10 evaluation requires beam_size >= 10.")
    if mixed_precision_type not in {"fp16", "bf16", "no"}:
        raise ValueError("mixed_precision_type must be 'fp16', 'bf16' or 'no'.")
    amp = amp and device.type == "cuda" and mixed_precision_type != "no"
    if amp and mixed_precision_type == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU does not support native BF16; use mixed_precision_type='fp16'.")

    vocab_size = codebook_size * sem_id_dim + 1

    model_config = {
        'num_layers': num_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_model': d_model,
        'd_ff': d_ff,
        'num_heads': num_heads,
        'd_kv': d_kv,
        'dropout_rate': dropout_rate,
        'vocab_size': vocab_size,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': feed_forward_proj,
        'sem_id_dim': sem_id_dim,
        'use_sdpa': use_sdpa,
        'first_beam_only': first_beam_only,
    }

    if wandb_logging:
        setup_wandb(
            project=wandb_project,
            config={**model_config, 'lr': learning_rate, 'batch_size': batch_size,
                    'max_seq_len': max_seq_len, 'beam_size': beam_size},
        )

    # Load datasets
    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
    )
    if hasattr(train_dataset, "with_split"):
        valid_dataset = train_dataset.with_split("valid")
        test_dataset = train_dataset.with_split("test")
    else:
        valid_dataset = dataset(root=dataset_folder, train_test_split="valid", max_seq_len=max_seq_len,
                                pretrained_rqvae_path=pretrained_rqvae_path)
        test_dataset = dataset(root=dataset_folder, train_test_split="test", max_seq_len=max_seq_len,
                               pretrained_rqvae_path=pretrained_rqvae_path)
    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        valid_dataset = Subset(valid_dataset, range(min(max_eval_samples, len(valid_dataset))))
        test_dataset = Subset(test_dataset, range(min(max_eval_samples, len(test_dataset))))

    max_history_tokens = max_seq_len * sem_id_dim
    collate_fn = partial(t5_collate, codebook_size=codebook_size, sem_id_dim=sem_id_dim,
                         max_history_tokens=max_history_tokens, dynamic_padding=dynamic_padding,
                         pad_to_multiple_of=pad_to_multiple_of)
    worker_options = {"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda", drop_last=True,
        collate_fn=collate_fn, **worker_options,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn, **worker_options,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn, **worker_options,
    )

    logger.info(f"Train: {len(train_loader)} batches, Valid: {len(valid_loader)}, Test: {len(test_loader)}")
    if not len(train_loader):
        raise ValueError("Training dataset is smaller than batch_size with drop_last=True.")

    # Model
    model = Tiger(model_config).to(device)
    total_params, emb_params = model.num_parameters
    logger.info(f"Device: {device}, Params: {total_params:,} (emb: {emb_params:,})")
    logger.info(f"Precision: {mixed_precision_type if amp else 'fp32'}, SDPA: {use_sdpa}, dynamic padding: {dynamic_padding}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and mixed_precision_type == "fp16")
    optimizer_updates = 0

    def count_optimizer_update(optimizer, args, kwargs):
        nonlocal optimizer_updates
        optimizer_updates += 1

    optimizer.register_step_post_hook(count_optimizer_update)

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine" and epochs > 0:
        total_steps = len(train_loader) * epochs
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - num_warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        logger.info(f"Using cosine schedule, warmup={num_warmup_steps}, total_steps={total_steps}")
    else:
        logger.info("Using plain Adam, no scheduler")

    # Early stopping state
    best_valid_recall10 = -float("inf")
    best_epoch = -1
    early_stop_counter = 0
    best_valid_metrics = {}
    evaluation_history = []
    preparation_seconds = time.perf_counter() - started_at
    run_eval = partial(evaluate, model, device=device, beam_size=beam_size,
                       amp=amp, mixed_precision_type=mixed_precision_type)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = torch.zeros((), device=device)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=1.0):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, mixed_precision_type):
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale() if scheduler is not None else None
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
            total_loss += loss.detach()

        avg_loss = (total_loss / len(train_loader)).item()
        if not math.isfinite(avg_loss):
            raise FloatingPointError("TIGER loss is not finite; retry with train.amp=False.")
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        log_dict = {"epoch": epoch, "train/loss": avg_loss}

        # Valid evaluation (skip non-eval epochs)
        if (epoch + 1) % eval_valid_every_n_epochs == 0 or epoch == 0 or epoch + 1 == epochs:
            valid_metrics = run_eval(valid_loader, desc=f"Valid (Epoch {epoch})")
            evaluation_history.append({"split": "valid", "epoch": epoch + 1})
            logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")

            for k, v in valid_metrics.items():
                log_dict[f"eval/valid_{k}"] = v

            cur_recall10 = valid_metrics['Recall@10']

            # Check improvement
            if cur_recall10 > best_valid_recall10:
                best_valid_recall10 = cur_recall10
                best_epoch = epoch
                best_valid_metrics = valid_metrics
                early_stop_counter = 0

                # Save best model
                os.makedirs(save_dir_root, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir_root, "best_model.pt"))
                logger.info(f"New best Valid R@10={cur_recall10:.4f} at epoch {epoch}")

            else:
                early_stop_counter += 1
                logger.info(f"No improvement. Counter: {early_stop_counter}/{early_stop_patience}")

        # Independent schedule: testing does not trigger on every improvement,
        # and never determines checkpoint selection or early stopping.
        if (epoch + 1) % eval_test_every_epoch == 0:
            test_metrics = run_eval(test_loader, desc=f"Test (Epoch {epoch})")
            evaluation_history.append({"split": "test", "epoch": epoch + 1})
            logger.info(f"Epoch {epoch} - Test: {test_metrics}")
            for k, v in test_metrics.items():
                log_dict[f"eval/test_{k}"] = v

        if wandb_logging:
            wandb.log(log_dict)

        if early_stop_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    logger.info(f"Training done. Best Valid R@10={best_valid_recall10:.4f} at epoch {best_epoch}")
    best_path = os.path.join(save_dir_root, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_test = run_eval(test_loader, desc="Final test (best checkpoint)")
    logger.info(f"Final Test (best epoch {best_epoch}): {final_test}")
    result = {
        "model": "tiger", "seed": seed, "precision": mixed_precision_type if amp else "fp32",
        "best_epoch_zero_based": best_epoch, "best_valid": best_valid_metrics, "final_test": final_test,
        "epochs_completed": epoch + 1, "evaluation_history": evaluation_history,
        "preparation_seconds": preparation_seconds, "total_seconds": time.perf_counter() - started_at,
        "train_samples": len(train_dataset), "valid_samples": len(valid_dataset), "test_samples": len(test_dataset),
        "beam_size": beam_size, "infer_batch_size": infer_batch_size,
        "use_sdpa": use_sdpa, "dynamic_padding": dynamic_padding,
        "first_beam_only": first_beam_only,
        "grad_scaler_scale": scaler.get_scale(),
        "optimizer_updates": optimizer_updates,
    }
    with open(os.path.join(save_dir_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(save_dir_root, "resolved.gin"), "w", encoding="utf-8") as f:
        f.write(gin.operative_config_str())

    if wandb_logging:
        wandb.log({f"final_test/{k}": v for k, v in final_test.items()})
        wandb.finish()
    return result


if __name__ == "__main__":
    parse_config()
    train()
