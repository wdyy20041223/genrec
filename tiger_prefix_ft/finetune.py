"""Full-parameter TIGER fine-tuning on one fixed K=8 Parquet file."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from cotiger.core.models.tiger import Tiger
from .losses import joint_loss

from .paths import ROOT, PREFIX_DATA


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_weights(model, path):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


class FixedDataset(Dataset):
    def __init__(self, path, limit=None):
        if limit is None:
            self.rows = pq.read_table(path).to_pylist()
        else:
            self.rows = []
            for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit, 1024)):
                self.rows.extend(batch.to_pylist()[:limit - len(self.rows)])
                if len(self.rows) == limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def pad_histories(histories):
    width = math.ceil(max(map(len, histories)) / 8) * 8
    ids = np.zeros((len(histories), width), dtype=np.int64)
    masks = np.zeros_like(ids)
    for index, history in enumerate(histories):
        ids[index, -len(history):] = history
        masks[index, -len(history):] = 1
    return torch.from_numpy(ids), torch.from_numpy(masks)


def collate_fixed(rows):
    histories = [[int(t) for t, mask in zip(row["input_ids"], row["attention_mask"]) if mask] for row in rows]
    ids, masks = pad_histories(histories)
    labels = np.asarray([row["ce_labels"] for row in rows], dtype=np.int64)
    depths = np.asarray([row["rank_depth"] for row in rows], dtype=np.int64)
    negatives = np.zeros((len(rows), 8, 3), dtype=np.int64)
    for index, row in enumerate(rows):
        depth = int(depths[index])
        assert depth in (1, 2, 3) and len(row["ce_labels"]) == 3
        assert row["positive_tokens"] == row["ce_labels"][:depth]
        value = np.asarray(row["negative_tokens"], dtype=np.int64)
        assert value.shape == (8, depth)
        negatives[index, :, :depth] = value
    return {"input_ids": ids, "attention_mask": masks, "ce_labels": torch.from_numpy(labels),
            "rank_depth": torch.from_numpy(depths), "negative_tokens": torch.from_numpy(negatives)}


class EvaluationDataset(Dataset):
    def __init__(self, path, item_sids, max_history_items=50, limit=None):
        self.rows = pq.read_table(path, columns=["history_item_indices", "target_sid"]).to_pylist()
        if limit is not None:
            self.rows = self.rows[:limit]
        self.item_sids = item_sids
        self.max_history_items = max_history_items

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        indices = row["history_item_indices"][-self.max_history_items:]
        history = (self.item_sids[np.asarray(indices, dtype=np.int64)] + np.array([1, 257, 513])).reshape(-1).tolist()
        labels = (np.asarray(row["target_sid"], dtype=np.int64) + np.array([1, 257, 513])).tolist()
        return {"history": history, "labels": labels}


def collate_evaluation(rows):
    ids, masks = pad_histories([row["history"] for row in rows])
    return {"input_ids": ids, "attention_mask": masks,
            "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long)}


def make_loader(dataset, batch_size, collate, shuffle, workers, seed, cuda):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        collate_fn=collate, num_workers=workers, pin_memory=cuda,
        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))


def run(args):
    from cotiger.core.trainers.tiger_trainer import evaluate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    milestones = sorted(set(args.epochs))
    if not milestones or milestones[0] < 1 or args.batch_size < 1:
        raise ValueError("Epoch budgets and batch size must be positive")
    if args.temperature <= 0 or args.lambda_nce < 0:
        raise ValueError("Invalid InfoNCE temperature/lambda")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"Existing run: {output}")
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    amp = cuda and args.precision == "fp16"
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    assert config["sem_id_dim"] == 3 and config["vocab_size"] == 769
    source_hash = file_sha256(args.checkpoint)
    model = Tiger(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    assert all(parameter.requires_grad for parameter in model.parameters())
    train = FixedDataset(args.data, args.max_train_samples)
    if args.max_train_samples is None:
        assert len(train) == 131413
    item_sids = np.load(args.analysis_dir / "item_semantic_ids.npy", allow_pickle=False)
    assert item_sids.shape == (12101, 3)
    valid = EvaluationDataset(args.analysis_dir / "valid_samples.parquet", item_sids, args.max_history_items, args.max_eval_samples)
    test = EvaluationDataset(args.analysis_dir / "test_samples.parquet", item_sids, args.max_history_items, args.max_eval_samples)
    train_loader = make_loader(train, args.batch_size, collate_fixed, True, args.workers, args.seed, cuda)
    valid_loader = make_loader(valid, args.infer_batch_size, collate_evaluation, False, args.workers, args.seed + 1, cuda)
    test_loader = make_loader(test, args.infer_batch_size, collate_evaluation, False, args.workers, args.seed + 2, cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0, foreach=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer_updates = 0

    def updated(optimizer, positional, keywords):
        nonlocal optimizer_updates
        optimizer_updates += 1

    optimizer.register_step_post_hook(updated)
    report = {"state": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()), "data_sha256": file_sha256(args.data),
        "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": source_hash,
        "model_config": config, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "loss_mode": args.loss_mode, "lambda_nce": args.lambda_nce, "temperature": args.temperature,
        "learning_rate": args.learning_rate, "lr_schedule": "constant", "batch_size": args.batch_size,
        "seed": args.seed, "precision": args.precision, "beam_size": args.beam_size,
        "train_samples": len(train), "valid_samples": len(valid), "test_samples": len(test),
        "epoch_budgets": milestones, "budget_protocol": "single shared trajectory with checkpoints at each epoch budget",
        "smoke_steps": args.smoke_steps, "epochs": [], "budgets": {}}
    write_json(output / "result.json", report)
    started = time.perf_counter()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def run_eval(evaluation_model, loader, description):
        return evaluate(evaluation_model, loader, device, args.beam_size,
                        amp=amp, mixed_precision_type="fp16", desc=description)

    report["baseline"] = {"valid": run_eval(model, valid_loader, "Original valid"),
                          "test": run_eval(model, test_loader, "Original test")}
    print(json.dumps({"baseline": report["baseline"], "train_samples": len(train),
                      "steps_per_epoch": len(train_loader), "model_parameters": report["trainable_parameters"]}), flush=True)
    write_json(output / "result.json", report)
    best_recall = -float("inf")
    best_epoch = 0
    best_valid = {}
    best_path = output / "best_model.pt"
    max_epochs = 1 if args.smoke_steps else milestones[-1]
    try:
        for epoch in range(1, max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            seen = 0
            updates_before = optimizer_updates
            for step, batch in enumerate(train_loader, 1):
                batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
                with context:
                    losses = joint_loss(model, batch, args.lambda_nce, args.temperature, args.loss_mode)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                size = batch["ce_labels"].shape[0]
                totals += torch.stack([losses[key].detach() for key in ("loss", "loss_ce", "loss_nce")]) * size
                seen += size
                if step % 100 == 0:
                    print(json.dumps({"epoch": epoch, "step": step, "steps": len(train_loader),
                        "mean_losses": (totals / seen).tolist(), "optimizer_updates": optimizer_updates}), flush=True)
                del losses, batch
                if args.smoke_steps and step >= args.smoke_steps:
                    break
            assert optimizer_updates > updates_before, "No optimizer update occurred"
            train_seconds = time.perf_counter() - epoch_started
            mean_losses = (totals / seen).tolist()
            valid_metrics = run_eval(model, valid_loader, f"Valid epoch {epoch}")
            entry = {"epoch": epoch, "train_samples": seen, "loss": mean_losses[0],
                "loss_ce": mean_losses[1], "loss_nce": mean_losses[2],
                "train_seconds": train_seconds, "optimizer_updates": optimizer_updates - updates_before,
                "valid": valid_metrics}
            report["epochs"].append(entry)
            if valid_metrics["Recall@10"] > best_recall:
                best_recall, best_epoch, best_valid = valid_metrics["Recall@10"], epoch, valid_metrics
                save_weights(model, best_path)
            print(json.dumps(entry), flush=True)
            if epoch in milestones or args.smoke_steps:
                save_weights(model, output / f"checkpoint_epoch_{epoch}.pt")
                selected = output / f"best_up_to_epoch_{epoch}.pt"
                shutil.copy2(best_path, selected)
                # Evaluation-model initialization must not alter the training RNG.
                devices = [device.index or 0] if cuda else []
                with torch.random.fork_rng(devices=devices):
                    eval_model = Tiger(config).to(device)
                    eval_model.load_state_dict(torch.load(selected, map_location=device, weights_only=True), strict=True)
                    eval_model.eval()
                    selected_test = run_eval(eval_model, test_loader, f"Test best within {epoch} epochs")
                    del eval_model
                report["budgets"][str(epoch)] = {"epoch_budget": epoch, "selected_epoch": best_epoch,
                    "selected_valid": best_valid, "selected_test": selected_test,
                    "selected_checkpoint": str(selected), "last_epoch_checkpoint": str(output / f"checkpoint_epoch_{epoch}.pt")}
            report.update(optimizer_updates=optimizer_updates, best_epoch=best_epoch,
                          elapsed_seconds=time.perf_counter() - started)
            if cuda:
                report["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            write_json(output / "result.json", report)
        assert file_sha256(args.checkpoint) == source_hash
        report.update(state="complete", source_checkpoint_unchanged=True,
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "result.json", report)
        print(json.dumps({"state": "complete", "result": str(output / "result.json"), "budgets": report["budgets"]}), flush=True)
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "result.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREFIX_DATA / "train_k8_h4.parquet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--loss-mode", choices=["mixed", "ce_only"], default="mixed")
    parser.add_argument("--lambda-nce", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=15)
    parser.add_argument("--precision", choices=["fp16", "no"], default="fp16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-history-items", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--smoke-steps", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
