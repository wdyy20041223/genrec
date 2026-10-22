"""Export an existing TIGER run for offline analysis, without training."""

import argparse
from collections import defaultdict
from functools import partial
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import gin
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotiger.core.data.amazon import AmazonSeqDataset, parse_gzip_json
from cotiger.core.models.rqvae import RqVae
from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers import tiger_trainer
from cotiger.core.trainers.trainer_utils import set_seed


def setting(name):
    try:
        return gin.query_parameter("train." + name)
    except ValueError:
        return inspect.signature(tiger_trainer.train).parameters[name].default


def rq_model():
    return RqVae(
        input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128, 64],
        codebook_size=256, codebook_kmeans_init=False,
        codebook_normalize=False, codebook_sim_vq=False,
        n_layers=3, n_cat_features=0, commitment_weight=0.25,
    ).eval()


def export_samples(dataset, destination, asins, sids, group_sizes):
    if dataset.train_test_split == "train":
        user_indices = np.repeat(
            np.arange(len(dataset.sequences)),
            [len(seq) - 3 for seq in dataset.sequences],
        )
    else:
        user_indices = np.arange(len(dataset.sequences))
    assert len(user_indices) == len(dataset.samples)
    rows = []
    for index, (sample, user_index) in enumerate(zip(dataset.samples, user_indices)):
        target = sample["target"]
        history = dataset._truncate_history(sample["history"])
        assert 0 <= target < len(sids)
        assert all(0 <= item < len(sids) for item in history)
        row = {
            "sample_index": index, "user_index": int(user_index),
            "reviewer_id": dataset.user_ids[user_index],
            "history_item_indices": history,
            "history_length": len(history),
            "full_history_length": len(sample["history"]),
            "target_item_index": target, "target_asin": asins[target],
            "target_sid": sids[target].tolist(),
            "target_sid_group_size": int(group_sizes[target]),
        }
        rows.append(row)
    pd.DataFrame(rows).to_parquet(destination, index=False)


@torch.inference_mode()
def export_predictions(model, dataset, destination, sids, code_to_items, expected):
    beam = int(setting("beam_size"))
    size = int(setting("codebook_size"))
    dim = int(setting("sem_id_dim"))
    collate = partial(
        tiger_trainer.t5_collate, codebook_size=size, sem_id_dim=dim,
        max_history_tokens=int(setting("max_seq_len")) * dim,
        dynamic_padding=bool(setting("dynamic_padding")),
        pad_to_multiple_of=int(setting("pad_to_multiple_of")),
    )
    loader = DataLoader(
        dataset, batch_size=int(setting("infer_batch_size")), shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate,
    )
    totals = torch.zeros(4, device="cuda", dtype=torch.float64)
    predictions, matches = [], []
    device = torch.device("cuda")
    started = time.perf_counter()
    model.eval()
    with tiger_trainer.autocast_context(
        device, bool(setting("amp")), setting("mixed_precision_type")
    ):
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pred = model.generate(inputs, mask, num_beams=beam)
            pred = pred[:, 1:].reshape(len(inputs), beam, dim)
            pos = tiger_trainer.calculate_pos_index(pred, labels, maxk=beam)
            discounts = 1.0 / torch.log2(torch.arange(2, beam + 2, device=device).float())
            for offset, k in enumerate((5, 10)):
                totals[offset] += pos[:, :k].sum()
                totals[offset + 2] += (pos[:, :k] * discounts[:k]).sum()
            predictions.append(pred.cpu().numpy())
            matches.append(pos.cpu().numpy())
    tokens = np.concatenate(predictions).astype(np.int16)
    matched = np.concatenate(matches)
    raw = tokens.astype(np.int32) - (np.arange(dim) * size + 1)
    metrics = dict(zip(
        ("Recall@5", "Recall@10", "NDCG@5", "NDCG@10"),
        (totals / len(dataset)).tolist(),
    ))
    deltas = {key: metrics[key] - expected[key] for key in metrics}
    if max(abs(value) for value in deltas.values()) > 2e-7:
        raise RuntimeError(f"Exported predictions do not reproduce saved metrics: {deltas}")
    split = dataset.train_test_split
    np.save(destination / f"{split}_prediction_tokens.npy", tokens)
    np.save(destination / f"{split}_prediction_raw_sids.npy", raw.astype(np.int16))
    target_indices = np.array([sample["target"] for sample in dataset.samples])
    np.save(destination / f"{split}_target_sids.npy", sids[target_indices])
    rows, users = [], []
    for index in range(len(dataset)):
        target = int(target_indices[index])
        valid_count = 0
        position_count = 0
        hit_ranks = np.flatnonzero(matched[index]) + 1
        first_hit = int(hit_ranks[0]) if len(hit_ranks) else 0
        for rank in range(beam):
            key = tuple(int(v) for v in raw[index, rank])
            item_indices = code_to_items.get(key, [])
            position_valid = all(0 <= v < size for v in key)
            valid_count += bool(item_indices)
            position_count += position_valid
            rows.append({
                "sample_index": index, "user_index": index, "rank": rank + 1,
                "token_0": int(tokens[index, rank, 0]),
                "token_1": int(tokens[index, rank, 1]),
                "token_2": int(tokens[index, rank, 2]),
                "sid_0": key[0], "sid_1": key[1], "sid_2": key[2],
                "position_valid": position_valid,
                "catalog_item_count": len(item_indices),
                "catalog_item_indices": item_indices,
                "target_sid_match": bool(matched[index, rank]),
            })
        users.append({
            "sample_index": index, "user_index": index,
            "reviewer_id": dataset.user_ids[index],
            "target_item_index": target,
            "target_sid_group_size": len(code_to_items[tuple(sids[target])]),
            "first_sid_hit_rank": first_hit,
            "recall_5": int(0 < first_hit <= 5),
            "recall_10": int(0 < first_hit <= 10),
            "ndcg_5": float(np.float32(1 / np.log2(first_hit + 1))) if 0 < first_hit <= 5 else 0.0,
            "ndcg_10": float(np.float32(1 / np.log2(first_hit + 1))) if 0 < first_hit <= 10 else 0.0,
            "catalog_valid_predictions": valid_count,
            "position_valid_predictions": position_count,
        })
    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_parquet(destination / f"{split}_predictions.parquet", index=False)
    user_frame = pd.DataFrame(users)
    user_frame.to_parquet(destination / f"{split}_user_metrics.parquet", index=False)
    user_frame.to_csv(destination / f"{split}_user_metrics.csv", index=False, encoding="utf-8")
    summary = {
        "samples": len(dataset), "candidates_per_sample": beam,
        "metrics": metrics, "difference_from_saved_metrics": deltas,
        "position_valid_fraction": float(prediction_frame.position_valid.mean()),
        "catalog_valid_fraction": float((prediction_frame.catalog_item_count > 0).mean()),
        "targets_in_collision_groups": int((user_frame.target_sid_group_size > 1).sum()),
        "seconds": time.perf_counter() - started,
    }
    print(split.upper() + "_PREDICTIONS " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    run = args.run_dir
    result = json.loads((run / "tiger/results.json").read_text())
    assert result["epochs_completed"] == 100
    set_seed(result["seed"])
    # Older exports used GenRec's module namespace; model state_dict keys are unchanged.
    config_text = (run / "tiger/resolved.gin").read_text(encoding="utf-8")
    gin.parse_config(config_text.replace("genrec.", "cotiger.core."))
    dataset = AmazonSeqDataset(
        root=str(args.dataset_root), split="beauty", train_test_split="train",
        max_seq_len=int(setting("max_seq_len")),
        pretrained_rqvae_path=str(run / "rqvae/checkpoint_epoch_1999.pt"),
    )
    sids = np.asarray(dataset.sem_ids_list, dtype=np.int16)
    assert sids.shape == (12101, 3)
    assert not dataset.add_disambiguation
    all_asins, asins = [], []
    seen_all, seen_valid = set(), set()
    for review in parse_gzip_json(args.dataset_root / "raw/beauty/reviews_Beauty_5.json.gz"):
        asin = review.get("asin")
        if asin and asin not in seen_all:
            seen_all.add(asin)
            all_asins.append(asin)
        if asin and review.get("reviewerID") and asin not in seen_valid:
            seen_valid.add(asin)
            asins.append(asin)
    embedding_ids = pd.read_parquet(
        args.dataset_root / "processed/beauty/item_emb_sentence-t5-xl.parquet",
        columns=["ItemID"],
    ).ItemID.to_numpy()
    assert len(asins) == len(sids)
    code_to_items = defaultdict(list)
    for index, sid in enumerate(sids):
        code_to_items[tuple(sid)].append(index)
    sizes = np.array([len(code_to_items[tuple(sid)]) for sid in sids])
    items = pd.DataFrame({
        "item_index": np.arange(len(sids)), "asin": asins,
        "embedding_row": np.arange(len(sids)), "embedding_item_id": embedding_ids,
        "embedding_asin": [all_asins[int(item_id) - 1] for item_id in embedding_ids],
        "sid_0": sids[:, 0], "sid_1": sids[:, 1], "sid_2": sids[:, 2],
        "sid": ["-".join(map(str, sid)) for sid in sids],
        "collision_group_size": sizes,
    })
    items.to_parquet(destination / "item_sid.parquet", index=False)
    items.to_csv(destination / "item_sid.csv", index=False, encoding="utf-8")
    np.save(destination / "item_semantic_ids.npy", sids)
    collisions = [{
        "sid_0": int(key[0]), "sid_1": int(key[1]), "sid_2": int(key[2]),
        "group_size": len(indices), "item_indices": indices,
        "asins": [asins[index] for index in indices],
    } for key, indices in code_to_items.items() if len(indices) > 1]
    pd.DataFrame(collisions, columns=[
        "sid_0", "sid_1", "sid_2", "group_size", "item_indices", "asins",
    ]).to_parquet(destination / "sid_collisions.parquet", index=False)
    pd.DataFrame({
        "user_index": np.arange(len(dataset.sequences)), "reviewer_id": dataset.user_ids,
        "item_indices": dataset.sequences,
        "sequence_length": [len(seq) for seq in dataset.sequences],
    }).to_parquet(destination / "user_sequences.parquet", index=False)
    splits = {"train": dataset, "valid": dataset.with_split("valid"), "test": dataset.with_split("test")}
    for name, split in splits.items():
        assert len(split) == result[name + "_samples"]
        export_samples(split, destination / f"{name}_samples.parquet", asins, sids, sizes)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    summary = {
        "state": "complete", "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "sid_method": "CPU FP32 eval argmin, identical to AmazonSeqDataset initialization",
        "item_count": len(sids), "sid_dimension": 3,
        "disambiguation_enabled": False,
        "unique_sids": len(code_to_items), "collision_groups": len(collisions),
        "items_in_collision_groups": int((sizes > 1).sum()),
        "largest_collision_group": int(sizes.max()),
        "asin_matches_embedding_asin": bool((items.asin == items.embedding_asin).all()),
        "split_sizes": {name: len(split) for name, split in splits.items()},
    }
    print("SID_EXPORT " + json.dumps(summary), flush=True)
    rq = rq_model()
    checkpoint_epochs, checkpoint_ids = [], []
    for checkpoint in sorted(run.glob("rqvae/checkpoint_epoch_*.pt"), key=lambda path: int(path.stem.split("_")[-1])):
        rq.load_pretrained(str(checkpoint))
        with torch.inference_mode():
            codes = rq.get_semantic_ids(dataset.item_embeddings).sem_ids.cpu().numpy().astype(np.int16)
        checkpoint_epochs.append(int(checkpoint.stem.split("_")[-1]) + 1)
        checkpoint_ids.append(codes)
    assert np.array_equal(checkpoint_ids[-1], sids)
    np.save(destination / "rqvae_checkpoint_epochs.npy", np.asarray(checkpoint_epochs))
    np.save(destination / "rqvae_checkpoint_semantic_ids.npy", np.stack(checkpoint_ids))
    with torch.inference_mode():
        np.save(destination / "item_latent_vectors.npy", rq.encode(dataset.item_embeddings).cpu().numpy())
        np.save(destination / "rqvae_codebooks.npy", np.stack([layer.weight.detach().cpu().numpy() for layer in rq.layers]))
    summary["checkpoint_epochs_one_based"] = checkpoint_epochs
    config_names = [
        "num_layers", "num_decoder_layers", "d_model", "d_ff", "num_heads", "d_kv",
        "dropout_rate", "feed_forward_proj", "sem_id_dim", "use_sdpa", "first_beam_only",
    ]
    model_config = {name: setting(name) for name in config_names}
    model_config.update(vocab_size=int(setting("codebook_size")) * int(setting("sem_id_dim")) + 1,
                        pad_token_id=0, eos_token_id=0)
    (destination / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    model = Tiger(model_config)
    model.load_state_dict(torch.load(run / "tiger/best_model.pt", map_location="cpu", weights_only=True))
    model.to("cuda").eval()
    summary["predictions"] = {
        "valid": export_predictions(model, splits["valid"], destination, sids, code_to_items, result["best_valid"]),
        "test": export_predictions(model, splits["test"], destination, sids, code_to_items, result["final_test"]),
    }
    (destination / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
