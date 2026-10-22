"""Collect lossless existing Beauty TIGER beams; never train or create labels.

All inputs must already exist. Uses original t5_collate and optional passive
_beam_trace_recorder in local GenRec T5. No datasets or models are downloaded.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from cotiger.core.data.schemas import SeqData
from cotiger.core.data.amazon import parse_gzip_json
from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import t5_collate
from .raw_utils import (D, K, V, OFFSETS, SCHEMA_VERSION, RawRecorder, add_readable_paths,
                       read_json, require, sha256, state_sha256, verify_tensors, write_json)


def now():
    return datetime.now(timezone.utc).isoformat()


def command_output(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rqvae-checkpoint", type=Path, required=True)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--resolved-gin", type=Path, required=True)
    p.add_argument("--embedding-parquet", type=Path, required=True)
    p.add_argument("--reviews-gzip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    p.add_argument("--max-samples", type=int, default=None, help="Per split; omitted means all")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--valid-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--batch-size", type=int, default=None, help="Defaults to original infer_batch_size")
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--compression-level", type=int, choices=range(1, 10), default=3, help="Lossless NPZ DEFLATE level")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-logits", action="store_true", help="Omit optional logits only; logprobs always FP32")
    p.add_argument("--no-parent-hidden", action="store_true")
    p.add_argument("--verify-every-batch", action="store_true", help="Compare baseline and recorded generation for all rows")
    p.add_argument("--min-free-gib", type=float, default=2)
    a = p.parse_args()
    for key in ("checkpoint", "rqvae_checkpoint", "analysis_dir", "resolved_gin", "embedding_parquet", "reviews_gzip", "output_dir"):
        setattr(a, key, getattr(a, key).resolve())
    require(len(set(a.splits)) == len(a.splits), "Repeated split")
    for limit in (a.max_samples, a.train_limit, a.valid_limit, a.test_limit):
        require(limit is None or limit > 0, "Sample limits must be positive")
    require(not a.output_dir.is_relative_to(a.analysis_dir), "Cannot overwrite source analysis")
    return a


def original_settings(path):
    text = path.read_text(encoding="utf-8")
    names = ("beam_size", "codebook_size", "sem_id_dim", "max_seq_len", "dynamic_padding",
             "pad_to_multiple_of", "amp", "mixed_precision_type", "infer_batch_size", "seed")
    values = {}
    for name in names:
        match = re.search(rf"^train\.{name}\s*=\s*(.+)$", text, re.MULTILINE)
        require(match is not None, f"Missing original setting train.{name}")
        values[name] = ast.literal_eval(match.group(1))
    require((values["beam_size"], values["codebook_size"], values["sem_id_dim"]) == (K, 256, D),
            "Collector schema must match existing three-code, beam-15 model")
    require(values["mixed_precision_type"] in ("fp16", "bf16"), "Unsupported original autocast precision")
    require("AmazonSeqDataset.add_disambiguation = False" in text, "Existing run must use three-token SID")
    return values


def check_sources(a, sids, frames, output):
    """Validate exported indices against original ordered embedding mapping and splits."""
    mapping = pd.read_parquet(a.analysis_dir / "item_sid.parquet").sort_values("item_index").reset_index(drop=True)
    embedding = pd.read_parquet(a.embedding_parquet, columns=["ItemID"])
    require(len(mapping) == len(sids) == len(embedding), "Item/embedding/SID row count mismatch")
    require(np.array_equal(mapping.item_index.to_numpy(), np.arange(len(sids))), "Item index is not zero based ordered")
    require(np.array_equal(mapping.embedding_row.to_numpy(), np.arange(len(sids))), "Embedding row mismatch")
    require(np.array_equal(mapping.embedding_item_id.to_numpy(), embedding.ItemID.to_numpy()), "Embedding item ID mismatch")
    all_asins, item_asins, seen_all, raw_item_index, raw_users = [], [], set(), {}, {}
    for review in parse_gzip_json(a.reviews_gzip):
        asin, uid = review.get("asin"), review.get("reviewerID")
        if asin and asin not in seen_all:
            seen_all.add(asin)
            all_asins.append(asin)
        if asin and uid:
            if asin not in raw_item_index:
                raw_item_index[asin] = len(item_asins)
                item_asins.append(asin)
            raw_users.setdefault(uid, []).append((review.get("unixReviewTime", 0), raw_item_index[asin]))
    embedding_asins = [all_asins[int(item) - 1] for item in embedding.ItemID]
    require(mapping.asin.tolist() == item_asins and mapping.embedding_asin.tolist() == embedding_asins and
            item_asins == embedding_asins, "Raw review/ASIN/embedding mismatch")
    require(np.array_equal(mapping[["sid_0", "sid_1", "sid_2"]].to_numpy(), sids), "SID mapping row mismatch")
    groups = defaultdict(list)
    for index, sid in enumerate(sids.tolist()):
        groups[tuple(sid)].append(index)
    mapping["offset_token_0"] = sids[:, 0] + 1
    mapping["offset_token_1"] = sids[:, 1] + 257
    mapping["offset_token_2"] = sids[:, 2] + 513
    mapping["sid_collision"] = [len(groups[tuple(sid)]) > 1 for sid in sids.tolist()]
    mapping["same_sid_item_indices"] = [groups[tuple(sid)] for sid in sids.tolist()]
    mapping["same_sid_asins"] = [[str(mapping.asin.iloc[i]) for i in groups[tuple(sid)]] for sid in sids.tolist()]
    require(np.array_equal(mapping.collision_group_size.to_numpy(), [len(groups[tuple(s)]) for s in sids]),
            "Exported collision sizes mismatch")
    mapping.to_parquet(output / "metadata/item_sid_mapping.parquet", index=False)
    np.save(output / "metadata/item_semantic_ids.npy", sids, allow_pickle=False)
    users = pd.read_parquet(a.analysis_dir / "user_sequences.parquet")
    original_users = [(uid, [item for _, item in sorted(values, key=lambda pair: pair[0])])
                      for uid, values in raw_users.items() if len(values) >= 5]
    require(users.reviewer_id.tolist() == [uid for uid, _ in original_users] and
            [list(seq) for seq in users.item_indices] == [seq for _, seq in original_users],
            "Original timestamp-sorted user sequences differ from source export")
    sequences = {int(row.user_index): [int(x) for x in row.item_indices] for row in users.itertuples()}
    checked = {}
    for split, frame in frames.items():
        require(np.array_equal(frame.sample_index.to_numpy(), np.arange(len(frame))), f"{split} original dataset indices differ")
        require(not frame.sample_index.duplicated().any(), f"Duplicate {split} source index")
        expected_count = sum(len(seq) - 3 for seq in sequences.values()) if split == "train" else len(sequences)
        require(len(frame) == expected_count, f"{split} source count does not match original leave-one-out split")
        for row in frame.itertuples():
            target = int(row.target_item_index)
            history = [int(x) for x in row.history_item_indices]
            require(0 <= target < len(sids) and all(0 <= x < len(sids) for x in history), "Out-of-range item index")
            require(np.array_equal(row.target_sid, sids[target]), "Target SID does not match item")
            require(str(row.target_asin) == str(mapping.asin.iloc[target]), "Target ASIN mismatch")
            require(int(row.history_length) == len(history) and 1 <= len(history) <= 50, "History length mismatch")
            seq = sequences[int(row.user_index)]
            full = int(row.full_history_length)
            if split == "train":
                require(1 <= full < len(seq) - 2, "Train sample leaks held-out interaction")
                expected_history, expected_target = seq[:full][-50:], seq[full]
            elif split == "valid":
                expected_history, expected_target = seq[:-2][-50:], seq[-2]
            else:
                expected_history, expected_target = seq[:-1][-50:], seq[-1]
            require(history == expected_history and target == expected_target, f"{split} history/target does not match original split")
        checked[split] = len(frame)
    report = {"state": "passed", "items": len(sids), "unique_sids": len(groups),
              "collision_groups": sum(len(items) > 1 for items in groups.values()),
              "items_in_collision_groups": sum(len(items) for items in groups.values() if len(items) > 1),
              "largest_collision_group": max(map(len, groups.values())), "mapping_anomalies": 0,
              "original_internal_indices_zero_based": True, "embedding_row_matches_internal_index": True,
              "embedding_item_id_matches_original_embedding_file": True, "asin_matches_embedding_asin": True,
              "split_samples_checked": checked, "source_sids_modified": False}
    write_json(output / "metadata/sid_mapping_check.json", report)
    return report


def make_records(frame, split, sids, settings):
    records, seq = [], []
    for row in frame.itertuples():
        history = [int(x) for x in row.history_item_indices]
        target = sids[int(row.target_item_index)].astype(np.int64).tolist()
        history_sids = sids[history].astype(np.int64)
        index = int(row.sample_index)
        records.append({"schema_version": SCHEMA_VERSION, "sample_id": f"{split}_{index:07d}",
                        "split": split, "dataset_index": index, "user_index": int(row.user_index),
                        "reviewer_id": str(row.reviewer_id), "history_item_ids": history,
                        "history_sid": history_sids.tolist(), "history_length_items": len(history),
                        "history_length": int(history_sids.size), "full_history_length_items": int(row.full_history_length),
                        "target_item_id": int(row.target_item_index), "target_asin": str(row.target_asin),
                        "target_sid": target, "target_tokens": (np.array(target) + OFFSETS).tolist(),
                        "target_sid_collision_group_size": int(row.target_sid_group_size), "beam_size": K,
                        "sid_length": D, "decoder_start_token_id": 0})
        # User ID never identifies a sample and is not an input to Tiger.
        seq.append(SeqData(int(row.user_index), history_sids.reshape(-1).tolist(), target))
    batch = t5_collate(seq, codebook_size=256, sem_id_dim=D, max_history_tokens=settings["max_seq_len"] * D,
                       dynamic_padding=settings["dynamic_padding"], pad_to_multiple_of=settings["pad_to_multiple_of"])
    for record, ids, mask, labels in zip(records, batch["input_ids"], batch["attention_mask"], batch["labels"]):
        record["input_ids"], record["attention_mask"] = ids.tolist(), mask.tolist()
        require(record["history_length"] == int(mask.sum()), "Original collate length mismatch")
        expected = (np.array(record["history_sid"]) + OFFSETS).reshape(-1).tolist()
        require(ids[mask.bool()].tolist() == expected and np.array_equal(labels.numpy(), record["target_tokens"]),
                "Original collate/offset encoding mismatch")
        require(all(x == 0 for x, valid in zip(record["input_ids"], record["attention_mask"]) if not valid), "Nonzero input padding")
    return records, batch


@torch.inference_mode()
def collect(a):
    require(torch.cuda.is_available(), "CUDA is not available")
    paths = {"tiger_checkpoint": a.checkpoint, "rqvae_checkpoint": a.rqvae_checkpoint,
             "model_config": a.analysis_dir / "model_config.json", "sid_map": a.analysis_dir / "item_semantic_ids.npy",
             "resolved_gin": a.resolved_gin, "embedding_parquet": a.embedding_parquet, "reviews_gzip": a.reviews_gzip,
             "item_mapping": a.analysis_dir / "item_sid.parquet", "user_sequences": a.analysis_dir / "user_sequences.parquet"}
    paths.update({f"{s}_source_samples": a.analysis_dir / f"{s}_samples.parquet" for s in a.splits})
    require(all(p.is_file() for p in paths.values()), "Missing existing source files: " + str([str(p) for p in paths.values() if not p.is_file()]))
    original = original_settings(a.resolved_gin)
    batch_size = a.batch_size or original["infer_batch_size"]
    require(batch_size >= 1 and a.shard_size >= batch_size, "Shard size must be at least batch size")
    config = read_json(paths["model_config"])
    require((config["sem_id_dim"], config["vocab_size"], config["pad_token_id"], config["eos_token_id"], config["d_model"]) ==
            (D, V, 0, 0, 128) and config["first_beam_only"], "Existing model does not match expected schema")
    sids = np.load(paths["sid_map"], allow_pickle=False)
    require(sids.ndim == 2 and sids.shape[1] == D and np.all((sids >= 0) & (sids < 256)), "Invalid original SID map")
    frames = {split: pd.read_parquet(a.analysis_dir / f"{split}_samples.parquet") for split in a.splits}
    hashes = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for name, path in paths.items()}
    limits = {s: getattr(a, s + "_limit") or a.max_samples for s in a.splits}
    settings = {"sources": hashes, "model_config": config, "original_generation": original,
                "code_sha256": {rel: sha256(ROOT / rel) for rel in ("cotiger/core/modules/t5.py", "cotiger/core/models/tiger.py", "cotiger/core/trainers/tiger_trainer.py", "cotiger/core/data/amazon.py")},
                "collector_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("collect_full_beam.py", "raw_utils.py")},
                "splits": a.splits, "limits": limits, "batch_size": batch_size, "shard_size": a.shard_size,
                "save_logits": not a.no_logits, "save_parent_hidden": not a.no_parent_hidden,
                "tensor_precision": "fp32", "compression_level": a.compression_level,
                "verify_every_batch": a.verify_every_batch, "schema_version": SCHEMA_VERSION}
    output, manifest_path = a.output_dir, a.output_dir / "collection_manifest.json"
    if manifest_path.is_file():
        require(a.resume, "Existing manifest; use --resume or a new output directory")
        manifest = read_json(manifest_path)
        require(manifest["settings"] == settings, "Resume source files or settings changed")
        for shard in manifest["shards"]:
            for file in shard["files"]:
                path = output / file["path"]
                require(path.is_file() and sha256(path) == file["sha256"], "Completed shard checksum mismatch")
        if manifest["state"] == "complete":
            inventory_path = output / "download_inventory.json"
            if not inventory_path.is_file():
                inventory = [{"path": str(p.relative_to(output)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(output.rglob("*")) if p.is_file() and not p.name.endswith(".partial")]
                write_json(inventory_path, {"state": "complete", "files": inventory,
                                           "total_bytes": sum(f["bytes"] for f in inventory),
                                           "total_samples": sum(c["success"] for c in manifest["counts"].values())})
            print(json.dumps({"state": "already_complete", "counts": manifest["counts"],
                              "skipped": sum(s["samples"] for s in manifest["shards"]), "new_samples": 0}), flush=True)
            return
        manifest.setdefault("previous_invocations", []).append({"state": manifest["state"], "counts": manifest.get("counts")})
    else:
        require(not output.exists() or not any(output.iterdir()), "Output directory must be empty")
        for name in ("metadata", "logs", *a.splits):
            (output / name).mkdir(parents=True, exist_ok=True)
        manifest = {"state": "starting", "created_at_utc": now(), "settings": settings, "shards": [], "counts": {}}
    started = time.perf_counter()
    sid_report = check_sources(a, sids, frames, output)
    for source, destination in ((a.resolved_gin, "original_resolved.gin"), (paths["model_config"], "model_config.json")):
        shutil.copyfile(source, output / "metadata" / destination)
    shutil.copyfile(Path(__file__).with_name("schema.md"), output / "metadata/schema.md")
    provenance_sources = ["cotiger/core/modules/t5.py", "cotiger/core/models/tiger.py", "cotiger/core/trainers/tiger_trainer.py", "cotiger/core/data/amazon.py"]
    for rel in provenance_sources:
        dest = output / "metadata/source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    for script in ("collect_full_beam.py", "raw_utils.py", "verify_full_beam.py"):
        dest = output / "metadata/collector" / script
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).with_name(script), dest)
    props = torch.cuda.get_device_properties(0)
    metadata = {"schema_version": SCHEMA_VERSION, "git_commit": command_output(["git", "rev-parse", "HEAD"]),
                "git_status": command_output(["git", "status", "--short"]), "sources": hashes, "settings": settings,
                "source_code_sha256": {rel: sha256(ROOT / rel) for rel in provenance_sources},
                "dataset": "Amazon 2014 Beauty 5-core", "split_method": "chronological leave-one-out; train sliding windows excluding last 2 interactions",
                "source_split_counts": {s: len(f) for s, f in frames.items()}, "source_sample_tables": "existing original offline export; fully checked against user_sequences",
                "seed": original["seed"], "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                "gpu_name": props.name, "gpu_total_bytes": props.total_memory, "gpu_free_bytes_at_start": torch.cuda.mem_get_info()[0],
                "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
                "offset_rule": "offset_token = raw_sid_code + sid_position*256 + 1; positions 0,1,2",
                "decoder_start_token_id": 0, "pad_token_id": 0, "eos_token_id": 0,
                "generation": {"fixed_generated_length": 3, "return_beams": 15, "early_eos_termination": False,
                               "length_penalty": False, "trie_constraint": False, "invalid_filter": False,
                               "rank_order": "original sorted torch.topk over parent-major flattened candidates"},
                "parent_hidden_semantics": "last-layer final-norm decoder state after parent prefix, BEFORE extension; depth d has prefix length d-1",
                "initial_placeholders": "ranks 2..15 have score -1e9; reused first-parent distributions, forward_evaluated=False",
                "gold_diagnostic": "separate frozen teacher-forced forward after free generation; all gold probabilities saved; no beam selection uses target",
                "sid_mapping_check": sid_report, "no_training_performed": True, "no_prefix_labels_created": True,
                "created_at_utc": now()}
    write_json(output / "metadata/run_metadata.json", metadata)
    model = Tiger(config).eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True), strict=True)
    before = state_sha256(model)
    model.to("cuda")
    require(all(not p.requires_grad for p in model.parameters()), "TIGER not frozen")
    torch.manual_seed(original["seed"])
    np.random.seed(original["seed"])
    torch.cuda.reset_peak_memory_stats()
    timings = {"baseline_generate_seconds": 0.0, "recorded_generate_seconds": 0.0, "gold_forward_seconds": 0.0,
               "cpu_integrity_and_json_seconds": 0.0, "shard_write_seconds": 0.0}
    dtype = torch.float16 if original["mixed_precision_type"] == "fp16" else torch.bfloat16
    baseline_samples, peak_batch, skipped, failed_attempts = 0, batch_size, 0, 0

    def generate_batch(frame, split, compare):
        nonlocal baseline_samples
        records, batch = make_records(frame, split, sids, original)
        ids, mask = batch["input_ids"].to("cuda"), batch["attention_mask"].to("cuda")
        with torch.autocast("cuda", dtype=dtype, enabled=original["amp"]):
            if compare:
                torch.cuda.synchronize()
                t = time.perf_counter()
                baseline = model.generate(ids, mask, num_beams=K)
                torch.cuda.synchronize()
                timings["baseline_generate_seconds"] += time.perf_counter() - t
            recorder = RawRecorder(len(records), not a.no_logits, not a.no_parent_hidden)
            hook = model.model.decoder.register_forward_hook(recorder.decoder_hook) if not a.no_parent_hidden else None
            require(not hasattr(model.model, "_beam_trace_recorder"), "Another recorder is already installed")
            model.model._beam_trace_recorder = recorder
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                generated = model.generate(ids, mask, num_beams=K)
            finally:
                delattr(model.model, "_beam_trace_recorder")
                if hook:
                    hook.remove()
            torch.cuda.synchronize()
            timings["recorded_generate_seconds"] += time.perf_counter() - t
            if compare:
                require(torch.equal(baseline, generated), "Optional recorder changed original final beam output")
                baseline_samples += len(records)
                del baseline
            data = recorder.tensors(generated)
            # Finished free-generation recording before reading gold target tokens.
            del generated, recorder
            targets = batch["labels"].to("cuda")
            decoder_ids = torch.cat([torch.zeros((len(records), 1), dtype=torch.long, device="cuda"), targets[:, :-1]], dim=1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            gold_output = model.model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_ids,
                                      decoder_attention_mask=torch.ones_like(decoder_ids), use_cache=False)
            gold_logp = torch.log_softmax(gold_output.logits.float(), dim=-1)
            gold_chosen = gold_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            data["gold_next_token_logprobs"] = gold_logp.cpu().numpy().copy()
            data["gold_next_token_ids"] = targets.cpu().numpy().astype(np.int16)
            data["gold_token_logprobs"] = gold_chosen.cpu().numpy().copy()
            data["gold_cumulative_logprobs"] = np.cumsum(data["gold_token_logprobs"], axis=1, dtype=np.float32)
            data["gold_token_ranks"] = (1 + (gold_logp > gold_chosen.unsqueeze(-1)).sum(-1)).cpu().numpy().astype(np.int16)
            data["gold_equal_score_token_counts"] = (gold_logp == gold_chosen.unsqueeze(-1)).sum(-1).cpu().numpy().astype(np.int16)
            torch.cuda.synchronize()
            timings["gold_forward_seconds"] += time.perf_counter() - t
            del gold_output, gold_logp, gold_chosen, targets, decoder_ids, ids, mask, batch
        t = time.perf_counter()
        check = verify_tensors(data)
        add_readable_paths(records, data)
        data["dataset_indices"] = frame.sample_index.to_numpy(dtype=np.int64)
        timings["cpu_integrity_and_json_seconds"] += time.perf_counter() - t
        return records, data, check

    def safe_batch(frame, split, compare):
        nonlocal failed_attempts
        try:
            return [generate_batch(frame, split, compare)]
        except torch.cuda.OutOfMemoryError:
            failed_attempts += 1
            with (output / "logs/retry_attempts.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"split": split, "dataset_indices": frame.sample_index.tolist(), "reason": "CUDA OOM; reducing batch"}) + "\n")
            torch.cuda.empty_cache()
            require(len(frame) > 1, "Single sample cannot fit CUDA memory")
            midpoint = len(frame) // 2
            return safe_batch(frame.iloc[:midpoint], split, True) + safe_batch(frame.iloc[midpoint:], split, True)
        except Exception as error:
            # Atomic shard writing means an individual failure cannot corrupt
            # completed data. Isolate failures; abort on bad row rather than
            # silently claiming complete coverage. Resume retries this shard.
            if len(frame) > 1:
                midpoint = len(frame) // 2
                return safe_batch(frame.iloc[:midpoint], split, True) + safe_batch(frame.iloc[midpoint:], split, True)
            with (output / "logs/failed_samples.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": f"{split}_{int(frame.iloc[0].sample_index):07d}", "error": str(error), "utc": now()}) + "\n")
            manifest["counts"][split]["failed"] += 1
            raise

    try:
        manifest["state"] = "running"
        write_json(manifest_path, manifest)
        checks = []
        for split, source_frame in frames.items():
            frame = source_frame if limits[split] is None else source_frame.head(limits[split])
            expected = len(frame)
            manifest["counts"][split] = {"source": len(source_frame), "requested": expected, "success": 0, "failed": 0, "skipped": 0}
            existing = {x["start_row"]: x for x in manifest["shards"] if x["split"] == split}
            for start in range(0, expected, a.shard_size):
                end = min(start + a.shard_size, expected)
                if start in existing:
                    require(existing[start]["end_row_exclusive"] == end, "Resume shard boundary mismatch")
                    manifest["counts"][split]["success"] += end - start
                    manifest["counts"][split]["skipped"] += end - start
                    skipped += end - start
                    checks.append(existing[start]["integrity"])
                    continue
                require(shutil.disk_usage(output).free > a.min_free_gib * 2**30, "Insufficient remaining disk space; complete shards retained for resume")
                name = f"shard_{start:07d}_{end:07d}"
                tensor_rel, json_rel = f"{split}/{name}.npz", f"{split}/{name}.jsonl.gz"
                rows, chunks, shard_checks = [], [], []
                for pos in range(start, end, batch_size):
                    current = frame.iloc[pos:min(pos + batch_size, end)]
                    compare = a.verify_every_batch or start == 0 and pos == start
                    for records, data, check in safe_batch(current, split, compare):
                        offset = len(rows)
                        for i, record in enumerate(records):
                            record["tensor_shard"], record["tensor_row"] = tensor_rel, offset + i
                        rows.extend(records)
                        chunks.append(data)
                        shard_checks.append(check)
                require(len(rows) == end - start and len(set(r["sample_id"] for r in rows)) == len(rows), "Incomplete/duplicate shard")
                arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
                require(np.array_equal(arrays["dataset_indices"], np.arange(start, end)), "Cross-batch sample ordering mismatch")
                del chunks
                t = time.perf_counter()
                tensor_tmp = output / (tensor_rel + ".partial")
                with zipfile.ZipFile(tensor_tmp, "w", compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=a.compression_level, allowZip64=True) as archive:
                    for key, array in arrays.items():
                        with archive.open(key + ".npy", "w", force_zip64=True) as f:
                            np.save(f, array, allow_pickle=False)
                json_tmp = output / (json_rel + ".partial")
                with gzip.open(json_tmp, "wt", encoding="utf-8", compresslevel=3) as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                tensor_tmp.replace(output / tensor_rel)
                json_tmp.replace(output / json_rel)
                file_records = [{"path": rel, "bytes": (output / rel).stat().st_size, "sha256": sha256(output / rel)}
                                for rel in (tensor_rel, json_rel)]
                timings["shard_write_seconds"] += time.perf_counter() - t
                shapes = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}
                integrity = {"samples": len(rows), "final_tokens_exact": True, "all_parent_paths_exact": True,
                             "max_score_abs_error": max(x["max_score_abs_error"] for x in shard_checks),
                             "max_probability_sum_error": max(x["max_probability_sum_error"] for x in shard_checks),
                             "steps_with_score_ties": sum(x["steps_with_score_ties"] for x in shard_checks),
                             "steps_cpu_stable_tie_order_differs": sum(x["steps_cpu_stable_tie_order_differs"] for x in shard_checks)}
                entry = {"split": split, "start_row": start, "end_row_exclusive": end, "samples": len(rows),
                         "files": file_records, "tensor_shapes": shapes, "integrity": integrity}
                manifest["shards"].append(entry)
                checks.append(integrity)
                manifest["counts"][split]["success"] += len(rows)
                manifest.update(updated_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                                baseline_compared_samples=baseline_samples, timings=timings)
                write_json(manifest_path, manifest)
                print(json.dumps({"split": split, "success": manifest["counts"][split]["success"], "requested": expected,
                                  "shard_bytes": sum(f["bytes"] for f in file_records), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
                del rows, arrays
            combined = output / split / "samples.jsonl.gz"
            tmp = combined.with_name(combined.name + ".partial")
            with tmp.open("wb") as out:
                for shard in sorted((s for s in manifest["shards"] if s["split"] == split), key=lambda x: x["start_row"]):
                    path = next(f["path"] for f in shard["files"] if f["path"].endswith(".jsonl.gz"))
                    with (output / path).open("rb") as inp:
                        shutil.copyfileobj(inp, out, length=4 * 2**20)
            tmp.replace(combined)
        after = state_sha256(model)
        require(before == after and all(p.grad is None for p in model.parameters()), "Frozen TIGER parameters changed")
        require(all(c["success"] == c["requested"] for c in manifest["counts"].values()), "Not all requested rows collected")
        report = {"state": "complete", "counts": manifest["counts"], "total_samples": sum(c["success"] for c in manifest["counts"].values()),
                  "baseline_compared_samples": baseline_samples, "baseline_final_tokens_exact": True,
                  "max_reconstructed_score_abs_error": max(x["max_score_abs_error"] for x in checks),
                  "max_probability_sum_error": max(x["max_probability_sum_error"] for x in checks),
                  "steps_with_score_ties": sum(x["steps_with_score_ties"] for x in checks),
                  "steps_cpu_stable_tie_order_differs": sum(x["steps_cpu_stable_tie_order_differs"] for x in checks),
                  "state_sha256_before": before, "state_sha256_after": after, "frozen_tiger_unchanged": True,
                  "elapsed_seconds": time.perf_counter() - started, "timings": timings,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "skipped_this_invocation": skipped, "retry_attempts": failed_attempts, "no_training_or_labels": True,
                  "completed_at_utc": now()}
        write_json(output / "logs/collection_report.json", report)
        metadata.update(completed_at_utc=now(), collected_counts=manifest["counts"], frozen_tiger_state_sha256=after)
        write_json(output / "metadata/run_metadata.json", metadata)
        manifest.update(state="complete", report=report, completed_at_utc=now())
        write_json(manifest_path, manifest)
        # Inventory includes every final output except itself. Downloads verify
        # each file against this immutable end-of-run checksum inventory.
        inventory = [{"path": str(p.relative_to(output)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in sorted(output.rglob("*")) if p.is_file() and not p.name.endswith(".partial") and p.name != "download_inventory.json"]
        write_json(output / "download_inventory.json", {"state": "complete", "files": inventory,
                                                       "total_bytes": sum(f["bytes"] for f in inventory), "total_samples": report["total_samples"]})
        print(json.dumps(report), flush=True)
    except Exception as error:
        manifest.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), updated_at_utc=now())
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    collect(arguments())
