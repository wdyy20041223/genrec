"""Prepare the three Beauty Parquet files one raw Train shard at a time."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .negative_sampler import OFFSETS, ROOT, SIDCatalog, position_valid, shared_length

from .paths import RAW_ROOT as DEFAULT_RAW, PREFIX_DATA
DEFAULT_OUT = PREFIX_DATA.parent
DOCS = ROOT / "docs/generated/tiger_prefix_ft_beauty_20261013"
TRAIN_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset_index", pa.int64()),
    ("input_ids", pa.list_(pa.int16())), ("attention_mask", pa.list_(pa.int8())),
    ("target_tokens", pa.list_(pa.int16(), 3)), ("target_sid", pa.list_(pa.int16(), 3)),
    ("target_item_id", pa.int64()), ("tensor_shard", pa.string()), ("tensor_row", pa.int32()),
])
CATALOG_SCHEMA = pa.schema([
    ("raw_sid", pa.list_(pa.int16(), 3)), ("offset_tokens", pa.list_(pa.int16(), 3)),
    ("collision_group_size", pa.int32()),
])
HARD_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("dataset_index", pa.int64()), ("depth", pa.int8()),
    ("negative_tokens", pa.list_(pa.int16())), ("score", pa.float32()), ("gold_score", pa.float32()),
    ("source", pa.string()), ("survived_beam", pa.bool_()),
    ("shared_prefix_length", pa.int8()), ("from_gold_parent", pa.bool_()),
    ("parent_beam_index", pa.int16()), ("next_token_id", pa.int16()),
])
REQUIRED = {
    "next_token_logprobs": (3, 15, 769), "beam_scores_before": (3, 15),
    "parent_prefix_tokens_with_start": (3, 15, 3), "active_beam_mask": (3, 15),
    "selected_parent_indices": (3, 15), "selected_token_ids": (3, 15),
    "selected_flat_indices": (3, 15), "selected_cumulative_scores": (3, 15),
    "final_tokens_with_start": (15, 4), "final_cumulative_scores": (15,),
    "gold_next_token_logprobs": (3, 769), "gold_next_token_ids": (3,),
    "gold_token_logprobs": (3,), "gold_cumulative_logprobs": (3,),
    "gold_token_ranks": (3,), "dataset_indices": (),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def catalog_from_mapping(raw_root):
    mapping = pq.read_table(raw_root / "metadata/item_sid_mapping.parquet").to_pylist()
    ids = np.load(raw_root / "metadata/item_semantic_ids.npy", allow_pickle=False)
    groups = Counter(tuple(row[f"sid_{i}"] for i in range(3)) for row in mapping)
    require(len(mapping) == len(ids), "Mapping/NPY item counts differ")
    seen_items = set()
    for row in mapping:
        item = row["item_index"]
        require(item not in seen_items and 0 <= item < len(ids), "Invalid/duplicate item index")
        seen_items.add(item)
        sid = tuple(row[f"sid_{i}"] for i in range(3))
        tokens = tuple(row[f"offset_token_{i}"] for i in range(3))
        require(sid == tuple(ids[item].tolist()), "Mapping/NPY SID mismatch")
        require(tokens == tuple(sid[i] + OFFSETS[i] for i in range(3)), "Catalog Offset mismatch")
        require(row["collision_group_size"] == groups[sid], "Collision count mismatch")
        require(len(row["same_sid_item_indices"]) == groups[sid], "Collision item list mismatch")
    rows = [{"raw_sid": list(sid), "offset_tokens": [sid[i] + OFFSETS[i] for i in range(3)],
             "collision_group_size": count} for sid, count in sorted(groups.items())]
    return SIDCatalog(rows), ids, len(mapping)


def load_shard(path):
    with np.load(path, allow_pickle=False) as z:
        require(set(REQUIRED).issubset(z.files), f"Missing NPZ keys: {path}")
        data = {key: z[key] for key in REQUIRED}
    n = len(data["dataset_indices"])
    for key, tail in REQUIRED.items():
        require(data[key].shape == (n, *tail), f"Wrong shape: {path.name}/{key}")
        if "logprob" in key or "score" in key:
            require(data[key].dtype == np.float32 and np.isfinite(data[key]).all(), f"Non-finite/non-FP32 {key}")
        elif key != "active_beam_mask":
            require(np.issubdtype(data[key].dtype, np.integer), f"Non-integer index/Token: {key}")
    require(data["active_beam_mask"].dtype == np.bool_, "Invalid active mask type")
    require(np.all(data["active_beam_mask"][:, 0, 0]) and not np.any(data["active_beam_mask"][:, 0, 1:]),
            "First-step placeholders were marked active")
    require(np.all(data["gold_cumulative_logprobs"] == np.cumsum(data["gold_token_logprobs"], axis=1, dtype=np.float32)),
            "Gold cumulative score mismatch")
    return data


def validate_record(record, data, row, relative, item_sids):
    sid = record["sample_id"]
    require(record["split"] == "train" and sid == f"train_{record['dataset_index']:07d}", "Non-Train/malformed ID")
    require(record["tensor_shard"] == relative and record["tensor_row"] == row, f"Tensor pointer mismatch: {sid}")
    require(int(data["dataset_indices"][row]) == record["dataset_index"], f"NPZ dataset index mismatch: {sid}")
    tokens = record["target_tokens"]
    require(tokens == data["gold_next_token_ids"][row].tolist(), f"Gold target mismatch: {sid}")
    require(position_valid(tokens) and len(tokens) == 3, f"Invalid target: {sid}")
    require(tokens == [record["target_sid"][i] + OFFSETS[i] for i in range(3)], f"Target Offset mismatch: {sid}")
    require(0 <= record["target_item_id"] < len(item_sids), f"Target item outside mapping: {sid}")
    require(record["target_sid"] == item_sids[record["target_item_id"]].tolist(), f"Target item mismatch: {sid}")
    history_items = np.asarray(record["history_item_ids"], dtype=np.int64)
    require(np.all((0 <= history_items) & (history_items < len(item_sids))), f"History item outside mapping: {sid}")
    require(item_sids[history_items].tolist() == record["history_sid"], f"History item/SID mismatch: {sid}")
    inputs, mask = record["input_ids"], record["attention_mask"]
    require(len(inputs) == len(mask) and all(m in (0, 1) for m in mask), f"Invalid input/mask: {sid}")
    require(mask == sorted(mask), f"Input is not left-padded: {sid}")
    require(all(token == 0 for token, m in zip(inputs, mask) if not m), f"Nonzero PAD: {sid}")
    actual = [token for token, m in zip(inputs, mask) if m]
    expected = [code + OFFSETS[i] for history_sid in record["history_sid"] for i, code in enumerate(history_sid)]
    require(actual == expected, f"History Token mismatch: {sid}")
    require(record["decoder_start_token_id"] == 0 and record["sid_length"] == 3 and record["beam_size"] == 15,
            "Unexpected generation encoding")
    for depth in (1, 2, 3):
        parents = data["parent_prefix_tokens_with_start"][row, depth - 1]
        require(np.all(parents[:, 0] == 0) and np.all(parents[:, depth:] == -1), f"Start/tail mismatch: {sid}")
        selected_parents = data["selected_parent_indices"][row, depth - 1]
        selected_tokens = data["selected_token_ids"][row, depth - 1]
        require(np.all((0 <= selected_parents) & (selected_parents < 15)), "Selected parent outside range")
        require(np.all(data["active_beam_mask"][row, depth - 1, selected_parents]), "Selected inactive parent")
        require(np.array_equal(data["selected_flat_indices"][row, depth - 1], selected_parents.astype(np.int32) * 769 + selected_tokens),
                "Selected flat-index mismatch")
        scores = np.add(data["beam_scores_before"][row, depth - 1, selected_parents],
                        data["next_token_logprobs"][row, depth - 1, selected_parents, selected_tokens], dtype=np.float32)
        require(np.array_equal(scores, data["selected_cumulative_scores"][row, depth - 1]), "Selected score mismatch")
        paths = np.concatenate((parents[selected_parents, :depth], selected_tokens[:, None]), axis=1)
        if depth < 3:
            require(np.array_equal(paths, data["parent_prefix_tokens_with_start"][row, depth, :, :depth + 1]), "Parent/child mismatch")
        else:
            require(np.array_equal(paths, data["final_tokens_with_start"][row]), "Final path mismatch")
            require(np.array_equal(scores, data["final_cumulative_scores"][row]), "Final score mismatch")
        gold = record["gold_teacher_forced"][depth - 1]
        require(gold["depth"] == depth and gold["gold_cumulative_logprob"] == float(data["gold_cumulative_logprobs"][row, depth - 1]),
                "JSON/NPZ Gold cumulative mismatch")


def mine_depth(record, data, row, depth, catalog, limit=30):
    """All actual active parents, all vocabulary scores, then legal-prefix filtering."""
    target = tuple(record["target_tokens"])
    positive = target[:depth]
    offset = depth - 1
    active = np.flatnonzero(data["active_beam_mask"][row, offset])
    scores = np.add(data["next_token_logprobs"][row, offset],
                    data["beam_scores_before"][row, offset, :, None], dtype=np.float32)
    selected = set(zip(data["selected_parent_indices"][row, offset].tolist(), data["selected_token_ids"][row, offset].tolist()))
    candidates = {}
    stats = Counter(beam_active_parents=len(active), beam_all_vocabulary_candidates=len(active) * 769,
                    beam_wrong_position_candidates=len(active) * (769 - 256))

    def add(prefix, score, source, parent, token, survived):
        if prefix == positive:
            stats[f"{source}_positive_excluded"] += 1
            return
        require(position_valid(prefix) and prefix in catalog.prefix_sets[depth], "Mining produced illegal prefix")
        candidate = {"sample_id": record["sample_id"], "dataset_index": record["dataset_index"], "depth": depth,
                     "negative_tokens": list(prefix), "score": float(score),
                     "gold_score": float(data["gold_cumulative_logprobs"][row, offset]), "source": source,
                     "survived_beam": bool(survived), "shared_prefix_length": shared_length(prefix, positive),
                     "from_gold_parent": prefix[:-1] == target[:depth - 1],
                     "parent_beam_index": int(parent), "next_token_id": int(token)}
        previous = candidates.get(prefix)
        if previous is None:
            candidates[prefix] = candidate
        elif previous["source"] == "beam" and source == "gold_conditional":
            stats["beam_gold_overlap_beam_preferred"] += 1
        elif candidate["score"] > previous["score"] or (candidate["score"] == previous["score"] and survived):
            candidates[prefix] = candidate
            stats["duplicates_removed"] += 1

    for parent in active:
        prefix = tuple(data["parent_prefix_tokens_with_start"][row, offset, parent, 1:depth].tolist())
        children = catalog.children.get(prefix, np.empty(0, dtype=np.int64))
        stats["beam_catalog_rejected_position_valid"] += 256 - len(children)
        stats["beam_catalog_legal_candidates"] += len(children)
        for token in children:
            add((*prefix, int(token)), scores[parent, token], "beam", parent, token, (int(parent), int(token)) in selected)
    gold_parent = target[:depth - 1]
    gold_base = np.float32(0) if depth == 1 else data["gold_cumulative_logprobs"][row, depth - 2]
    gold_children = catalog.children.get(gold_parent, np.empty(0, dtype=np.int64))
    stats["gold_catalog_legal_candidates"] += len(gold_children)
    for token in gold_children:
        score = np.add(gold_base, data["gold_next_token_logprobs"][row, offset, token], dtype=np.float32)
        add((*gold_parent, int(token)), score, "gold_conditional", -1, token, False)
    ordered = sorted(candidates.values(), key=lambda r: (-r["score"], r["negative_tokens"]))
    stats["unique_legal_negatives_before_limit"] = len(ordered)
    stats["negatives_removed_by_limit"] = max(0, len(ordered) - limit)
    return ordered[:limit], stats


def prepare(raw_root, output_dir, report_path, max_samples=None, hard_limit=30):
    started = time.monotonic()
    raw_root, output_dir, report_path = Path(raw_root).resolve(), Path(output_dir).resolve(), Path(report_path).resolve()
    require(hard_limit > 0, "Hard limit must be positive")
    require(max_samples is None or max_samples > 0, "max-samples must be positive")
    manifest = json.loads((raw_root / "collection_manifest.json").read_text(encoding="utf-8"))
    require(manifest["state"] == "complete", "Raw collection is incomplete")
    specs = [s for s in manifest["shards"] if s["split"] == "train"]
    json_paths = sorted((raw_root / "train").glob("shard_*.jsonl.gz"))
    npz_paths = sorted((raw_root / "train").glob("shard_*.npz"))
    require(len(specs) == len(json_paths) == len(npz_paths), "Train shard counts differ")
    expected_npz = {f["path"] for spec in specs for f in spec["files"] if f["path"].endswith(".npz")}
    require(expected_npz == {str(p.relative_to(raw_root)).replace('\\', '/') for p in npz_paths}, "Manifest/NPZ inventory differs")
    catalog, item_sids, item_count = catalog_from_mapping(raw_root)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [data_dir / f"{name}.parquet" for name in ("train_samples", "sid_catalog", "hard_negatives")]
    require(not any(p.exists() or p.with_name(p.name + ".partial").exists() for p in final_paths),
            "Output exists; choose a new output directory rather than overwrite a run")
    require(shutil.disk_usage(data_dir).free > 2 * 1024**3, "Less than 2 GiB disk space available")
    partials = [p.with_name(p.name + ".partial") for p in final_paths]
    pq.write_table(pa.Table.from_pylist(catalog.rows, schema=CATALOG_SCHEMA), partials[1], compression="zstd", compression_level=3)
    totals = {d: Counter() for d in (1, 2, 3)}
    counts = {d: Counter() for d in (1, 2, 3)}
    input_lengths = Counter()
    processed = 0
    used_shards = []
    formats = []
    first_example = None
    state = {"state": "running", "raw_root": str(raw_root), "output_dir": str(output_dir), "hard_limit": hard_limit,
             "max_samples": max_samples, "source_train_samples": manifest["counts"]["train"]["source"],
             "source_train_shards": len(specs), "no_training_performed": True}
    progress_path = report_path.with_name(report_path.stem + "_progress.json")
    write_json(progress_path, state)
    try:
        with pq.ParquetWriter(partials[0], TRAIN_SCHEMA, compression="zstd", compression_level=3) as train_writer, \
             pq.ParquetWriter(partials[2], HARD_SCHEMA, compression="zstd", compression_level=3) as hard_writer:
            for json_path, spec in zip(json_paths, specs):
                if max_samples is not None and processed >= max_samples:
                    break
                npz_relative = next(f["path"] for f in spec["files"] if f["path"].endswith(".npz"))
                require(json_path.stem.removesuffix(".jsonl") == Path(npz_relative).stem, "JSON/NPZ shard names differ")
                shard = load_shard(raw_root / npz_relative)
                n = len(shard["dataset_indices"])
                require(n == spec["samples"], "Actual shard size differs from manifest")
                if not formats:
                    formats = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in shard.items()}
                train_rows, hard_rows = [], []
                seen_rows = 0
                with gzip.open(json_path, "rt", encoding="utf-8") as f:
                    for row, line in enumerate(f):
                        if max_samples is not None and processed >= max_samples:
                            break
                        require(row < n, "More JSON records than NPZ rows")
                        record = json.loads(line)
                        validate_record(record, shard, row, npz_relative, item_sids)
                        require(record["dataset_index"] == processed, "Missing, reordered, or duplicate training index")
                        train_rows.append({name: record[name] for name in TRAIN_SCHEMA.names})
                        input_lengths[len(record["input_ids"])] += 1
                        case = {"sample_id": record["sample_id"], "history_sid": record["history_sid"],
                                "effective_history_tokens": [t for t, m in zip(record["input_ids"], record["attention_mask"]) if m],
                                "input_length": len(record["input_ids"]), "target_sid": record["target_sid"],
                                "target_tokens": record["target_tokens"], "depths": []}
                        for depth in (1, 2, 3):
                            mined, stats = mine_depth(record, shard, row, depth, catalog, hard_limit)
                            hard_rows.extend(mined)
                            totals[depth].update(stats)
                            counts[depth][len(mined)] += 1
                            totals[depth]["groups"] += 1
                            totals[depth]["covered_groups"] += bool(mined)
                            totals[depth]["beam_covered_groups"] += any(r["source"] == "beam" for r in mined)
                            for negative in mined:
                                totals[depth]["rows"] += 1
                                totals[depth][f"source_{negative['source']}"] += 1
                                totals[depth]["survived_beam_rows"] += negative["survived_beam"]
                                totals[depth]["pruned_beam_rows"] += negative["source"] == "beam" and not negative["survived_beam"]
                                totals[depth]["from_gold_parent_rows"] += negative["from_gold_parent"]
                                totals[depth]["score_above_gold_rows"] += negative["score"] > negative["gold_score"]
                            if first_example is None:
                                d = depth - 1
                                surviving = []
                                for p, token, score in zip(shard["selected_parent_indices"][row, d], shard["selected_token_ids"][row, d], shard["selected_cumulative_scores"][row, d]):
                                    prefix = [*shard["parent_prefix_tokens_with_start"][row, d, p, 1:depth].tolist(), int(token)]
                                    reason = "positive" if prefix == record["target_tokens"][:depth] else (
                                        "wrong_token_position" if not position_valid(prefix) else (
                                            "absent_from_sid_catalog" if tuple(prefix) not in catalog.prefix_sets[depth] else (
                                                "retained_in_hard_pool" if any(r["negative_tokens"] == prefix for r in mined) else "below_pool_limit")))
                                    surviving.append({"tokens": prefix, "score": float(score), "catalog_legal": tuple(prefix) in catalog.prefix_sets[depth], "decision": reason})
                                case["depths"].append({"depth": depth, "positive": record["target_tokens"][:depth],
                                                       "gold_score": float(shard["gold_cumulative_logprobs"][row, d]),
                                                       "gold_token_rank": int(shard["gold_token_ranks"][row, d]),
                                                       "actual_survivors": surviving, "hard_pool": mined})
                        if first_example is None:
                            first_example = case
                        processed += 1
                        seen_rows += 1
                require(seen_rows == n or (max_samples is not None and processed == max_samples), "JSON/NPZ row count differs")
                train_writer.write_table(pa.Table.from_pylist(train_rows, schema=TRAIN_SCHEMA))
                hard_writer.write_table(pa.Table.from_pylist(hard_rows, schema=HARD_SCHEMA))
                used_shards.append({"npz": npz_relative, "source_rows": n, "processed_rows": seen_rows})
                del shard, train_rows, hard_rows
                state.update(processed_samples=processed, processed_shards=len(used_shards),
                             hard_rows=sum(totals[d]["rows"] for d in totals), elapsed_seconds=time.monotonic() - started)
                write_json(progress_path, state)
                print(json.dumps({key: state[key] for key in ("processed_samples", "processed_shards", "hard_rows", "elapsed_seconds")}), flush=True)
                require(shutil.disk_usage(data_dir).free > 2 * 1024**3, "Disk reserve below 2 GiB")
        expected = min(max_samples, manifest["counts"]["train"]["source"]) if max_samples else manifest["counts"]["train"]["source"]
        require(processed == expected, "Incomplete Train processing")
        for temporary, final in zip(partials, final_paths):
            temporary.replace(final)
        state.update(state="complete", processed_samples=processed, processed_shards=len(used_shards),
                     catalog_items=item_count, catalog_unique_sids=len(catalog.rows),
                     catalog_prefix_counts={d: len(catalog.prefixes[d]) for d in (1, 2, 3)},
                     input_length_distribution=dict(sorted(input_lengths.items())),
                     depth_statistics={d: dict(totals[d], count_distribution=dict(sorted(counts[d].items())),
                                               coverage=totals[d]["covered_groups"] / processed,
                                               beam_coverage=totals[d]["beam_covered_groups"] / processed) for d in totals},
                     actual_first_shard_tensors=formats, used_shards=used_shards,
                     outputs=[{"path": str(path), "rows": pq.ParquetFile(path).metadata.num_rows,
                               "bytes": path.stat().st_size, "sha256": sha256(path)} for path in final_paths],
                     raw_inventory_sha256=sha256(raw_root / "download_inventory.json"),
                     completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(report_path, state)
        write_json(progress_path, state)
        write_json(report_path.with_name(report_path.stem + "_train_0000000.json"), first_example)
        print(json.dumps({"state": "complete", "train_samples": processed,
                          "hard_rows": sum(totals[d]["rows"] for d in totals), "report": str(report_path)}), flush=True)
        return state
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", processed_samples=processed)
        write_json(progress_path, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=DOCS / "prepare_report.json")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--hard-limit", type=int, default=30)
    a = p.parse_args()
    prepare(a.raw_root, a.output_dir, a.report, a.max_samples, a.hard_limit)


if __name__ == "__main__":
    main()
