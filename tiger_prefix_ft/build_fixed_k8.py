"""Materialize three fixed-K=8 Beauty datasets from the existing Parquet files."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .negative_sampler import DEFAULT_DATA, NegativeSampler, position_valid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs/generated/tiger_prefix_ft_beauty_20261013/fixed_k8_report.json"
# H4 first preserves its previously demonstrated seed-42 sample at dataset row 0.
PROFILES = {"h4": (2, 2, 4), "h2": (3, 3, 2), "h6": (1, 1, 6)}
TYPE_ORDER = {"easy": 0, "medium": 1, "hard": 2}
SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("input_ids", pa.list_(pa.int16())),
    ("attention_mask", pa.list_(pa.int8())),
    ("ce_labels", pa.list_(pa.int16(), 3)),
    ("rank_depth", pa.int8()),
    ("positive_tokens", pa.list_(pa.int16())),
    ("negative_tokens", pa.list_(pa.list_(pa.int16()), 8)),
    ("negative_types", pa.list_(pa.string(), 8)),
    ("easy_count", pa.int8()),
    ("medium_count", pa.int8()),
    ("hard_count", pa.int8()),
])


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def assign_depths(size, seed):
    # One extra Depth-2 row, and the documented first sample remains at Depth 2.
    depths = np.resize(np.array([2, 1, 3], dtype=np.int8), size)
    np.random.default_rng(seed).shuffle(depths[1:])
    return depths


def requested_counts(profile, depth):
    easy, medium, hard = PROFILES[profile]
    return (8 - hard, 0, hard) if depth == 1 else (easy, medium, hard)


def check_record(record, original, depth, catalog, beam_prefixes):
    assert record["sample_id"] == original["sample_id"]
    assert record["input_ids"] == original["input_ids"]
    assert record["attention_mask"] == original["attention_mask"]
    assert record["ce_labels"] == original["target_tokens"]
    assert record["rank_depth"] == depth
    positive = tuple(original["target_tokens"][:depth])
    assert tuple(record["positive_tokens"]) == positive
    prefixes = [tuple(p) for p in record["negative_tokens"]]
    kinds = record["negative_types"]
    assert len(prefixes) == len(kinds) == 8 and len(set(prefixes)) == 8
    assert positive not in prefixes
    for prefix, kind in zip(prefixes, kinds):
        assert len(prefix) == depth and position_valid(prefix)
        assert prefix in catalog.prefix_sets[depth]
        if kind == "easy":
            assert prefix[0] != positive[0]
        elif kind == "medium":
            assert depth > 1 and prefix[0] == positive[0]
        else:
            assert kind == "hard" and prefix in beam_prefixes
    counts = Counter(kinds)
    assert all(record[f"{kind}_count"] == counts[kind] for kind in TYPE_ORDER)
    assert [p for p, kind in zip(prefixes, kinds) if kind == "hard"] == beam_prefixes[:counts["hard"]]


def make_record(sampler, original, depth, profile, beam_prefixes):
    easy, medium, hard = requested_counts(profile, depth)
    group = sampler.sample(original["sample_id"], depth, easy, medium, hard)
    shortage = 8 - len(group["negatives"])
    if shortage > 0:
        # Keep truthful types; replace missing candidates with distinct legal Easy.
        group = sampler.sample(original["sample_id"], depth, easy + shortage, medium, hard)
    if len(group["negatives"]) != 8:
        raise ValueError(f"K=8 unavailable: {original['sample_id']}, depth={depth}, profile={profile}, counts={group['actual_counts']}")
    assert group["hard_source_policy"] == "beam_only"
    assert all(n["source"] == "beam" for n in group["negatives"] if n["type"] == "hard")
    negatives = sorted(group["negatives"], key=lambda n: TYPE_ORDER[n["type"]])
    counts = Counter(n["type"] for n in negatives)
    record = {
        "sample_id": group["sample_id"], "input_ids": group["input_ids"],
        "attention_mask": group["attention_mask"], "ce_labels": group["ce_labels"],
        "rank_depth": depth, "positive_tokens": group["positive"],
        "negative_tokens": [n["tokens"] for n in negatives],
        "negative_types": [n["type"] for n in negatives],
        **{f"{kind}_count": counts[kind] for kind in TYPE_ORDER},
    }
    check_record(record, original, depth, sampler.catalog, beam_prefixes)
    replaced = tuple(counts[kind] for kind in TYPE_ORDER) != (easy, medium, hard)
    return record, replaced


def build(data_dir, output_dir, seed, report_path):
    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"train_k8_{name}.parquet" for name in PROFILES}
    partials = {name: path.with_suffix(".parquet.partial") for name, path in paths.items()}
    for path in [*paths.values(), *partials.values()]:
        if path.exists():
            raise FileExistsError(f"Existing output: {path}; use a new --output-dir")
    source_hashes = {name: sha256(data_dir / name) for name in (
        "train_samples.parquet", "sid_catalog.parquet", "hard_negatives.parquet")}
    sampler = NegativeSampler(data_dir, seed=seed)
    size = sampler.train.num_rows
    depths = assign_depths(size, seed)
    stats = {name: {"rows": 0, "depth_counts": Counter(), "type_counts": Counter(),
                    "ratio_replaced_rows": 0, "actual_ratios": Counter(), "replacement_examples": []}
             for name in PROFILES}
    first_examples = {}
    processed = 0
    with ExitStack() as stack:
        writers = {name: stack.enter_context(pq.ParquetWriter(path, SCHEMA, compression="zstd"))
                   for name, path in partials.items()}
        for batch in sampler.train.to_batches(max_chunksize=1024):
            buffers = {name: [] for name in PROFILES}
            for original in batch.to_pylist():
                assert original["dataset_index"] == processed
                assert original["sample_id"] == f"train_{processed:07d}"
                depth = int(depths[processed])
                beam = sorted((r for r in sampler.hard_for(original["sample_id"], depth)
                               if r["source"] == "beam"), key=lambda r: (-r["score"], r["negative_tokens"]))
                beam_prefixes = [tuple(r["negative_tokens"]) for r in beam]
                for name in PROFILES:
                    record, replaced = make_record(sampler, original, depth, name, beam_prefixes)
                    buffers[name].append(record)
                    first_examples.setdefault(name, record)
                    counts = tuple(record[f"{kind}_count"] for kind in TYPE_ORDER)
                    stat = stats[name]
                    stat["rows"] += 1
                    stat["depth_counts"][depth] += 1
                    stat["type_counts"].update(dict(zip(TYPE_ORDER, counts)))
                    stat["actual_ratios"][f"d{depth}_e{counts[0]}_m{counts[1]}_h{counts[2]}"] += 1
                    stat["ratio_replaced_rows"] += int(replaced)
                    if replaced and len(stat["replacement_examples"]) < 5:
                        stat["replacement_examples"].append({"sample_id": record["sample_id"],
                            "depth": depth, "requested": requested_counts(name, depth), "actual": counts})
                processed += 1
            for name, writer in writers.items():
                writer.write_table(pa.Table.from_pylist(buffers[name], schema=SCHEMA))
            if processed % 4096 == 0 or processed == size:
                print(json.dumps({"state": "building", "samples": processed,
                    "ratio_replaced_rows": {name: stats[name]["ratio_replaced_rows"] for name in PROFILES},
                    "elapsed_seconds": round(time.perf_counter() - started, 3)}), flush=True)
    assert processed == size
    assert source_hashes == {name: sha256(data_dir / name) for name in source_hashes}
    for name, path in partials.items():
        assert pq.ParquetFile(path).metadata.num_rows == size
        path.rename(paths[name])
    report = {
        "state": "generated_and_checked_before_write", "seed": seed, "k": 8,
        "data_dir": str(data_dir.resolve()), "train_samples": size,
        "depth_assignment": "balanced shuffled depths; first Train row fixed at depth 2",
        "shared_depth_counts": dict(Counter(int(d) for d in depths)),
        "depth_assignment_sha256": hashlib.sha256(depths.tobytes()).hexdigest(),
        "hard_source": "beam_only", "shortfall_policy": "replace missing candidates with Easy; fail if K=8 remains unavailable",
        "source_sha256": source_hashes, "source_files_unchanged": True,
        "no_valid_test_targets_used": True, "no_training_performed": True,
        "elapsed_seconds": time.perf_counter() - started, "first_sample_examples": first_examples,
        "profiles": {name: {**stats[name], "path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()},
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "generated", "rows_per_file": size,
        "report": str(report_path.resolve()), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    build(args.data_dir, args.output_dir or args.data_dir, args.seed, args.report)


if __name__ == "__main__":
    main()
