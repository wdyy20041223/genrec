"""Check every prepared Train/Hard row against its original user/depth and score."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .negative_sampler import NegativeSampler, position_valid, shared_length
from .prepare_data import (DEFAULT_OUT, DEFAULT_RAW, DOCS, TRAIN_SCHEMA, catalog_from_mapping,
                           load_shard, require, sha256, validate_record, write_json)

PROFILES = [(2, 2, 4), (4, 2, 2), (0, 0, 8)]


def expected_pool(record, data, row, depth, catalog, limit):
    """Independent score reconstruction, without using prepare_data.mine_depth."""
    positive = tuple(record["target_tokens"][:depth])
    candidates = {}
    layer = depth - 1
    for parent in range(15):
        if not data["active_beam_mask"][row, layer, parent]:
            continue
        prefix = tuple(int(t) for t in data["parent_prefix_tokens_with_start"][row, layer, parent, 1:depth])
        for next_token in catalog.children.get(prefix, ()):
            negative = prefix + (int(next_token),)
            if negative == positive:
                continue
            score = float(np.float32(np.float32(data["beam_scores_before"][row, layer, parent])
                                      + np.float32(data["next_token_logprobs"][row, layer, parent, next_token])))
            previous = candidates.get(negative)
            if previous is None or score > previous[0]:
                candidates[negative] = (score, "beam")
    gold_parent = tuple(record["target_tokens"][:depth - 1])
    base = np.float32(0) if depth == 1 else np.float32(data["gold_cumulative_logprobs"][row, depth - 2])
    for next_token in catalog.children.get(gold_parent, ()):
        negative = gold_parent + (int(next_token),)
        if negative != positive and negative not in candidates:
            score = float(np.float32(base + np.float32(data["gold_next_token_logprobs"][row, layer, next_token])))
            candidates[negative] = (score, "gold_conditional")
    return sorted(candidates.items(), key=lambda entry: (-entry[1][0], entry[0]))[:limit]


def validate_sampled(group, sampler):
    sample = sampler.get_sample(group["sample_id"])
    target = tuple(sample["target_tokens"])
    depth = group["depth"]
    require(group["input_ids"] == sample["input_ids"] and group["attention_mask"] == sample["attention_mask"], "Sampler changed history")
    require(group["ce_labels"] == list(target) and group["positive"] == list(target[:depth]), "Sampler target/depth mismatch")
    seen = {target[:depth]}
    hard_rows = sampler.hard_for(group["sample_id"], depth)
    eligible_hard = {tuple(r["negative_tokens"]): r for r in hard_rows
                     if sampler.include_gold_conditional or r["source"] == "beam"}
    counts = Counter()
    selected_hard = []
    for negative in group["negatives"]:
        prefix = tuple(negative["tokens"])
        kind = negative["type"]
        require(len(prefix) == depth and position_valid(prefix), "Sampled prefix depth/Offset mismatch")
        require(prefix in sampler.catalog.prefix_sets[depth] and prefix not in seen, "Illegal/duplicate/positive sampled prefix")
        require(kind in ("easy", "medium", "hard"), "Missing Negative Type")
        seen.add(prefix)
        counts[kind] += 1
        if kind == "easy":
            require(prefix[0] != target[0], "Easy shared first Token")
        elif kind == "medium":
            require(depth > 1 and 1 <= shared_length(prefix, target) < depth, "Invalid Medium")
        else:
            require(prefix in eligible_hard, "Hard came from a different sample/depth/source")
            require(negative["cached_score"] == eligible_hard[prefix]["score"], "Sampled Hard score changed")
            selected_hard.append(prefix)
    requested = group["requested_counts"]
    expected_hard = min(requested["hard"], len(eligible_hard))
    medium_available = 0 if depth == 1 else len(sampler.catalog.by_first[depth].get(target[0], ())) - 1
    medium_available -= sum(prefix[0] == target[0] for prefix in selected_hard) if depth > 1 else 0
    easy_available = len(sampler.catalog.prefixes[depth]) - len(sampler.catalog.by_first[depth].get(target[0], ()))
    easy_available -= sum(prefix[0] != target[0] for prefix in selected_hard)
    expected = {"hard": expected_hard, "medium": min(requested["medium"], medium_available),
                "easy": min(requested["easy"], easy_available)}
    require(all(counts[kind] == expected[kind] == group["actual_counts"][kind] for kind in expected), "Sampler count/shortfall mismatch")
    return counts


def check(raw_root, output_dir, prepare_report, report_path, sampler_samples=100):
    started = time.monotonic()
    require(sampler_samples > 0, "sampler-samples must be positive")
    raw_root, output_dir, prepare_report, report_path = map(Path, (raw_root, output_dir, prepare_report, report_path))
    prepared = json.loads(prepare_report.read_text(encoding="utf-8"))
    require(prepared["state"] == "complete", "Preparation is incomplete")
    require(raw_root.resolve() == Path(prepared["raw_root"]).resolve() and output_dir.resolve() == Path(prepared["output_dir"]).resolve(),
            "Report/data locations do not correspond")
    require(sha256(raw_root / "download_inventory.json") == prepared["raw_inventory_sha256"], "Raw inventory identity changed")
    data_dir = output_dir / "data"
    for output in prepared["outputs"]:
        path = Path(output["path"])
        require(path.resolve().parent == data_dir.resolve(), "Unexpected output location")
        require(path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"], "Prepared file hash changed")
    catalog, item_sids, item_count = catalog_from_mapping(raw_root)
    stored_catalog = pq.read_table(data_dir / "sid_catalog.parquet").to_pylist()
    require(stored_catalog == catalog.rows, "Stored catalog differs from original item mapping")
    train_file = pq.ParquetFile(data_dir / "train_samples.parquet")
    hard_file = pq.ParquetFile(data_dir / "hard_negatives.parquet")
    require(train_file.num_row_groups == hard_file.num_row_groups == len(prepared["used_shards"]), "Output row groups do not match raw shards")
    samples = 0
    negatives = 0
    ids = set()
    depth_stats = {d: Counter() for d in (1, 2, 3)}
    empty_beam_examples = []
    gold_conditional_contexts = []
    progress_path = report_path.with_name(report_path.stem + "_progress.json")
    for group_index, spec in enumerate(prepared["used_shards"]):
        shard = load_shard(raw_root / spec["npz"])
        original_json = raw_root / Path(spec["npz"]).with_suffix(".jsonl.gz")
        train_rows = train_file.read_row_group(group_index).to_pylist()
        hard_rows = hard_file.read_row_group(group_index).to_pylist()
        require(len(train_rows) == spec["processed_rows"], "Prepared Train group lost/duplicated rows")
        by_key = {}
        for negative in hard_rows:
            key = (negative["sample_id"], negative["depth"])
            by_key.setdefault(key, []).append(negative)
        train_group_ids = {r["sample_id"] for r in train_rows}
        require(all(r["sample_id"] in train_group_ids for r in hard_rows), "Hard has no corresponding Train context")
        with gzip.open(original_json, "rt", encoding="utf-8") as f:
            for row, line in enumerate(f):
                if row >= len(train_rows):
                    break
                record = json.loads(line)
                validate_record(record, shard, row, spec["npz"], item_sids)
                train = train_rows[row]
                require(all(train[name] == record[name] for name in TRAIN_SCHEMA.names), "CE/history/locator differs from original JSON")
                require(train["sample_id"] not in ids and train["dataset_index"] == samples, "Duplicate/missing Train sample")
                ids.add(train["sample_id"])
                target = tuple(train["target_tokens"])
                for depth in (1, 2, 3):
                    layer = depth - 1
                    rows = by_key.get((train["sample_id"], depth), [])
                    expected = expected_pool(record, shard, row, depth, catalog, prepared["hard_limit"])
                    require([tuple(r["negative_tokens"]) for r in rows] == [p for p, _ in expected], "Pool is not the complete legal top candidates")
                    depth_stats[depth]["groups"] += 1
                    depth_stats[depth]["covered_groups"] += bool(rows)
                    depth_stats[depth]["beam_covered_groups"] += any(r["source"] == "beam" for r in rows)
                    seen = set()
                    selected_pairs = set(zip(shard["selected_parent_indices"][row, layer].tolist(), shard["selected_token_ids"][row, layer].tolist()))
                    for negative, (_, (expected_score, expected_source)) in zip(rows, expected):
                        prefix = tuple(negative["negative_tokens"])
                        require(negative["dataset_index"] == record["dataset_index"], "Hard/NPZ dataset index mismatch")
                        require(len(prefix) == depth and position_valid(prefix) and prefix in catalog.prefix_sets[depth], "Illegal Hard prefix/depth")
                        require(prefix != target[:depth] and prefix not in seen, "Positive/duplicate Hard")
                        seen.add(prefix)
                        require(negative["score"] == expected_score and negative["source"] == expected_source, "Hard score/source does not match raw expansion")
                        require(negative["gold_score"] == float(shard["gold_cumulative_logprobs"][row, layer]), "Gold score has wrong context/depth")
                        require(negative["next_token_id"] == prefix[-1], "Hard Token provenance mismatch")
                        if negative["source"] == "beam":
                            parent = negative["parent_beam_index"]
                            require(0 <= parent < 15 and shard["active_beam_mask"][row, layer, parent], "Hard uses inactive/absent Beam parent")
                            require(prefix[:-1] == tuple(shard["parent_prefix_tokens_with_start"][row, layer, parent, 1:depth]), "Hard parent prefix belongs to a different path")
                            actual_score = float(np.float32(shard["beam_scores_before"][row, layer, parent] + shard["next_token_logprobs"][row, layer, parent, prefix[-1]]))
                            survived = (parent, prefix[-1]) in selected_pairs
                        else:
                            require(negative["source"] == "gold_conditional" and negative["parent_beam_index"] == -1,
                                    "Gold source/provenance mismatch")
                            require(prefix[:-1] == target[:depth - 1], "Gold Conditional uses wrong parent")
                            require(not any(shard["active_beam_mask"][row, layer, p] and tuple(shard["parent_prefix_tokens_with_start"][row, layer, p, 1:depth]) == prefix[:-1] for p in range(15)),
                                    "Beam/Gold overlap did not prefer Beam")
                            base = np.float32(0) if depth == 1 else shard["gold_cumulative_logprobs"][row, depth - 2]
                            actual_score = float(np.float32(base + shard["gold_next_token_logprobs"][row, layer, prefix[-1]]))
                            survived = False
                        require(negative["score"] == actual_score and np.isfinite(actual_score), "Non-finite/wrong original score")
                        require(negative["survived_beam"] == survived, "Survival flag does not match actual selected indices")
                        require(negative["from_gold_parent"] == (prefix[:-1] == target[:depth - 1]), "Gold-parent flag is incorrect")
                        require(negative["shared_prefix_length"] == shared_length(prefix, target), "Shared-prefix length is incorrect")
                        depth_stats[depth]["rows"] += 1
                        depth_stats[depth][f"source_{negative['source']}"] += 1
                        negatives += 1
                    if not any(r["source"] == "beam" for r in rows) and len(empty_beam_examples) < 5:
                        empty_beam_examples.append({"sample_id": train["sample_id"], "depth": depth, "pool_count": len(rows)})
                    if any(r["source"] == "gold_conditional" for r in rows) and len(gold_conditional_contexts) < 5:
                        gold_conditional_contexts.append({"sample_id": train["sample_id"], "depth": depth})
                samples += 1
        del shard, train_rows, hard_rows, by_key
        progress = {"state": "running", "verified_samples": samples, "verified_hard_rows": negatives,
                    "verified_shards": group_index + 1, "elapsed_seconds": time.monotonic() - started}
        write_json(progress_path, progress)
        print(json.dumps(progress), flush=True)
    require(samples == train_file.metadata.num_rows == prepared["processed_samples"], "Incomplete CE verification")
    require(negatives == hard_file.metadata.num_rows, "Some Hard rows were not checked")
    sampler = NegativeSampler(data_dir, seed=42)
    checked_groups = 0
    sampled_counts = Counter()
    sample_ids = list(sampler.sample_lookup)[:sampler_samples]
    for sample_id in sample_ids:
        for depth in (1, 2, 3):
            for profile in PROFILES:
                group = sampler.sample(sample_id, depth, *profile)
                sampled_counts.update(validate_sampled(group, sampler))
                checked_groups += 1
    for context in empty_beam_examples:
        group = sampler.sample(context["sample_id"], context["depth"], 0, 0, 8)
        validate_sampled(group, sampler)
        require(group["actual_counts"]["hard"] == 0, "No-Hard context was filled with invented negatives")
    sampler.include_gold_conditional = True
    for context in gold_conditional_contexts:
        group = sampler.sample(context["sample_id"], context["depth"], 0, 0, prepared["hard_limit"])
        validate_sampled(group, sampler)
        require(any(n["source"] == "gold_conditional" for n in group["negatives"]), "Explicit Gold Conditional option lost its source")
    sampler.include_gold_conditional = False
    example_sampler = sampler
    example_sampler.rng.seed(42)
    first_id = sample_ids[0]
    examples = [example_sampler.sample(first_id, 2, *profile) for profile in PROFILES]
    examples.append(example_sampler.sample(sample_ids[min(1, len(sample_ids) - 1)], 1, 2, 2, 4))
    examples.append(example_sampler.sample(sample_ids[min(2, len(sample_ids) - 1)], 3, 2, 2, 4))
    examples_path = report_path.with_name(report_path.stem + "_sampling_examples.json")
    write_json(examples_path, {"seed": 42, "profiles": PROFILES, "groups": examples,
                               "empty_beam_contexts": empty_beam_examples})
    result = {"state": "passed", "scope": "all prepared Train and Hard rows",
              "verified_samples": samples, "verified_hard_rows": negatives, "verified_shards": len(prepared["used_shards"]),
              "catalog_items": item_count, "catalog_unique_sids": len(catalog.rows),
              "depth_statistics": {d: dict(depth_stats[d]) for d in depth_stats},
              "max_original_score_abs_error": 0.0, "all_legal_top_candidate_sets_exact": True,
              "ce_history_targets_and_locators_exact": True, "no_valid_test_targets_read": True,
              "checked_sampler_groups": checked_groups, "sampler_profiles": PROFILES,
              "sampler_negative_counts": dict(sampled_counts), "empty_beam_contexts": empty_beam_examples,
              "checked_optional_gold_conditional_groups": len(gold_conditional_contexts),
              "sampling_examples": str(examples_path), "no_model_forward_or_training": True,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started}
    write_json(report_path, result)
    write_json(progress_path, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--prepare-report", type=Path, default=DOCS / "prepare_report.json")
    p.add_argument("--report", type=Path, default=DOCS / "check_report.json")
    p.add_argument("--sampler-samples", type=int, default=100)
    a = p.parse_args()
    check(a.raw_root, a.output_dir, a.prepare_report, a.report, a.sampler_samples)


if __name__ == "__main__":
    main()
