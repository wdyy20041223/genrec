"""Audit/export the completed frozen-TIGER inference study; no model execution."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cotiger.core.trainers.tiger_trainer import calculate_pos_index
from .finetune import file_sha256
from .inference_lab import BASE, ROOT, metrics_from_ranks, new_json, ranks_for
from .inference_rerank import load_cache


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bootstrap(candidate, reference, rng):
    result = {}
    for metric in ("Recall@10", "NDCG@10"):
        def contribution(ranks):
            ok = (ranks > 0) & (ranks <= 10)
            return ok.astype(float) if metric == "Recall@10" else np.where(ok, 1 / np.log2(np.maximum(ranks, 1) + 1), 0)
        difference = contribution(candidate) - contribution(reference)
        values, counts = np.unique(difference, return_counts=True)
        draws = rng.multinomial(len(difference), counts / len(difference), size=10000) @ values / len(difference)
        result[metric] = {"delta_pp": float(difference.mean() * 100),
                          "ci95_pp": (np.quantile(draws, [.025, .975]) * 100).tolist()}
    return result


def compare(candidate, baseline):
    hit = (candidate > 0) & (candidate <= 10)
    old = (baseline > 0) & (baseline <= 10)
    a, b = metrics_from_ranks(candidate), metrics_from_ranks(baseline)
    return {"gained": int((hit & ~old).sum()), "lost": int((old & ~hit).sum()),
            "net": int(hit.sum() - old.sum()),
            "absolute_change_pp": {k: (a[k] - b[k]) * 100 for k in a},
            "relative_change_percent": {k: (a[k] / b[k] - 1) * 100 for k in a}}


def provenance(experiment):
    saved = read_json(experiment / "selected_sources.json")
    assert all(file_sha256(Path(path)) == expected for path, expected in saved.items())
    processed_path = ROOT / "out/tiger_prefix_ft/beauty_20261013/data/train_samples.parquet"
    processed = pq.read_table(processed_path, columns=["dataset_index", "tensor_shard", "input_ids", "target_tokens"])
    original = pq.read_table(BASE / "analysis/train_samples.parquet", columns=["sample_index", "target_sid", "history_item_indices"])
    assert len(processed) == len(original) == 131413
    assert processed["dataset_index"].to_pylist() == original["sample_index"].to_pylist()
    assert all(value.startswith("train/") for value in processed["tensor_shard"].to_pylist())
    target = np.array(original["target_sid"].to_pylist()) + np.array([1, 257, 513])
    assert np.array_equal(target, np.array(processed["target_tokens"].to_pylist()))
    last_items = []
    for chunk in original["history_item_indices"].chunks:
        last_items.extend(chunk.values.to_numpy()[chunk.offsets.to_numpy()[1:] - 1].tolist())
    actual = []
    for chunk in processed["input_ids"].chunks:
        actual.append(chunk.values.to_numpy()[chunk.offsets.to_numpy()[1:, None] - np.array([3, 2, 1])])
    sids = np.load(BASE / "analysis/item_semantic_ids.npy", allow_pickle=False)
    assert np.array_equal(np.concatenate(actual), sids[np.array(last_items)] + np.array([1, 257, 513]))
    source_split_check = read_json(ROOT / "out/cloud_reproduction/beauty_tiger_raw_20261013/full/metadata/sid_mapping_check.json")
    assert source_split_check["state"] == "passed" and source_split_check["split_samples_checked"]["train"] == 131413
    model_ast = ast.parse((ROOT / "cotiger/core/models/tiger.py").read_text())
    generate = next(node for node in ast.walk(model_ast) if isinstance(node, ast.FunctionDef) and node.name == "generate")
    assert ast.literal_eval(generate.args.defaults[1]) == 15
    infer_ast = ast.parse((ROOT / "tiger_prefix_ft/infer.py").read_text())
    beam_option = next(node for node in ast.walk(infer_ast) if isinstance(node, ast.Call)
                       and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--beam-sizes")
    assert ast.literal_eval(next(key.value for key in beam_option.keywords if key.arg == "default")) == [15, 15, 15]
    return {"frozen_sources_unchanged": True, "train_rows": len(processed),
            "all_train_targets_and_contexts_match_original_train_export": True,
            "all_tensor_references_are_train": True, "original_leave_one_out_check": source_split_check["split_samples_checked"],
            "production_defaults_still_15_15_15": True,
            "train_statistics_source_sha256": file_sha256(processed_path)}


def official_check(cache):
    k = cache["meta"]["widths"][-1]
    predictions = torch.from_numpy(cache["tokens"][:, :k].astype(np.int64))
    labels = torch.from_numpy(cache["targets"].astype(np.int64))
    pos = calculate_pos_index(predictions, labels, k)
    discount = 1 / torch.log2(torch.arange(2, k + 2).float())
    values = {}
    for cutoff in (5, 10):
        values[f"Recall@{cutoff}"] = float(pos[:, :cutoff].sum()) / len(labels)
        values[f"NDCG@{cutoff}"] = float((pos[:, :cutoff] * discount[:cutoff]).double().sum()) / len(labels)
    assert all(abs(values[key] - cache["meta"]["metrics"][key]) < 1e-7 for key in values)
    return values


def run(args):
    experiment, output = args.experiment, args.output
    if output.exists():
        raise FileExistsError(output)
    lock = read_json(experiment / "selection_lock.json")
    assert lock["selection_split"] == "valid" and not lock["test_used_for_selection"]
    checks = provenance(experiment)
    output.mkdir(parents=True, exist_ok=False)
    old_path = ROOT / "out/tiger_prefix_ft/beauty_beam_30_30_30_eval_20261015/beam_evaluation_samples.parquet"
    rng = np.random.default_rng(20261016)
    summary = {"state": "complete", "selection_lock": lock, "provenance": checks,
               "production_benchmark": read_json(experiment / "production_benchmark_valid/summary.json"),
               "splits": {}, "no_tuning_after_test": True}
    samples, examples = [], []
    for split in ("valid", "test"):
        caches = {}
        for name, folder in (("original30", "raw30_pool256"), ("prior15", "prior15_h3_d2_04_d3_04" if split == "valid" else "prior15"),
                             ("prior30", "prior_h3_d2_04_d3_04" if split == "valid" else "prior30")):
            c = load_cache(experiment / split / folder)
            assert c["meta"]["state"] == "complete" and c["meta"]["samples"] == 22363
            assert c["meta"]["checkpoint_sha256"] == lock["checkpoint_sha256"]
            if name != "original30":
                selected = next(item for item in lock["selected"] if item["name"] == name)
                for key in ("widths", "prior_strengths", "prior_history", "suppress_seen_singletons"):
                    assert c["meta"][key] == selected[key]
                assert c["meta"]["prior_train_stats_sha256"] == lock["train_statistics_sha256"]
            if split == "test":
                assert Path(c["source_cache"]).stat().st_ctime >= datetime.fromisoformat(lock["locked_at_utc"]).timestamp()
            official_check(c)
            assert np.array_equal(ranks_for(c["tokens"][:, :c["meta"]["widths"][-1]], c["targets"]), c["depth_ranks"][:, 2])
            caches[name] = c
        base = caches["original30"]
        old_rows = pq.read_table(old_path, filters=[("split", "=", split)]).to_pylist()
        baseline, baseline_rows = {}, {}
        for model, name in (("original_free", "original15"), ("beam_30_30_30", "original30")):
            by_id = {row["sample_id"]: row for row in old_rows if row["model_name"] == model}
            baseline[name] = np.array([by_id[str(sid)]["gold_rank_depth3"] or 0 for sid in base["sample_ids"]])
            baseline_rows[name] = [by_id[str(sid)] for sid in base["sample_ids"]]
            assert all(by_id[str(sid)]["target_tokens"] == target.tolist() for sid, target in zip(base["sample_ids"], base["targets"]))
        assert np.array_equal(base["depth_ranks"][:, 2], baseline["original30"])
        result = {"samples": len(base["targets"]), "models": {}}
        summary["splits"][split] = result
        result["models"]["original15"] = {"metrics": metrics_from_ranks(baseline["original15"]),
            "survival_counts": [sum(row[f"survived_depth{d}"] for row in baseline_rows["original15"]) for d in (1, 2, 3)],
            "final_beam_size": 15,
            "first_prune_counts": {str(d): sum(row["first_prune_depth"] == d for row in baseline_rows["original15"]) for d in (0, 1, 2, 3)},
            "source": "archived same-environment baseline; reverified on full Validation in production benchmark"}
        for name, cache in caches.items():
            rank = cache["depth_ranks"][:, 2]
            item = {"metrics": official_check(cache), "survival_counts": cache["meta"]["survival_counts"],
                    "final_beam_size": cache["meta"]["widths"][-1],
                    "first_prune_counts": {str(d): 0 for d in (0, 1, 2, 3)},
                    "target_top15_count": int(((rank > 0) & (rank <= 15)).sum()), "comparisons": {},
                    "collection_seconds": cache["meta"]["seconds"], "peak_gpu_mib": cache["meta"]["peak_gpu_mib"]}
            for other, old in baseline.items():
                item["comparisons"][other] = compare(rank, old)
            if name != "original30":
                item["paired_bootstrap_vs_original30"] = bootstrap(rank, baseline["original30"], rng)
            result["models"][name] = item
            alive = cache["depth_ranks"] > 0
            first = np.where(alive.all(1), 0, (~alive).argmax(1) + 1)
            for d in (0, 1, 2, 3):
                item["first_prune_counts"][str(d)] = int((first == d).sum())
            if name == "original30":
                continue
            before = (baseline["original30"] > 0) & (baseline["original30"] <= 10)
            after = (rank > 0) & (rank <= 10)
            lost = before & ~after
            item["lost_top10_fates"] = {"d1_pruned": int((lost & ~alive[:, 0]).sum()),
                "d2_pruned": int((lost & alive[:, 0] & ~alive[:, 1]).sum()),
                "d3_pruned": int((lost & alive[:, 1] & ~alive[:, 2]).sum()),
                "rank11_15": int((lost & (rank > 10) & (rank <= 15)).sum()),
                "rank16_final": int((lost & (rank > 15)).sum())}
            assert sum(item["lost_top10_fates"].values()) == int(lost.sum())
            for i, sid in enumerate(cache["sample_ids"]):
                samples.append({"sample_id": str(sid), "split": split, "model_name": name,
                    "target_tokens": cache["targets"][i].tolist(), "original15_rank": int(baseline["original15"][i]) or None,
                    "original30_rank": int(baseline["original30"][i]) or None, "final_rank": int(rank[i]) or None,
                    "gold_rank_depth1": int(cache["depth_ranks"][i, 0]) or None,
                    "gold_rank_depth2": int(cache["depth_ranks"][i, 1]) or None,
                    "survived_depth1": bool(alive[i, 0]), "survived_depth2": bool(alive[i, 1]), "survived_depth3": bool(alive[i, 2]),
                    "first_prune_depth": int(first[i]), "hit_at_10": bool(after[i]),
                    "gained_vs_original30": bool(after[i] and not before[i]), "lost_vs_original30": bool(lost[i]),
                    "predicted_top10": cache["tokens"][i, :10].tolist()})
            if split == "valid" and name == "prior30":
                for label, mask in (("rescued_d2", after & (base["depth_ranks"][:, 1] == 0)),
                                    ("improved_final_rank", after & (baseline["original30"] > 10)), ("lost_hit", lost)):
                    i = int(np.flatnonzero(mask)[0])
                    def predictions(source):
                        return [{"tokens": source["tokens"][i, j].tolist(), "raw_score": float(source["raw_scores"][i, j]),
                                 "search_score": float(source["search_scores"][i, j]), "logps": source["logps"][i, j].tolist()} for j in range(10)]
                    examples.append({"kind": label, "sample_id": str(cache["sample_ids"][i]), "target_tokens": cache["targets"][i].tolist(),
                        "recent_history_tokens": cache["histories"][i, -3:].tolist(),
                        "original30_ranks": base["depth_ranks"][i].tolist(), "prior30_ranks": cache["depth_ranks"][i].tolist(),
                        "original30_top10": predictions(base), "prior30_top10": predictions(cache)})
    trial_records = [read_json(path) for path in sorted((experiment / "offline_valid").glob("*.json"))]
    new_json(output / "all_validation_experiments.json", trial_records)
    new_json(output / "study_summary.json", summary)
    new_json(output / "real_examples.json", examples)
    pq.write_table(pa.Table.from_pylist(samples), output / "selected_samples.parquet", compression="zstd")
    assert pq.read_metadata(output / "selected_samples.parquet").num_rows == 89452
    assert provenance(experiment)["frozen_sources_unchanged"]
    print(json.dumps({"state": "passed", "sample_rows": len(samples), "validation_settings": len(trial_records), "splits": summary["splits"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
