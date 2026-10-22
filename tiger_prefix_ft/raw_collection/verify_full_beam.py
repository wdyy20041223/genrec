"""Read saved raw Beauty shards and verify reconstruction without loading T5."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from .raw_utils import D, K, OFFSETS, add_readable_paths, read_json, require, sha256, verify_tensors, write_json


def verify(output, limit_shards=None, cuda_topk=False):
    manifest = read_json(output / "collection_manifest.json")
    require(manifest["state"] == "complete", "Collection is not complete")
    sids = np.load(output / "metadata/item_semantic_ids.npy", allow_pickle=False)
    ids, counts, results = set(), {}, []
    shards = manifest["shards"] if limit_shards is None else manifest["shards"][:limit_shards]
    for entry in shards:
        split = entry["split"]
        for info in entry["files"]:
            path = output / info["path"]
            require(path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"], "Shard hash/size mismatch")
        tensor = next(info["path"] for info in entry["files"] if info["path"].endswith(".npz"))
        text = next(info["path"] for info in entry["files"] if info["path"].endswith(".jsonl.gz"))
        with np.load(output / tensor, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        result = verify_tensors(data)
        if cuda_topk:
            import torch
            require(torch.cuda.is_available(), "CUDA top-k verification requested but unavailable")
            for step in range(D):
                scores = data["next_token_logprobs"][:, step] + data["beam_scores_before"][:, step, :, None]
                values, indices = torch.topk(torch.from_numpy(scores.reshape(len(scores), -1)).to("cuda"), K,
                                            dim=-1, largest=True, sorted=True)
                require(np.array_equal(indices.cpu().numpy(), data["selected_flat_indices"][:, step]),
                        "Saved-matrix CUDA top-k indices differ from actual search")
                require(np.array_equal(values.cpu().numpy(), data["selected_cumulative_scores"][:, step]),
                        "Saved-matrix CUDA top-k scores differ from actual search")
                del values, indices
        with gzip.open(output / text, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        require(len(rows) == entry["samples"] == len(data["dataset_indices"]), "Metadata/tensor count mismatch")
        require(np.array_equal(data["dataset_indices"], np.arange(entry["start_row"], entry["end_row_exclusive"])), "Dataset index gap/overlap")
        for index, row in enumerate(rows):
            require(row["sample_id"] not in ids, "Duplicate stable sample ID")
            ids.add(row["sample_id"])
            require(row["sample_id"] == f"{split}_{row['dataset_index']:07d}" and row["split"] == split,
                    "Wrong stable sample ID/split")
            require(row["dataset_index"] == int(data["dataset_indices"][index]), "Cross-sample tensor index mismatch")
            require(row["tensor_shard"] == tensor and row["tensor_row"] == index, "Wrong tensor pointer")
            require(row["target_sid"] == sids[row["target_item_id"]].tolist(), "Target SID map mismatch")
            require(row["history_sid"] == sids[row["history_item_ids"]].tolist(), "History SID map mismatch")
            require(row["target_tokens"] == (np.array(row["target_sid"]) + OFFSETS).tolist(), "Target offset mismatch")
            history = (np.array(row["history_sid"]) + OFFSETS).reshape(-1).tolist()
            valid = [v for v, m in zip(row["input_ids"], row["attention_mask"]) if m]
            require(history == valid and len(valid) == row["history_length"], "Actual encoder history differs")
            rebuilt = dict(row)
            # Rebuild readable decisions and finals exclusively from saved tensors.
            add_readable_paths([rebuilt], {key: value[index:index + 1] for key, value in data.items()})
            for key in ("steps", "final_beams", "gold_teacher_forced"):
                require(rebuilt[key] == row[key], f"JSON/tensor mismatch: {key}")
        counts[split] = counts.get(split, 0) + len(rows)
        results.append(result)
        print(json.dumps({"verified_shard": tensor, "samples": len(rows), "score_abs_error": result["max_score_abs_error"]}), flush=True)
    if limit_shards is None:
        for split, count in manifest["counts"].items():
            require(counts[split] == count["requested"] == count["success"], "Incomplete split coverage")
            with gzip.open(output / split / "samples.jsonl.gz", "rt", encoding="utf-8") as f:
                combined = [json.loads(line)["sample_id"] for line in f]
            expected = [f"{split}_{i:07d}" for i in range(counts[split])]
            require(combined == expected, "Combined gzip metadata missing/duplicated/reordered rows")
    return {"state": "passed", "loaded_t5": False, "verified_samples": len(ids), "counts": counts,
            "verified_shards": len(results), "complete_run_checked": limit_shards is None,
            "max_reconstructed_score_abs_error": max(x["max_score_abs_error"] for x in results),
            "max_probability_sum_error": max(x["max_probability_sum_error"] for x in results),
            "steps_with_score_ties": sum(x["steps_with_score_ties"] for x in results),
            "steps_cpu_stable_tie_order_differs": sum(x["steps_cpu_stable_tie_order_differs"] for x in results),
            "tie_rule": "Recorded original CUDA top-k indices resolve exact score ties; full top-k score multiset independently reconstructed",
            "cuda_topk_original_indices_exact": True if cuda_topk else None,
            "final_tokens_exact": True, "all_parent_paths_exact": True, "offset_history_and_tensor_pointers_exact": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--limit-shards", type=int)
    p.add_argument("--cuda-topk", action="store_true", help="Independently repeat original torch.topk on saved matrices, without T5")
    p.add_argument("--report", type=Path, required=True, help="Write outside immutable raw dataset")
    a = p.parse_args()
    output = a.output_dir.resolve()
    require(not a.report.resolve().is_relative_to(output), "Do not modify finalized raw dataset during verification")
    report = verify(output, a.limit_shards, a.cuda_topk)
    write_json(a.report, report)
    print(json.dumps(report), flush=True)
