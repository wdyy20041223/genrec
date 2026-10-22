"""Label-free candidate scoring and offline evaluation for frozen TIGER."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .inference_lab import BASE, ROOT, metrics_from_ranks, new_json, ranks_for
from .finetune import file_sha256


RADIX = 769
SID_SPACE = RADIX ** 3
PREFIX_SPACE = RADIX ** 2


def sid_keys(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    return (x[..., 0] * RADIX + x[..., 1]) * RADIX + x[..., 2]


def prefix_keys(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    return x[..., 0] * RADIX + x[..., 1]


def lookup(keys, values, query):
    positions = np.searchsorted(keys, query)
    safe = np.minimum(positions, len(keys) - 1)
    return np.where((positions < len(keys)) & (keys[safe] == query), values[safe], 0)


def load_cache(path):
    with np.load(Path(path) / "candidates.npz", allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    with np.load(Path(path) / "contexts.npz", allow_pickle=False) as archive:
        result.update({k: archive[k] for k in archive.files})
    result["meta"] = json.loads((Path(path) / "summary.json").read_text())
    result["source_cache"] = str(Path(path).resolve())
    assert np.allclose(result["logps"].sum(-1), result["raw_scores"], atol=3e-6, rtol=1e-6)
    assert (result["logps"] <= 0).all()
    return result


def train_statistics(path, output):
    """Only train targets and their observed last-history items form counts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    before = file_sha256(path)
    targets, previous = [], []
    columns = ["sample_id", "input_ids", "attention_mask", "target_tokens"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            assert row["sample_id"].startswith("train_")
            assert row["attention_mask"][-3:] == [1, 1, 1]
            targets.append(row["target_tokens"])
            previous.append(row["input_ids"][-3:])
    targets, previous = np.array(targets), np.array(previous)
    assert len(targets) == 131413
    data = {"samples": np.array(len(targets))}
    for name, current, prior, space in (("sid", sid_keys(targets), sid_keys(previous), SID_SPACE),
                                         ("prefix", prefix_keys(targets), prefix_keys(previous), PREFIX_SPACE)):
        for group, items in (("target", current), ("context", prior), ("pair", prior * space + current)):
            keys, counts = np.unique(items, return_counts=True)
            data[f"{name}_{group}_keys"], data[f"{name}_{group}_counts"] = keys, counts.astype(np.int32)
    data["code1_counts"] = np.bincount(targets[:, 0], minlength=RADIX).astype(np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    assert before == file_sha256(path)
    new_json(output.with_suffix(".json"), {"source": str(path), "source_sha256": before,
             "split": "train", "samples": len(targets), "seconds": time.perf_counter() - started,
             "definition": "Train next-SID counts and train last-history-SID -> next-SID transition counts; no optimization/training"})
    return data


def recent_transition_features(cache, stats, last_lift):
    """Two observed older contexts; additive log lift and centered PMI maximum."""
    keys, histories = sid_keys(cache["tokens"]), sid_keys(cache["histories"])
    frequency = lookup(stats["sid_target_keys"], stats["sid_target_counts"], keys)
    marginal = (frequency + 1) / (int(stats["samples"]) + 11401)
    support = lookup(stats["sid_context_keys"], stats["sid_context_counts"], histories[:, -1])
    maximum = last_lift - np.log1p(support[:, None] / 20).astype(np.float32)
    older = np.zeros_like(last_lift)
    for age in (1, 2):
        context = histories[:, -age - 1]
        valid = (context != 0) & (context[:, None] != histories[:, -age:]).all(-1)
        pairs = lookup(stats["sid_pair_keys"], stats["sid_pair_counts"], context[:, None] * SID_SPACE + keys)
        lift = np.log1p(pairs / (20 * marginal)).astype(np.float32)
        older += (.5 ** age) * lift * valid[:, None]
        support = lookup(stats["sid_context_keys"], stats["sid_context_counts"], context)
        centered = lift - np.log1p(support[:, None] / 20).astype(np.float32)
        maximum = np.maximum(maximum, np.where(valid[:, None], (.7 ** age) * centered, -np.inf))
    return {"older_sid_transition": older, "recent_transition_max": maximum}


def make_features(cache, stats):
    """Inputs are candidate IDs, observed histories and train statistics ONLY."""
    tokens, histories = cache["tokens"], cache["histories"]
    keys, hkeys = sid_keys(tokens), sid_keys(histories)
    prefix, hprefix = prefix_keys(tokens), prefix_keys(histories)
    result = {}
    for name, query, contexts, space, size in (("sid", keys, hkeys[:, -1], SID_SPACE, 11401),
                                               ("prefix", prefix, hprefix[:, -1], PREFIX_SPACE, 6243)):
        frequency = lookup(stats[f"{name}_target_keys"], stats[f"{name}_target_counts"], query)
        result[f"{name}_logpop"] = np.log1p(frequency).astype(np.float32)
        marginal = (frequency + 1) / (int(stats["samples"]) + size)
        pair = lookup(stats[f"{name}_pair_keys"], stats[f"{name}_pair_counts"], contexts[:, None] * space + query)
        # log(P(candidate|last)/P(candidate)) with shrinkage tau=20;
        # the omitted context-only normalizer cannot affect candidate ranking.
        result[f"{name}_transition"] = np.log1p(pair / (20 * marginal)).astype(np.float32)
    result["code1_logpop"] = np.log1p(stats["code1_counts"][tokens[:, :, 0]]).astype(np.float32)
    result["seen"] = np.zeros(keys.shape, np.float32)
    result["recent_prefix1"] = np.zeros(keys.shape, np.float32)
    result["recent_prefix2"] = np.zeros(keys.shape, np.float32)
    recency = np.exp(np.arange(-49, 1) / 5).astype(np.float32)
    for start in range(0, len(tokens), 256):
        stop = min(start + 256, len(tokens))
        valid = (hkeys[start:stop] != 0)[:, None, :]
        result["seen"][start:stop] = ((keys[start:stop, :, None] == hkeys[start:stop, None, :]) & valid).any(-1)
        denom = (valid * recency).sum(-1).clip(1e-6)
        result["recent_prefix1"][start:stop] = (((tokens[start:stop, :, 0, None] == histories[start:stop, None, :, 0]) & valid) * recency).sum(-1) / denom
        result["recent_prefix2"][start:stop] = (((prefix[start:stop, :, None] == hprefix[start:stop, None, :]) & valid) * recency).sum(-1) / denom
    catalog = np.load(BASE / "analysis/item_semantic_ids.npy", allow_pickle=False) + np.array([1, 257, 513])
    catalog_keys, group_counts = np.unique(sid_keys(catalog), return_counts=True)
    sizes = lookup(catalog_keys, group_counts, keys)
    result["legal"] = sizes > 0
    # A previously observed item cannot identify which alias in a collided SID
    # group will be recommended. Preserve those groups instead of filtering them.
    result["seen_singleton"] = result["seen"] * (sizes == 1)
    result.update(recent_transition_features(cache, stats, result["sid_transition"]))
    return result


def score_candidates(cache, parameters, features):
    """No evaluation targets or ranks are accessed by this scorer."""
    cap = int(parameters.get("pool", 30))
    assert 10 <= cap <= cache["tokens"].shape[1]
    weights = parameters.get("weights", [1., 1., 1.])
    if parameters.get("search_order", False):
        return cache["search_scores"][:, :cap].copy()
    if weights == [1., 1., 1.]:
        scores = cache["raw_scores"][:, :cap].copy()
    else:
        lp = cache["logps"][:, :cap]
        scores = weights[0] * lp[:, :, 0] + weights[1] * lp[:, :, 1] + weights[2] * lp[:, :, 2]
    for name, coefficient in parameters.get("features", {}).items():
        scores += coefficient * features[name][:, :cap]
    if parameters.get("legal_only", False):
        scores = np.where(features["legal"][:, :cap], scores, -np.inf)
    if "candidate_mask" in cache:
        scores = np.where(cache["candidate_mask"][:, :cap], scores, -np.inf)
    return scores


def rank_candidates(cache, scores, parameters):
    strength = parameters.get("diversity", 0.)
    if not strength:
        return np.argsort(-scores, axis=1, kind="stable")
    # Greedy final-list diversification, with a soft penalty on repeated D1/D2.
    n, cap = scores.shape
    current = scores.copy()
    visits = np.zeros_like(current)
    chosen = []
    rows = np.arange(n)
    groups = cache["tokens"][:, :cap, 0] if parameters.get("diversity_depth", 1) == 1 else prefix_keys(cache["tokens"][:, :cap])
    for _ in range(cap):
        index = current.argmax(1)
        chosen.append(index)
        visits += groups == groups[rows, index][:, None]
        current = scores - strength * np.log1p(visits)
        current[rows[:, None], np.stack(chosen, 1)] = -np.inf
    return np.stack(chosen, 1)


def existing_baseline(split, sample_ids, model):
    path = ROOT / "out/tiger_prefix_ft/beauty_beam_30_30_30_eval_20261015/beam_evaluation_samples.parquet"
    rows = pq.read_table(path, filters=[("split", "=", split), ("model_name", "=", model)]).to_pylist()
    by_id = {row["sample_id"]: row for row in rows}
    return np.array([by_id[sid]["gold_rank_depth3"] or 0 for sid in sample_ids], np.int16)


def fuse_caches(primary, secondary, output):
    """Union real generated candidates; duplicate SIDs keep primary LM values.

    Source-specific calibrated scores are discarded. The final scorer must
    recompute ONE common score from the original LM probabilities and history.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    assert not primary["meta"].get("fusion") and not secondary["meta"].get("fusion"), "Use two generated caches"
    for key in ("sample_ids", "histories", "targets"):
        assert np.array_equal(primary[key], secondary[key]), key
    for key in ("split", "checkpoint_sha256", "sid_sha256", "data_sha256"):
        assert primary["meta"][key] == secondary["meta"][key], key
    started = time.perf_counter()
    tokens = np.concatenate((primary["tokens"], secondary["tokens"]), 1)
    lp = np.concatenate((primary["logps"], secondary["logps"]), 1)
    raw = np.concatenate((primary["raw_scores"], secondary["raw_scores"]), 1)
    keys = sid_keys(tokens)
    mask = np.zeros(raw.shape, bool)
    for row in range(len(tokens)):
        indices = np.sort(np.unique(keys[row], return_index=True)[1])
        count = len(indices)
        tokens[row, :count] = tokens[row, indices]
        lp[row, :count] = lp[row, indices]
        raw[row, :count] = raw[row, indices]
        tokens[row, count:] = -1
        lp[row, count:] = -np.inf
        raw[row, count:] = -np.inf
        mask[row, :count] = True
    parents = (primary["depth_ranks"][:, :2] > 0) | (secondary["depth_ranks"][:, :2] > 0)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "candidates.npz", tokens=tokens, logps=lp, raw_scores=raw,
                        search_scores=raw, depth_ranks=primary["depth_ranks"],
                        candidate_mask=mask, union_parent_survived=parents)
    np.savez_compressed(output / "contexts.npz", sample_ids=primary["sample_ids"],
                        histories=primary["histories"], targets=primary["targets"])
    meta = dict(primary["meta"])
    meta.pop("metrics", None)
    meta.pop("candidate_pool_target_count", None)
    meta.update(fusion=True, source_caches=[primary["source_cache"], secondary["source_cache"]],
                samples=len(tokens), state="candidate_union_only",
                seconds=primary["meta"]["seconds"] + secondary["meta"]["seconds"] + time.perf_counter() - started,
                peak_gpu_mib=max(primary["meta"]["peak_gpu_mib"], secondary["meta"]["peak_gpu_mib"]),
                fusion_parent_coverage_counts=parents.sum(0).tolist(),
                survival_counts_scope="primary source's actual beams; union parent coverage is separate",
                fusion_score_rule="Recompute one common score; never mix source calibrated scores",
                pool_size=tokens.shape[1], padded_candidates_are_not_predictions=True)
    new_json(output / "summary.json", meta)
    return load_cache(output)


def evaluate_cached(cache, parameters, features, output, name):
    output = Path(output)
    if (output / f"{name}.json").exists() or (output / f"{name}.npz").exists():
        raise FileExistsError(name)
    started = time.perf_counter()
    scores = score_candidates(cache, parameters, features)
    order = rank_candidates(cache, scores, parameters)
    tokens = np.take_along_axis(cache["tokens"][:, :scores.shape[1]], order[:, :, None], 1)
    ranks = ranks_for(tokens, cache["targets"])
    seconds = time.perf_counter() - started
    metrics = metrics_from_ranks(ranks)
    result = {"name": name, "parameters": parameters, "metrics": metrics, "rerank_seconds": seconds,
              "source_cache": cache["source_cache"],
              "source_search": {key: cache["meta"].get(key) for key in ("widths", "weights", "sibling", "prior_strengths", "suppress_seen_singletons")},
              "search_seconds": cache["meta"]["seconds"], "peak_gpu_mib": cache["meta"]["peak_gpu_mib"],
              "samples": len(ranks), "split": cache["meta"]["split"], "comparisons": {},
              "survival_counts": [(cache["depth_ranks"][:, 0] > 0).sum().item(),
                                  (cache["depth_ranks"][:, 1] > 0).sum().item(),
                                  int(((ranks > 0) & (ranks <= cache["meta"]["widths"][-1])).sum())],
              "final_beam_size": cache["meta"]["widths"][-1],
              "source_search_survival_counts": cache["meta"]["survival_counts"],
              "final_pool_target_count": int((ranks > 0).sum())}
    if "union_parent_survived" in cache:
        result["survival_counts"][:2] = cache["union_parent_survived"].sum(0).tolist()
        result["survival_definition"] = "D1/D2 union coverage from actual separate searches; D3 reranked final Beam. No fabricated fused parent ranks."
    for model in ("original_free", "beam_30_30_30"):
        reference = existing_baseline(cache["meta"]["split"], cache["sample_ids"], model)
        before, after = (reference > 0) & (reference <= 10), (ranks > 0) & (ranks <= 10)
        old = metrics_from_ranks(reference)
        result["comparisons"][model] = {"gained": int((after & ~before).sum()), "lost": int((before & ~after).sum()),
            "net": int(after.sum() - before.sum()),
            "absolute_change_pp": {k: (metrics[k] - old[k]) * 100 for k in metrics},
            "relative_change_percent": {k: (metrics[k] / old[k] - 1) * 100 for k in metrics}}
    halves = np.random.default_rng(42).permutation(len(ranks))
    result["validation_halves"] = [metrics_from_ranks(ranks[group]) for group in np.array_split(halves, 2)] if result["split"] == "valid" else None
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{name}.npz", ranks=ranks, top10_indices=order[:, :10], sample_ids=cache["sample_ids"])
    new_json(output / f"{name}.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True, help="JSON object containing scorer parameters")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    cache = load_cache(args.cache)
    with np.load(args.train_stats, allow_pickle=False) as archive:
        stats = {key: archive[key] for key in archive.files}
    features = make_features(cache, stats)
    evaluate_cached(cache, json.loads(args.parameters.read_text()), features, args.output, args.name)


if __name__ == "__main__":
    main()
