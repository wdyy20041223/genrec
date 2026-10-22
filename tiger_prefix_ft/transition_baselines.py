"""Train-only transition baselines and paired comparisons; no TIGER forward pass."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cotiger.core.trainers.tiger_trainer import calculate_pos_index
from .beam_diagnosis import BASE, ROOT
from .finetune import file_sha256
from .inference_lab import metrics_from_ranks, new_json, ranks_for
from .inference_priors import PrefixPrior
from .inference_rerank import SID_SPACE, lookup, sid_keys
from .summarize_inference_study import bootstrap, compare


from .paths import STUDY
# All definitions are fixed before reading Validation metrics. No Test tuning.
PROFILES = [
    dict(name="popularity", kind="popularity", history=0, smoothing=0, filter_seen=True),
    dict(name="sid_markov_last1", kind="sid_probability", history=1, smoothing=0, filter_seen=True),
    dict(name="sid_markov_recent3", kind="sid_probability", history=3, smoothing=0, filter_seen=True),
    dict(name="sid_markov_recent3_sm20", kind="sid_probability", history=3, smoothing=20, filter_seen=True),
    dict(name="sid_lift_last1", kind="sid_lift", history=1, smoothing=20, filter_seen=True),
    dict(name="sid_lift_recent3", kind="sid_lift", history=3, smoothing=20, filter_seen=True),
    dict(name="sid_lift_recent3_no_seen_filter", kind="sid_lift", history=3, smoothing=20, filter_seen=False),
    dict(name="item_transition_last1", kind="item_probability", history=1, smoothing=0, filter_seen=True),
    dict(name="item_transition_recent3_sm20", kind="item_probability", history=3, smoothing=20, filter_seen=True),
]


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grouped_rows(contexts, destinations, counts, size):
    """Sparse context -> (unique SID columns, counts), including context item 0."""
    keys, inverse = np.unique(contexts * size + destinations, return_inverse=True)
    totals = np.bincount(inverse, weights=counts)
    parents, children = keys // size, keys % size
    boundaries = np.r_[0, np.flatnonzero(parents[1:] != parents[:-1]) + 1, len(parents)]
    return {int(parents[a]): (children[a:b], totals[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])}


def topk(scores, tie_order, k=10):
    """Exact score Top-k; ties use Train popularity, then lexicographic SID."""
    ordered = scores[:, tie_order]
    thresholds = np.partition(ordered, ordered.shape[1] - k, axis=1)[:, -k]
    selected = []
    for row, threshold in zip(ordered, thresholds):
        above = np.flatnonzero(row > threshold)
        tied = np.flatnonzero(row == threshold)[:k - len(above)]
        columns = np.r_[above, tied]
        assert len(columns) == k and np.isfinite(row[columns]).all()
        selected.append(tie_order[columns[np.argsort(-row[columns], kind="stable")]])
    return np.asarray(selected, dtype=np.int32)


class Transitions:
    def __init__(self, item_sids, statistics, train_path):
        self.item_tokens = item_sids + np.array([1, 257, 513])
        self.keys, self.item_to_sid, self.group_sizes = np.unique(
            sid_keys(self.item_tokens), return_inverse=True, return_counts=True)
        self.catalog = self.item_tokens[np.unique(self.item_to_sid, return_index=True)[1]]
        assert np.array_equal(sid_keys(self.catalog), self.keys)
        self.frequency = lookup(statistics["sid_target_keys"], statistics["sid_target_counts"], self.keys)
        self.marginal = (self.frequency + 1) / (int(statistics["samples"]) + len(self.keys))
        self.tie_order = np.lexsort((self.keys, -self.frequency))
        pairs = statistics["sid_pair_keys"]
        self.sid_rows = grouped_rows(pairs // SID_SPACE,
                                    np.searchsorted(self.keys, pairs % SID_SPACE),
                                    statistics["sid_pair_counts"], len(self.keys))
        table = pq.read_table(train_path, columns=["sample_index", "history_item_indices", "target_item_index", "target_sid"])
        assert len(table) == int(statistics["samples"]) == 131413
        assert np.array_equal(table["sample_index"].to_numpy(), np.arange(len(table)))
        histories = table["history_item_indices"].to_pylist()
        previous = np.array([h[-1] for h in histories])
        target = table["target_item_index"].to_numpy()
        assert np.array_equal(item_sids[target], np.array(table["target_sid"].to_pylist()))
        projected, count = np.unique(sid_keys(self.item_tokens[previous]) * SID_SPACE + sid_keys(self.item_tokens[target]), return_counts=True)
        assert np.array_equal(projected, pairs) and np.array_equal(count, statistics["sid_pair_counts"])
        frequency = np.bincount(self.item_to_sid[target], minlength=len(self.keys))
        assert np.array_equal(frequency, self.frequency)
        item_pairs, item_counts = np.unique(previous * len(item_sids) + target, return_counts=True)
        self.item_rows = grouped_rows(item_pairs // len(item_sids), self.item_to_sid[item_pairs % len(item_sids)], item_counts, len(self.keys))
        self.item_pairs, self.item_counts = item_pairs, item_counts

    def scores(self, profile, sid_history, item_history):
        """Only observed histories, static catalog and Train counts enter scoring."""
        size = len(self.keys)
        values = np.zeros((len(sid_history), size), np.float32)
        support = np.zeros(len(sid_history), bool)
        if profile["kind"] == "popularity":
            values[:] = self.marginal
        else:
            history = item_history if profile["kind"].startswith("item_") else sid_history
            rows = self.item_rows if profile["kind"].startswith("item_") else self.sid_rows
            for age in range(profile["history"]):
                context = history[:, -age - 1]
                active = context >= 0
                if age:
                    active &= (context[:, None] != history[:, -age:]).all(-1)
                for row in np.flatnonzero(active):
                    found = rows.get(int(context[row]))
                    columns, counts = found if found is not None else (np.array([], dtype=int), np.array([]))
                    support[row] |= len(columns) > 0
                    weight = .5 ** age
                    if profile["kind"] == "sid_lift":
                        lift = np.log1p(counts / (20 * self.marginal[columns])).astype(np.float32)
                        # Exactly the existing final-layer prior, including float32 order.
                        values[row, columns] += .4 * weight * lift
                    else:
                        beta = profile["smoothing"]
                        denominator = counts.sum() + beta
                        if denominator:
                            if beta:
                                values[row] += weight * beta / denominator * self.marginal
                            values[row, columns] += weight * counts / denominator
        if profile["filter_seen"]:
            for row, history in enumerate(sid_history):
                seen = np.unique(history[history >= 0])
                columns = np.searchsorted(self.keys, seen)
                assert np.array_equal(self.keys[columns], seen)
                values[row, columns[self.group_sizes[columns] == 1]] -= 1000.
        return values, support


def evaluation_data(split, model, limit=None):
    table = pq.read_table(BASE / f"analysis/{split}_samples.parquet",
                          columns=["sample_index", "user_index", "history_item_indices", "target_item_index", "target_sid"])
    if limit is not None:
        table = table.slice(0, limit)
    histories = [h[-50:] for h in table["history_item_indices"].to_pylist()]
    items = np.full((len(table), 50), -1, np.int64)
    sid_history = np.full_like(items, -1)
    for row, history in enumerate(histories):
        assert len(history) > 0
        items[row, -len(history):] = history
        sid_history[row, -len(history):] = sid_keys(model.item_tokens[history])
    targets = np.array(table["target_sid"].to_pylist()) + [1, 257, 513]
    assert np.array_equal(targets, model.item_tokens[table["target_item_index"].to_numpy()])
    ids = np.array([f"{split}_{i:07d}" for i in table["sample_index"].to_pylist()])
    assert len(set(table["user_index"].to_pylist())) == len(table)
    with np.load(STUDY / split / "raw30_pool256/contexts.npz", allow_pickle=False) as old:
        assert np.array_equal(ids, old["sample_ids"][:len(ids)])
        assert np.array_equal(targets, old["targets"][:len(ids)])
        previous = sid_keys(old["histories"][:len(ids)])
        previous[previous == 0] = -1
        assert np.array_equal(previous, sid_history)
    return dict(sample_ids=ids, targets=targets, sid_history=sid_history, item_history=items)


def evaluate(model, data, batch_size):
    outputs, summary = {}, {}
    target_indices = np.searchsorted(model.keys, sid_keys(data["targets"]))
    for profile in PROFILES:
        indices, supports = [], []
        started = time.perf_counter()
        for start in range(0, len(target_indices), batch_size):
            sl = slice(start, start + batch_size)
            scores, support = model.scores(profile, data["sid_history"][sl], data["item_history"][sl])
            indices.append(topk(scores, model.tie_order))
            supports.append(support)
        predictions = np.concatenate(indices)
        elapsed = time.perf_counter() - started
        match = predictions == target_indices[:, None]
        ranks = np.where(match.any(1), match.argmax(1) + 1, 0).astype(np.int16)
        assert np.all(np.diff(np.sort(predictions, axis=1), axis=1) > 0)
        pos = calculate_pos_index(torch.from_numpy(model.catalog[predictions].astype(np.int64)), torch.from_numpy(data["targets"].astype(np.int64)), 10).numpy()
        assert np.array_equal(pos, match)
        metrics = metrics_from_ranks(ranks)
        for k in (5, 10):
            assert abs(metrics[f"Recall@{k}"] - pos[:, :k].sum() / len(pos)) < 1e-7
            assert abs(metrics[f"NDCG@{k}"] - (pos[:, :k] / np.log2(np.arange(2, k + 2))).sum() / len(pos)) < 1e-7
        outputs[profile["name"]] = dict(indices=predictions, ranks=ranks)
        summary[profile["name"]] = dict(metrics=metrics, seconds=elapsed,
            history_has_train_transition_count=int(np.concatenate(supports).sum()), profile=profile)
        print(json.dumps(dict(method=profile["name"], samples=len(ranks), seconds=elapsed, metrics=metrics)), flush=True)
    return outputs, summary


def check_smoke(model, stats, data, output):
    n = len(data["sample_ids"])
    candidates = []
    for history in data["sid_history"]:
        columns = []
        for context in history[-3:][::-1]:
            if int(context) in model.sid_rows:
                columns.extend(model.sid_rows[int(context)][0][:32].tolist())
        candidates.append(list(dict.fromkeys(columns + model.tie_order[:32].tolist()))[:32])
    candidates = np.array(candidates)
    profile = next(p for p in PROFILES if p["name"] == "sid_lift_recent3")
    scores, _ = model.scores(profile, data["sid_history"], data["item_history"])
    ids = np.zeros((n, 150), np.int64)
    mask = np.repeat(data["item_history"] >= 0, 3, axis=1).astype(np.int64)
    for row in range(n):
        valid = data["item_history"][row] >= 0
        ids[row, mask[row].astype(bool)] = model.item_tokens[data["item_history"][row, valid]].reshape(-1)
    prior = PrefixPrior(stats, model.item_tokens, [0., 0., .4], True, history_items=3)
    prior.prepare(torch.from_numpy(ids), torch.from_numpy(mask))
    tokens = model.catalog[candidates]
    parents = np.concatenate((np.zeros((n, 32, 1), np.int64), tokens[:, :, :2]), axis=-1)
    adjustment = prior(3, torch.from_numpy(parents), torch.zeros(n, 32, 769))
    actual = adjustment.gather(-1, torch.from_numpy(tokens[:, :, -1:])).squeeze(-1).numpy()
    expected = np.take_along_axis(scores, candidates, axis=1)
    error = float(np.abs(actual - expected).max())
    assert error < 2e-6
    new_json(output / "smoke_check.json", dict(state="passed", real_validation_samples=n,
        exact_existing_prior_max_error=error, selected_candidates_include_actual_train_transitions=True,
        labels_not_used_by_scorer=True, top10_unique_catalog_sids=True, official_metric_check=True))


def references(split, data):
    rows = pq.read_table(STUDY / "analysis/selected_samples.parquet",
                         filters=[("split", "=", split), ("model_name", "=", "prior30")])
    assert rows["sample_id"].to_pylist() == data["sample_ids"].tolist()
    assert np.array_equal(np.array(rows["target_tokens"].to_pylist()), data["targets"])
    rank = lambda field: np.array([v or 0 for v in rows[field].to_pylist()], np.int16)
    result = {"original15": rank("original15_rank"), "original30": rank("original30_rank"), "prior30": rank("final_rank")}
    smaller = pq.read_table(STUDY / "analysis/selected_samples.parquet", filters=[("split", "=", split), ("model_name", "=", "prior15")])
    assert smaller["sample_id"].to_pylist() == data["sample_ids"].tolist()
    result["prior15"] = np.array([v or 0 for v in smaller["final_rank"].to_pylist()], np.int16)
    with np.load(STUDY / split / "raw30_pool256/candidates.npz", allow_pickle=False) as old:
        tokens = old["tokens"][:, :10]
    assert np.array_equal(ranks_for(tokens, data["targets"]), np.where(result["original30"] <= 10, result["original30"], 0))
    return result, tokens, np.array(rows["predicted_top10"].to_pylist())


def paired(reference, standalone, combined):
    hit = lambda rank: (rank > 0) & (rank <= 10)
    a, b, c = map(hit, (reference, standalone, combined))
    groups = {"both": a & b, "tiger_only": a & ~b, "statistics_only": b & ~a, "neither": ~a & ~b}
    return dict(contingency={k: int(v.sum()) for k, v in groups.items()},
        combined_hits_by_group={k: int((v & c).sum()) for k, v in groups.items()},
        oracle_union_hits=int((a | b).sum()), oracle_union_recall=float((a | b).mean()),
        oracle_is_not_a_deployable_top10_metric=True,
        combined_vs_statistics=compare(combined, standalone), combined_vs_tiger=compare(combined, reference))


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source_paths = [BASE / f"analysis/{s}_samples.parquet" for s in ("train", "valid", "test")]
    source_paths += [BASE / "analysis/item_semantic_ids.npy", BASE / "run/tiger/best_model.pt", STUDY / "train_statistics.npz",
                     STUDY / "analysis/selected_samples.parquet", Path(__file__)]
    sources = {str(p): file_sha256(p) for p in source_paths}
    new_json(output / "protocol.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(), profiles=PROFILES,
        selection_split="valid", samples_per_evaluation=22363, candidate_catalog="all 11401 unique legal SIDs; no TIGER candidate restriction",
        ties="descending Train SID frequency, then ascending SID; also fills zero-transition slots",
        item_baseline="directed adjacent real-item transitions; probability mass summed within SID collision groups",
        probability_score="sum of recency weighted (C(context,SID)+beta*q(SID))/(C(context)+beta); last3 positions, duplicates kept once",
        lift_score="exact prior used by the frozen hybrid: 0.4*sum 0.5**age*log1p(C/(20*q))",
        filter_seen="same as hybrid: only seen SIDs mapping to one catalog item get -1000", source_sha256=sources,
        no_model_training=True, no_tiger_forward=True, historical_test_already_examined=True))
    with np.load(STUDY / "train_statistics.npz", allow_pickle=False) as f:
        stats = {k: f[k] for k in f.files}
    started = time.perf_counter()
    model = Transitions(np.load(BASE / "analysis/item_semantic_ids.npy", allow_pickle=False), stats, source_paths[0])
    np.savez_compressed(output / "item_transition_counts.npz", item_pair_keys=model.item_pairs, counts=model.item_counts,
                        catalog_tokens=model.catalog, item_to_sid=model.item_to_sid)
    new_json(output / "train_count_check.json", dict(state="passed", rows=131413, sid_projection_exactly_matches_existing_counts=True,
        unique_item_transitions=len(model.item_pairs), unique_sid_transitions=len(stats["sid_pair_keys"]), seconds=time.perf_counter()-started))
    smoke = evaluation_data("valid", model, 100)
    _, smoke_metrics = evaluate(model, smoke, args.batch_size)
    check_smoke(model, stats, smoke, output)
    new_json(output / "smoke_metrics.json", smoke_metrics)
    all_summary, parquet_rows, rng = {}, [], np.random.default_rng(20261016)
    for split in ("valid", "test"):
        data = evaluation_data(split, model)
        assert len(data["sample_ids"]) == 22363
        predictions, summary = evaluate(model, data, args.batch_size)
        if split == "valid":
            best = max(summary, key=lambda name: (summary[name]["metrics"]["Recall@10"], summary[name]["metrics"]["NDCG@10"]))
            new_json(output / "selection_lock.json", dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),
                selection_split="valid", best_standalone_by_validation=best, profiles=PROFILES,
                criterion="Recall@10 first, then NDCG@10", no_parameter_changes_for_test=True))
        baseline, tiger_tokens, hybrid_tokens = references(split, data)
        ranks = {name: result["ranks"] for name, result in predictions.items()} | baseline
        comparisons = {}
        for name in dict.fromkeys(["sid_lift_recent3", best]):
            comparisons[name] = paired(baseline["original30"], ranks[name], baseline["prior30"])
            comparisons[name]["combined_vs_statistics_ci95"] = bootstrap(baseline["prior30"], ranks[name], rng)
            selected = model.catalog[predictions[name]["indices"]]
            comparisons[name]["mean_top10_overlap_with_tiger30"] = float((selected[:, :, None] == tiger_tokens[:, None]).all(-1).sum((1, 2)).mean())
            comparisons[name]["mean_top10_overlap_with_hybrid30"] = float((selected[:, :, None] == hybrid_tokens[:, None]).all(-1).sum((1, 2)).mean())
        arrays = {"sample_ids": data["sample_ids"], "target_tokens": data["targets"], "catalog_tokens": model.catalog}
        arrays.update({name + "_top10_indices": result["indices"].astype(np.int16) for name, result in predictions.items()})
        arrays.update({name + "_ranks": values for name, values in ranks.items()})
        np.savez_compressed(output / f"{split}_predictions.npz", **arrays)
        for i, sample_id in enumerate(data["sample_ids"]):
            parquet_rows.append(dict(sample_id=str(sample_id), split=split, target_tokens=data["targets"][i].tolist(),
                **{name + "_rank": int(values[i]) or None for name, values in ranks.items()}))
        all_summary[split] = dict(samples=len(data["sample_ids"]), standalone=summary,
            tiger_references={name: metrics_from_ranks(value) for name, value in baseline.items()},
            complementarity=comparisons, best_standalone_fixed_on_validation=best)
        new_json(output / f"{split}_summary.json", all_summary[split])
    assert all(file_sha256(Path(path)) == expected for path, expected in sources.items())
    pq.write_table(pa.Table.from_pylist(parquet_rows), output / "paired_samples.parquet", compression="zstd")
    new_json(output / "comparison.json", dict(state="complete", no_tiger_forward=True, no_training=True,
        train_only_counts=True, frozen_sources_unchanged=True, split_results=all_summary,
        selection_lock=json_read(output / "selection_lock.json")))
    print(json.dumps(dict(state="complete", best_standalone_by_validation=best, output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    assert args.batch_size > 0
    run(args)


if __name__ == "__main__":
    main()
