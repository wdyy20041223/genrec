"""Offline diagnosis of the original Beauty TIGER's actual three-step Beam tree."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import ROOT, RAW_ROOT as RAW
DOCS = ROOT / "docs/generated/tiger_prefix_ft_beauty_20261013"
DIAG = ROOT / "out/tiger_prefix_ft/beauty_diagnosis_lambda_20261015_121635/diagnosis"
K, D, V = 15, 3, 769
KEYS = ("next_token_logprobs", "beam_scores_before", "selected_flat_indices",
        "selected_parent_indices", "selected_token_ids", "selected_cumulative_scores",
        "active_beam_mask", "initial_placeholder_mask", "forward_evaluated_mask",
        "logprob_source_parent_index", "parent_prefix_tokens_with_start",
        "final_tokens_with_start", "final_cumulative_scores", "dataset_indices")


def raw_helpers():
    from .raw_collection import raw_utils
    return raw_utils


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def deviation(paths, target):
    """First disagreement with the observed target, 1-based; 0 means matching."""
    if not len(target):
        return np.zeros(len(paths), dtype=np.int64)
    different = paths != np.asarray(target)[None, :]
    return np.where(different.any(1), different.argmax(1) + 1, 0)


def rank_bounds(values, gold):
    return 1 + int(np.count_nonzero(values > gold)), int(np.count_nonzero(values >= gold))


def oracle_bounds(greater, equal, foreign_greater, foreign_equal, remove):
    """Exact score-order bounds; never choose an invented order within ties.

    Finite removal deletes the highest strict-above-Gold foreign candidates.
    'all' removes every foreign-parent candidate, including ties with Gold.
    equal includes Gold itself. Guaranteed/possible are valid for every/any
    ordering among the remaining equal-score candidates, respectively.
    """
    if remove == "all":
        remaining_greater = greater - foreign_greater
        remaining_equal = equal - foreign_equal
    else:
        removed = min(remove, foreign_greater)
        remaining_greater = greater - removed
        remaining_equal = equal
        if removed == 0:
            # The actual Gold was pruned and the candidate matrix is unchanged.
            return {"guaranteed": False, "possible": False,
                    "rank_min": max(K + 1, greater + 1), "rank_max": greater + equal}
    assert remaining_greater >= 0 and remaining_equal >= 1
    return {"guaranteed": remaining_greater + remaining_equal <= K,
            "possible": remaining_greater < K,
            "rank_min": remaining_greater + 1,
            "rank_max": remaining_greater + remaining_equal}


def inspect_sample(record, data, index):
    target = np.asarray(record["target_tokens"], dtype=np.int64)
    assert np.array_equal(target, np.asarray(record["target_sid"]) + [1, 257, 513])
    assert record["split"] == "valid" and record["decoder_start_token_id"] == 0
    assert record["beam_size"] == K and record["sid_length"] == D
    assert record["dataset_index"] == int(data["dataset_indices"][index])
    rows, details, earlier = [], [], []
    d1_ancestors = None
    for step in range(D):
        depth = step + 1
        parents = data["parent_prefix_tokens_with_start"][index, step, :, :depth]
        parent_score = data["beam_scores_before"][index, step]
        logp = data["next_token_logprobs"][index, step]
        active = data["active_beam_mask"][index, step]
        pidx = data["selected_parent_indices"][index, step].astype(np.int64)
        tokens = data["selected_token_ids"][index, step]
        scores = data["selected_cumulative_scores"][index, step]
        selected = np.concatenate([parents[pidx, 1:], tokens[:, None]], axis=1)
        origins = deviation(selected, target[:depth])
        parent_origins = deviation(parents[:, 1:], target[:step])
        parent_origins[~active] = -1
        gold_parents = np.flatnonzero((parent_origins == 0) & active)
        assert len(gold_parents) <= 1
        parent_alive = bool(len(gold_parents))
        if earlier:
            assert parent_alive == earlier[-1]["gold_survived"]
        gold_matches = np.flatnonzero(origins == 0)
        assert len(gold_matches) <= 1 and (not len(gold_matches) or parent_alive)
        alive = bool(len(gold_matches))
        rank = int(gold_matches[0]) + 1 if alive else None
        first_pruned = parent_alive and not alive
        d1_ancestors = np.arange(K) if step == 0 else d1_ancestors[pidx]
        fanout = np.bincount(pidx, minlength=K)
        wrong_parent = (parent_origins > 0) & active
        from_wrong_parent = parent_origins[pidx] > 0
        wrong_roots = np.bincount(d1_ancestors[origins == 1], minlength=K)
        top5_wrong_parents = np.flatnonzero(wrong_parent & (np.arange(K) < 5))
        selected_top5_parents = set(pidx[:5].tolist())
        row = {"sample_id": record["sample_id"], "dataset_index": record["dataset_index"],
            "tensor_shard": record["tensor_shard"], "tensor_row": record["tensor_row"],
            "depth": depth, "target_tokens": target.tolist(), "gold_parent_alive": parent_alive,
            "gold_survived": alive, "first_pruned_here": first_pruned,
            "gold_rank": rank, "gold_score": float(scores[rank - 1]) if alive else None,
            "gold_parent_rank": int(gold_parents[0]) + 1 if parent_alive else None,
            "gold_candidate_score": None, "gold_margin": None,
            "gold_local_rank_min": None, "gold_local_rank_max": None,
            "gold_global_rank_min": None, "gold_global_rank_max": None,
            "gold_equal_score_candidates": None, "prune_class": None,
            "top15_threshold": float(scores[-1]),
            "gold_parent_child_slots": int(fanout[gold_parents[0]]) if parent_alive else 0,
            "non_target_parent_slots": int(from_wrong_parent.sum()),
            "distinct_selected_parents": int(np.count_nonzero(fanout)),
            "max_children_one_parent": int(fanout.max()),
            "non_target_parent_count": int(wrong_parent.sum()),
            "non_target_parents_with_children": int(np.count_nonzero(fanout[wrong_parent])),
            "non_target_parents_with_multiple_children": int(np.count_nonzero(fanout[wrong_parent] >= 2)),
            "max_children_one_non_target_parent": int(fanout[wrong_parent].max()) if wrong_parent.any() else 0,
            "max_d1_non_target_ancestor_slots": int(wrong_roots.max()),
            "d1_non_target_ancestors_represented": int(np.count_nonzero(wrong_roots)),
            "non_target_top5_parent_count": len(top5_wrong_parents),
            "non_target_top5_parents_continued_in_top5": sum(int(p) in selected_top5_parents for p in top5_wrong_parents),
            "selected_first_deviation_depths": origins.tolist(),
            "selected_parent_ranks": (pidx + 1).tolist(),
            "selected_parent_first_deviation_depths": parent_origins[pidx].tolist(),
            "selected_d1_ancestor_ranks": (d1_ancestors + 1).tolist(),
            "slots_by_first_deviation": np.bincount(origins, minlength=4).tolist(),
            "best_non_target_minus_gold": None, "mean_continuing_relative_gap_change": None,
            "continuing_gap_expands": None, "continuing_gap_shrinks": None,
            "continuing_gap_unchanged": None, "non_target_leads_parent_and_child": None,
            "d1_non_target_leads_all_three": None, "d1_non_target_top5_all_three": None,
            "foreign_candidates_above_gold": None, "foreign_above_gold_by_first_deviation": None,
            "foreign_selected_above_gold": None}
        for removed in (1, 3, 5, "all"):
            for bound in ("guaranteed", "possible"):
                row[f"oracle_remove_{removed}_{bound}"] = None
        candidate = parent_score[:, None] + logp
        assert np.array_equal(candidate[pidx, tokens], scores)
        gold_candidate = None
        if parent_alive:
            gp = int(gold_parents[0])
            gold_candidate = candidate[gp, target[step]]
            local_min, local_max = rank_bounds(logp[gp], logp[gp, target[step]])
            global_min, global_max = rank_bounds(candidate[active], gold_candidate)
            equal = global_max - global_min + 1
            if alive:
                assert global_min <= rank <= global_max and scores[rank - 1] == gold_candidate
                if earlier:
                    assert gold_candidate <= earlier[-1]["gold_score"] + 1e-5
            else:
                assert global_max > K
            foreign_greater = int(np.count_nonzero(candidate[wrong_parent] > gold_candidate))
            foreign_equal = int(np.count_nonzero(candidate[wrong_parent] == gold_candidate))
            foreign_origins = [int(np.count_nonzero(candidate[parent_origins == o] > gold_candidate)) for o in (1, 2)]
            row.update(gold_candidate_score=float(gold_candidate),
                gold_margin=float(gold_candidate - scores[-1]),
                gold_local_rank_min=local_min, gold_local_rank_max=local_max,
                gold_global_rank_min=rank if alive else max(K + 1, global_min),
                gold_global_rank_max=rank if alive else global_max,
                gold_equal_score_candidates=equal,
                foreign_candidates_above_gold=foreign_greater,
                foreign_above_gold_by_first_deviation=foreign_origins,
                foreign_selected_above_gold=int(np.count_nonzero(from_wrong_parent & (scores > gold_candidate))),
                best_non_target_minus_gold=float(scores[origins > 0].max() - gold_candidate))
            if first_pruned:
                row["prune_class"] = "A_local_poor" if local_min > K else (
                    "B_cross_parent" if local_max <= K else "local_tie_boundary")
                for removed in ((1, 3, 5, "all") if depth > 1 else ()):
                    bounds = oracle_bounds(global_min - 1, equal, foreign_greater, foreign_equal, removed)
                    for bound in ("guaranteed", "possible"):
                        row[f"oracle_remove_{removed}_{bound}"] = bool(bounds[bound])
            if depth > 1:
                ids = np.flatnonzero(from_wrong_parent)
                parent_gap = parent_score[pidx[ids]] - parent_score[gp]
                child_gap = scores[ids] - gold_candidate
                change = child_gap - parent_gap
                row.update(mean_continuing_relative_gap_change=float(change.mean()) if len(change) else None,
                    continuing_gap_expands=int(np.count_nonzero(change > 1e-6)),
                    continuing_gap_shrinks=int(np.count_nonzero(change < -1e-6)),
                    continuing_gap_unchanged=int(np.count_nonzero(np.abs(change) <= 1e-6)),
                    non_target_leads_parent_and_child=int(np.count_nonzero((parent_gap > 0) & (child_gap > 0))))
            if depth == 3:
                early = origins == 1
                d1_gap = details[0]["scores"][d1_ancestors] - earlier[0]["gold_score"]
                d2_gap = parent_score[pidx] - parent_score[gp]
                row["d1_non_target_leads_all_three"] = int(np.count_nonzero(early & (d1_gap > 0) & (d2_gap > 0) & (scores > gold_candidate)))
                row["d1_non_target_top5_all_three"] = int(np.count_nonzero(early & (d1_ancestors < 5) & (pidx < 5) & (np.arange(K) < 5)))
        rows.append(row)
        earlier.append(row)
        details.append({"depth": depth, "paths": selected.copy(), "parents": parents[:, 1:].copy(),
                        "parent_scores": parent_score.copy(), "parent_indices": pidx.copy(),
                        "scores": scores.copy(), "origins": origins.copy(), "row": row})
    first = next((r["depth"] for r in rows if r["first_pruned_here"]), 0)
    for row in rows:
        row["first_prune_depth"] = first
        row["final_gold_rank"] = rows[-1]["gold_rank"]
        row["hit_at_10"] = rows[-1]["gold_rank"] is not None and rows[-1]["gold_rank"] <= 10
    assert np.array_equal(details[-1]["paths"], data["final_tokens_with_start"][index, :, 1:])
    return rows, details


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    assert np.isfinite(array).all()
    return {"n": len(values), "mean": float(array.mean()), **{
        f"p{q}": float(np.percentile(array, q)) for q in (0, 10, 25, 50, 75, 90, 95, 99, 100)}}


def occupancy(rows):
    n = len(rows)
    counts = np.sum([r["slots_by_first_deviation"] for r in rows], axis=0).tolist() if n else [0] * 4
    parents = sum(r["non_target_parent_count"] for r in rows)
    top5_parents = sum(r["non_target_top5_parent_count"] for r in rows)
    return {"samples": n, "beam_slots_denominator": K * n,
        "slots_by_first_deviation": {str(d): int(counts[d]) for d in range(4)},
        "slots_share_by_first_deviation": {str(d): counts[d] / (K * n) if n else None for d in range(4)},
        "non_target_parent_slots": sum(r["non_target_parent_slots"] for r in rows),
        "non_target_parent_slot_share": sum(r["non_target_parent_slots"] for r in rows) / (K * n) if n else None,
        "distinct_parents": describe([r["distinct_selected_parents"] for r in rows]),
        "max_children_one_parent": describe([r["max_children_one_parent"] for r in rows]),
        "max_children_one_non_target_parent": describe([r["max_children_one_non_target_parent"] for r in rows]),
        "max_d1_non_target_ancestor_slots": describe([r["max_d1_non_target_ancestor_slots"] for r in rows]),
        "non_target_parents": parents,
        "non_target_parents_with_children": sum(r["non_target_parents_with_children"] for r in rows),
        "non_target_parent_continuation_rate": sum(r["non_target_parents_with_children"] for r in rows) / parents if parents else None,
        "non_target_parents_with_multiple_children": sum(r["non_target_parents_with_multiple_children"] for r in rows),
        "non_target_top5_parents": top5_parents,
        "non_target_top5_parents_continued_in_top5": sum(r["non_target_top5_parents_continued_in_top5"] for r in rows),
        "non_target_top5_parent_continuation_rate": sum(r["non_target_top5_parents_continued_in_top5"] for r in rows) / top5_parents if top5_parents else None}


def summarize(records, total_validation):
    layers = {d: [r for r in records if r["depth"] == d] for d in (1, 2, 3)}
    n = len(layers[1])
    summary = {"state": "running", "split": "valid", "samples": n,
        "validation_total": total_validation, "beam_size": K, "sid_length": D,
        "scope": "full_validation" if n == total_validation else "smoke",
        "model": "original_TIGER", "depths": {}, "no_training": True,
        "no_model_inference": True, "no_test_data": True,
        "rank_tie_policy": "Saved selection defines actual survival/rank. Unselected ranks use strict/equal bounds; Oracle guaranteed/possible bounds preserve all FP32 scores.",
        "oracle_scope": "Label-based, one-step removal of foreign-parent candidates; not deployable and not a downstream generation replay."}
    for d, rows in layers.items():
        incoming = [r for r in rows if r["gold_parent_alive"]]
        alive = [r for r in rows if r["gold_survived"]]
        pruned = [r for r in incoming if r["first_pruned_here"]]
        classes = {c: sum(r["prune_class"] == c for r in pruned) for c in ("A_local_poor", "B_cross_parent", "local_tie_boundary")}
        groups = {"all": rows, "gold_parent_alive": incoming, "gold_survived": alive,
                  "first_pruned": pruned, "gold_already_pruned": [r for r in rows if not r["gold_parent_alive"]]}
        strata = {}
        for label, lo, hi in (("1_5", 1, 5), ("6_15", 6, 15), ("16_50", 16, 50), ("51_plus", 51, 100000)):
            subset = [r for r in incoming if lo <= r["gold_local_rank_min"] <= hi]
            strata[label] = {"definition": "strict-greater local probability rank lower bound",
                "survived": occupancy([r for r in subset if r["gold_survived"]]),
                "pruned": occupancy([r for r in subset if r["first_pruned_here"]])}
        for label, lo, hi in (("within_0p05", -0.05, 0.05), ("moderate_negative", -0.5, -0.05), ("below_minus0p5", -100000, -0.5)):
            subset = [r for r in incoming if lo <= r["gold_margin"] < hi]
            strata["margin_" + label] = {"survived": occupancy([r for r in subset if r["gold_survived"]]),
                "pruned": occupancy([r for r in subset if r["first_pruned_here"]])}
        matrices = {}
        for label, subset in (("all", rows), ("gold_parent_alive", incoming)):
            matrix = np.zeros((K, K), dtype=np.int64)
            by_origin = {o: np.zeros((K, K), dtype=np.int64) for o in (1, 2)}
            for r in subset:
                for child, (parent_rank, origin) in enumerate(zip(r["selected_parent_ranks"], r["selected_parent_first_deviation_depths"])):
                    if origin > 0:
                        matrix[parent_rank - 1, child] += 1
                        by_origin[origin][parent_rank - 1, child] += 1
            matrices[label] = {"axes": "parent rank 1..15 x child rank 1..15; already non-target parents only",
                "counts": matrix.tolist(), "by_first_deviation": {str(o): a.tolist() for o, a in by_origin.items()}}
        continuations = {}
        for label, subset in (("gold_parent_alive", incoming), ("gold_survived", alive), ("first_pruned", pruned)):
            path_count = sum(r["non_target_parent_slots"] for r in subset)
            continuations[label] = {"samples": len(subset), "continued_paths": path_count,
                "expanded_relative_advantage": sum(r["continuing_gap_expands"] or 0 for r in subset),
                "shrunk_relative_advantage": sum(r["continuing_gap_shrinks"] or 0 for r in subset),
                "unchanged_relative_advantage": sum(r["continuing_gap_unchanged"] or 0 for r in subset),
                "leads_parent_and_child_paths": sum(r["non_target_leads_parent_and_child"] or 0 for r in subset),
                "samples_with_continuous_leading_competitors": sum((r["non_target_leads_parent_and_child"] or 0) > 0 for r in subset),
                "d1_wrong_paths_lead_all_three": sum(r["d1_non_target_leads_all_three"] or 0 for r in subset),
                "samples_with_d1_wrong_leading_all_three": sum((r["d1_non_target_leads_all_three"] or 0) > 0 for r in subset),
                "d1_wrong_paths_in_top5_all_three": sum(r["d1_non_target_top5_all_three"] or 0 for r in subset),
                "relative_gap_change_per_user_mean": describe([r["mean_continuing_relative_gap_change"] for r in subset])}
            if d > 1:
                assert path_count == sum(continuations[label][key] for key in ("expanded_relative_advantage", "shrunk_relative_advantage", "unchanged_relative_advantage"))
        oracle = {}
        for removed in (1, 3, 5, "all"):
            guaranteed = sum(bool(r[f"oracle_remove_{removed}_guaranteed"]) for r in pruned)
            possible = sum(bool(r[f"oracle_remove_{removed}_possible"]) for r in pruned)
            oracle[str(removed)] = {"denominator_first_pruned": len(pruned), "guaranteed_rescued": guaranteed,
                "possible_rescued_including_guaranteed": possible, "tie_ambiguous": possible - guaranteed,
                "guaranteed_rate": guaranteed / len(pruned) if pruned else None,
                "possible_rate": possible / len(pruned) if pruned else None}
        summary["depths"][str(d)] = {"samples": n, "gold_parent_alive": len(incoming),
            "survival_count": len(alive), "survival_rate": len(alive) / n,
            "conditional_survival_rate": len(alive) / len(incoming) if incoming else None,
            "first_prune_count": len(pruned), "first_prune_rate_all_samples": len(pruned) / n,
            "first_prune_rate_given_parent": len(pruned) / len(incoming) if incoming else None,
            "gold_rank_distribution_when_survived": {str(rank): sum(r["gold_rank"] == rank for r in alive) for rank in range(1, K + 1)},
            "gold_margin_parent_alive": describe([r["gold_margin"] for r in incoming]),
            "gold_margin_first_pruned": describe([r["gold_margin"] for r in pruned]),
            "near_miss_counts_first_pruned": {str(epsilon): sum(-epsilon <= r["gold_margin"] <= 0 for r in pruned) for epsilon in (0.01, 0.05, 0.1, 0.5)},
            "prune_classes": {c: {"count": count, "fraction_first_pruned": count / len(pruned) if pruned else None,
                "fraction_all_samples": count / n} for c, count in classes.items()},
            "prune_class_profiles": {c: {
                "local_rank_lower": describe([r["gold_local_rank_min"] for r in pruned if r["prune_class"] == c]),
                "global_rank_lower": describe([r["gold_global_rank_min"] for r in pruned if r["prune_class"] == c]),
                "margin": describe([r["gold_margin"] for r in pruned if r["prune_class"] == c])}
                for c in classes},
            "gold_local_rank_first_pruned": describe([r["gold_local_rank_min"] for r in pruned]),
            "gold_global_rank_first_pruned": describe([r["gold_global_rank_min"] for r in pruned]),
            "foreign_above_gold_first_pruned": describe([r["foreign_candidates_above_gold"] for r in pruned]),
            "foreign_above_gold_by_first_deviation": {str(o): sum(r["foreign_above_gold_by_first_deviation"][o - 1] for r in pruned) for o in (1, 2)},
            "occupancy": {label: occupancy(subset) for label, subset in groups.items()},
            "controlled_comparisons": strata, "non_target_rank_transitions": matrices,
            "continuing_branch_advantage": continuations,
            "best_non_target_minus_gold": {label: describe([r["best_non_target_minus_gold"] for r in subset]) for label, subset in groups.items()},
            "oracle": oracle if d > 1 else {"not_applicable": "D2/D3 only"}}
    final = layers[3]
    summary["first_prune_counts"] = {str(d): sum(r["first_prune_depth"] == d for r in final) for d in range(4)}
    summary["final_gold_rank_groups"] = {label: sum(r["gold_rank"] is not None and lo <= r["gold_rank"] <= hi for r in final)
        for label, lo, hi in (("top1_5", 1, 5), ("rank6_10", 6, 10), ("rank11_15", 11, 15))}
    summary["final_non_target_slots_by_origin"] = {str(k): {
        str(o): sum(sum(origin == o for origin in r["selected_first_deviation_depths"][:k]) for r in final)
        for o in range(4)} for k in (5, 10, 15)}
    partition = {"local_prediction_poor": sum(summary["depths"][str(d)]["prune_classes"]["A_local_poor"]["count"] for d in (1, 2, 3)),
        "cross_parent_competition": sum(summary["depths"][str(d)]["prune_classes"]["B_cross_parent"]["count"] for d in (2, 3)),
        "local_rank_boundary_tie": sum(summary["depths"][str(d)]["prune_classes"]["local_tie_boundary"]["count"] for d in (1, 2, 3)),
        "final_rank11_15": summary["final_gold_rank_groups"]["rank11_15"],
        "hit_top10": sum(summary["final_gold_rank_groups"][key] for key in ("top1_5", "rank6_10"))}
    assert sum(partition.values()) == n
    summary["exclusive_first_failure_partition"] = {key: {"count": count, "fraction_all_samples": count / n} for key, count in partition.items()}
    summary["conclusions"] = conclusions(summary)
    return summary


def conclusions(summary):
    depths = summary["depths"]
    a, b = [depths[d]["occupancy"]["gold_parent_alive"] for d in ("2", "3")]
    classes = [depths[d]["prune_classes"]["B_cross_parent"]["count"] for d in ("2", "3")]
    rescued = {m: sum(depths[d]["oracle"][str(m)]["guaranteed_rescued"] for d in ("2", "3")) for m in (1, 3, 5, "all")}
    upper_all = sum(depths[d]["oracle"]["all"]["possible_rescued_including_guaranteed"] for d in ("2", "3"))
    percent = lambda x: "无适用样本" if x is None else f"{x:.4%}"
    return [
        f'非目标前缀的后续存活子路径按真实树统计；Gold 父路径存在时，D1 非目标父到 D2 的延续率为 {percent(a["non_target_parent_continuation_rate"])}，D2 非目标父到 D3 为 {percent(b["non_target_parent_continuation_rate"])}。',
        f'Gold 父路径仍存在时，D2/D3 分别有 {a["non_target_parent_slots"]}/{b["non_target_parent_slots"]} 个名额来自已偏离父路径，分别占 {percent(a["non_target_parent_slot_share"])}/{percent(b["non_target_parent_slot_share"])}；当前层才首次偏离的子路径另计。',
        f'竞争与剪枝有关：D2/D3 B 类分别 {classes[0]}/{classes[1]} 条，Gold 局部排名上界≤15却被全局淘汰；A 类也可能受竞争影响，不能把它们全部归为纯基础能力问题或把占位相关性当作因果。',
        f'局部不差却全局淘汰共 {sum(classes)} 条，占全部样本 {sum(classes)/summary["samples"]:.4%}。局部跨第15名并列单列，不虚构原始 CUDA 并列排序。',
        f'首次在 D1/D2/D3 剪枝分别为 {summary["first_prune_counts"]["1"]}/{summary["first_prune_counts"]["2"]}/{summary["first_prune_counts"]["3"]}；另有 {summary["final_gold_rank_groups"]["rank11_15"]} 条完整存活但最终第11～15名。',
        f'相对优势按真实父子链比较；Gold D2 父路径仍存在的 {b["samples"]} 条样本中，{depths["3"]["continuing_branch_advantage"]["gold_parent_alive"]["samples_with_d1_wrong_leading_all_three"]} 条存在 D1 非目标链连续三层严格领先 Gold，不以绝对 Log Probability 增加判定级联。',
        f'固定原分数的单步 Oracle 移除1/3/5个高分非目标父扩展，分别确定恢复 {rescued[1]}/{rescued[3]}/{rescued[5]} 条；全排除时确定恢复 {rescued["all"]} 条，计入并列后可能最多 {upper_all} 条。这不是最终 Recall 改善，也不是可部署算法。',
        ('现有数据支持继续研究跨父分支的可识别风险及竞争控制，但不能证明 Teacher Forcing / Exposure Bias 是根因，也不能把非目标商品等同于不相关商品。少量删候选只恢复一部分，仍需考虑基础预测与最终排序。'
         if sum(classes) and rescued[5] else '现有恢复证据不足，应先检查局部预测、排序和并列，不能据此断言下一步方法有效。')]


def choose_cases(records):
    selected, used, missing = [], set(), []
    criteria = [
        ("D1 早期失败", lambda r: r["depth"] == 1 and r["first_pruned_here"], lambda r: r["dataset_index"]),
        ("D2 局部预测差", lambda r: r["depth"] == 2 and r["prune_class"] == "A_local_poor", lambda r: -r["gold_local_rank_min"]),
        ("D2 跨父路径竞争", lambda r: r["depth"] == 2 and r["prune_class"] == "B_cross_parent", lambda r: -r["gold_margin"]),
        ("D3 跨父路径竞争", lambda r: r["depth"] == 3 and r["prune_class"] == "B_cross_parent", lambda r: -r["gold_margin"]),
        ("D1 非目标分支连续三层领先", lambda r: (r["d1_non_target_leads_all_three"] or 0) > 0,
            lambda r: -(r["d1_non_target_leads_all_three"] or 0)),
        ("D2 少数非目标父路径集中占位", lambda r: r["depth"] == 2 and r["first_pruned_here"], lambda r: -r["max_children_one_non_target_parent"]),
        ("D3 少数早期非目标祖先集中占位", lambda r: r["depth"] == 3 and r["gold_parent_alive"], lambda r: -r["max_d1_non_target_ancestor_slots"]),
        ("小 Margin 首次剪枝", lambda r: r["depth"] > 1 and r["first_pruned_here"] and r["gold_margin"] < 0, lambda r: -r["gold_margin"]),
        ("最终 Top-15 但未命中 Top-10", lambda r: r["depth"] == 3 and r["gold_survived"] and r["gold_rank"] > 10, lambda r: r["dataset_index"]),
        ("移除一个竞争候选即可确定恢复", lambda r: r["depth"] > 1 and r["oracle_remove_1_guaranteed"] is True, lambda r: r["dataset_index"]),
        ("并列筛选边界", lambda r: r["first_pruned_here"] and (r["gold_margin"] == 0 or r["prune_class"] == "local_tie_boundary"), lambda r: r["dataset_index"]),
        ("正确完整路径最终第一名", lambda r: r["depth"] == 3 and r["gold_rank"] == 1, lambda r: r["dataset_index"])]
    for label, predicate, key in criteria:
        candidates = [r for r in records if r["sample_id"] not in used and predicate(r)]
        if not candidates:
            missing.append(label)
            continue
        row = min(candidates, key=key)
        used.add(row["sample_id"])
        selected.append({"category": label, "sample_id": row["sample_id"], "tensor_shard": row["tensor_shard"], "tensor_row": row["tensor_row"]})
    return selected, missing


def case_text(cases, records, raw_root):
    by_id = {}
    for row in records:
        by_id.setdefault(row["sample_id"], []).append(row)
    lines = []
    grouped = {}
    for case in cases:
        grouped.setdefault(case["tensor_shard"], []).append(case)
    material = {}
    for shard, shard_cases in grouped.items():
        # Case display needs only compact recorded paths/scores, not full logits.
        with np.load(raw_root / shard, allow_pickle=False) as data:
            values = {key: data[key] for key in ("parent_prefix_tokens_with_start", "beam_scores_before",
                "selected_parent_indices", "selected_token_ids", "selected_cumulative_scores")}
        for case in shard_cases:
            material[case["sample_id"]] = {key: value[case["tensor_row"]].copy() for key, value in values.items()}
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    rank = lambda lo, hi: "—" if lo is None else str(lo) if lo == hi else f"{lo}～{hi}"
    for number, case in enumerate(cases, 1):
        rows = by_id[case["sample_id"]]
        value = material[case["sample_id"]]
        final = rows[-1]
        lines.extend([f'### 案例 {number}：{case["category"]} — {case["sample_id"]}', "",
            f'目标 Offset Token：`{final["target_tokens"]}`；原始 SID：`{(np.asarray(final["target_tokens"]) - [1, 257, 513]).tolist()}`。来源：`{case["tensor_shard"]}` 第 {case["tensor_row"]} 行。', "",
            "| 层 | Gold 父路径存活 | 本层 Gold 存活/排名 | 局部概率排名 | 实际全局排名/区间 | Gold 候选累计分数 | Top-15 阈值 | Margin |",
            "|---|---|---|---|---|---:|---:|---:|"])
        for row in rows:
            lines.append(f'| D{row["depth"]} | {row["gold_parent_alive"]} | {row["gold_rank"] if row["gold_survived"] else "未存活"} | '
                f'{rank(row["gold_local_rank_min"], row["gold_local_rank_max"])} | {rank(row["gold_global_rank_min"], row["gold_global_rank_max"])} | '
                f'{fmt(row["gold_candidate_score"])} | {fmt(row["top15_threshold"])} | {fmt(row["gold_margin"])} |')
        for row in rows:
            j, d = row["depth"] - 1, row["depth"]
            parents = value["parent_prefix_tokens_with_start"][j, :, 1:d]
            pidx = value["selected_parent_indices"][j].astype(np.int64)
            paths = np.concatenate([parents[pidx], value["selected_token_ids"][j, :, None]], axis=1)
            parent_origins = deviation(parents, row["target_tokens"][:j])
            fanout = np.bincount(pidx, minlength=K)
            wrong = np.flatnonzero(parent_origins > 0)
            shown = set(range(5))
            if row["gold_rank"] is not None:
                shown.add(row["gold_rank"] - 1)
            if len(wrong):
                largest = int(wrong[np.argmax(fanout[wrong])])
                shown.update(np.flatnonzero(pidx == largest)[:3].tolist())
            lines.extend(["", f'**D{d} 关键真实保留路径**：正确父路径占 {row["gold_parent_child_slots"]}/15，已有非目标父路径占 {row["non_target_parent_slots"]}/15；单个非目标父路径最多产生 {row["max_children_one_non_target_parent"]} 条子路径。', "",
                "| 子 Beam 排名 | 父 Beam 排名 | 父前缀 | 子前缀 | 首次偏离层（0=匹配） | 父累计分数 | 子累计分数 | 子分数 − Gold |",
                "|---:|---:|---|---|---:|---:|---:|---:|"])
            for child in sorted(shown):
                parent = int(pidx[child])
                score = float(value["selected_cumulative_scores"][j, child])
                gap = score - row["gold_candidate_score"] if row["gold_candidate_score"] is not None else None
                lines.append(f'| {child + 1} | {parent + 1} | `{parents[parent].tolist()}` | `{paths[child].tolist()}` | '
                    f'{row["selected_first_deviation_depths"][child]} | {fmt(float(value["beam_scores_before"][j, parent]))} | {fmt(score)} | {fmt(gap)} |')
            if row["first_pruned_here"] and d > 1:
                lines.append(f'\n本层分类 `{row["prune_class"]}`；其他父路径有 {row["foreign_candidates_above_gold"]} 个严格高于 Gold 的扩展。'
                    + "Oracle 移除 1/3/5/全部后的确定/可能恢复："
                    + "，".join(f'{m}={row[f"oracle_remove_{m}_guaranteed"]}/{row[f"oracle_remove_{m}_possible"]}' for m in (1, 3, 5, "all")) + "。")
        outcome = f'完整 SID 最终第 {final["gold_rank"]} 名，Recall@10={final["hit_at_10"]}' if final["gold_survived"] else f'首次在 D{final["first_prune_depth"]} 被剪枝'
        lines.extend(["", f"最终结果：{outcome}。", ""])
    return lines


def render_report(summary, records, raw_root, output_dir):
    n = summary["samples"]
    depths = summary["depths"]
    percent = lambda x: "—" if x is None else f"{x * 100:.4f}%"
    value = lambda x: "—" if x is None else f"{x:.4f}"
    lines = ["# 原始 Beauty TIGER：自回归非目标路径级联分析", "",
        f'范围：原版 TIGER，Validation {n:,}/{summary["validation_total"]:,} 条，Beam=15，三位 SID。状态范围为 `{summary["scope"]}`。仅离线读取真实 JSONL/NPZ 和已有原版诊断，没有加载 T5、训练、使用 Test 或改变生成规则。', "",
        "## 研究问题与方法", "",
        "研究非目标前缀能否延续、保持相对优势、占据 Beam 名额，以及当前一步 Gold 剪枝是否对这些竞争者敏感。非目标仅指不匹配本次观测目标，不等于商品不相关。", "",
        "复用原始 raw_utils.verify_tensors 校验真实 Top-15 分数多重集、原始索引、所有父子路径和最终路径。候选分数严格使用 FP32 父累计分数 + 当前真实父 Beam 下的 Log Probability；保留全部 769 维候选，不增加位置或商品 Catalog 过滤。", "",
        "局部排名根据真实 Gold 父 Beam 的概率计算。全局排名根据本轮所有有效父 Beam 扩展计算。已保留路径的排名以原始 CUDA 选择顺序为准；未保留路径只给出严格大于/等分得到的秩区间。父路径已经消失后的 Gold 排名、分数和 Margin 为空，未用 Teacher Forcing 补齐。", "",
        "## 逐层丢失过程", "",
        "| 层 | 正确前缀存活数 | 累计存活率 | 条件存活率 | 首次剪枝数 | 占全部样本 |",
        "|---|---:|---:|---:|---:|---:|"]
    for d, result in depths.items():
        lines.append(f'| D{d} | {result["survival_count"]} | {percent(result["survival_rate"])} | {percent(result["conditional_survival_rate"])} | {result["first_prune_count"]} | {percent(result["first_prune_rate_all_samples"])} |')
    final = summary["final_gold_rank_groups"]
    lines.extend(["", f'最终正确完整 SID：Top-5={final["top1_5"]}，第 6～10 名={final["rank6_10"]}，第 11～15 名={final["rank11_15"]}。'
        f'前三层提前剪枝合计 {sum(summary["first_prune_counts"][str(d)] for d in (1, 2, 3))} 条；另有 {final["rank11_15"]} 条完整路径保留却无法命中 Recall@10。', "",
        "| 层 | 存活时排名 1～5 / 6～10 / 11～15 | 首剪 Margin P10 / P50 / P90 | 首剪距阈值 ≤0.01 / ≤0.05 / ≤0.1 |", "|---|---|---|---|"])
    for d, result in depths.items():
        ranks = result["gold_rank_distribution_when_survived"]
        bins = [sum(ranks[str(i)] for i in range(lo, hi + 1)) for lo, hi in ((1, 5), (6, 10), (11, 15))]
        margin = result["gold_margin_first_pruned"]
        near = result["near_miss_counts_first_pruned"]
        lines.append(f'| D{d} | {bins} | ' + " / ".join(value(margin.get(q)) for q in ("p10", "p50", "p90"))
            + " | " + " / ".join(str(near[str(e)]) for e in (0.01, 0.05, 0.1)) + " |")
    lines.extend(["", "## 局部预测与跨父路径竞争", "",
        "A：局部排名下界 >15，确定局部预测差；B：局部排名上界 ≤15，但实际全局未保留；跨越第 15 名的局部等分单列。百分比分母为该层首次剪枝样本。A 也可能受到跨父竞争，A/B 是观察分类，不是训练原因的因果分类。", "",
        "| 首剪层 | 分母 | A：局部差 | B：局部不差但全局淘汰 | 局部边界并列 |", "|---|---:|---:|---:|---:|"])
    for d in ("2", "3"):
        result = depths[d]
        cells = [f'{entry["count"]}（{percent(entry["fraction_first_pruned"])}）' for entry in result["prune_classes"].values()]
        lines.append(f'| D{d} | {result["first_prune_count"]} | ' + " | ".join(cells) + " |")
    b = sum(depths[d]["prune_classes"]["B_cross_parent"]["count"] for d in ("2", "3"))
    lines.extend(["", f'合计 {b} 条属于 B，占全部 Validation 的 {percent(b / n)}。D1 的失败另列为第一步预测/筛选失败；最终 11～15 名属于排名失败。按首次失败位置和最终结果计数互斥，但非目标分支竞争、局部预测困难及分数并列在机制上可同时出现。', "",
        "## 分支延续与 Beam 占位", "",
        "优先看 Gold 父路径仍存活的样本，避免把 Gold 早已消失后剩余全是非目标路径当成竞争证据。来源 1/2/3 表示首次偏离目标的深度；0 表示仍匹配。每行分母为样本数 ×15。", "",
        "此时实际有 14 个非目标父 Beam 和 1 个正确父 Beam，总占位比例有结构性的基数因素，不能单凭比例高认定异常挤占或因果影响。14/15 也不是推荐目标或严格统计基线；需要结合局部排名、父子延续和固定分数的干预结果。", "",
        "| 层/分组 | 样本数 | 来自已偏离父路径的名额占比 | 首次偏离 D1/D2/D3 的平均名额 | 不同父 Beam 均值 | 单个非目标父最大子数均值 |", "|---|---:|---:|---|---:|---:|"])
    for d in ("2", "3"):
        for group in ("gold_parent_alive", "gold_survived", "first_pruned"):
            stat = depths[d]["occupancy"][group]
            means = [stat["slots_by_first_deviation"][str(o)] / stat["samples"] if stat["samples"] else None for o in (1, 2, 3)]
            lines.append(f'| D{d}/{group} | {stat["samples"]} | {percent(stat["non_target_parent_slot_share"])} | '
                + " / ".join(value(x) for x in means) + f' | {value(stat["distinct_parents"].get("mean"))} | {value(stat["max_children_one_non_target_parent"].get("mean"))} |')
    lines.extend(["", "| 层（Gold 父路径存在） | 已有非目标父分支数 | 至少产生一条存活子路径 | 产生多条子路径 | Top-5 非目标父至少一条子路径仍为 Top-5 |", "|---|---:|---:|---:|---|"])
    for d in ("2", "3"):
        stat = depths[d]["occupancy"]["gold_parent_alive"]
        lines.append(f'| D{d} | {stat["non_target_parents"]} | {stat["non_target_parents_with_children"]}（{percent(stat["non_target_parent_continuation_rate"])}） | '
            f'{stat["non_target_parents_with_multiple_children"]} | {stat["non_target_top5_parents_continued_in_top5"]}/{stat["non_target_top5_parents"]}（{percent(stat["non_target_top5_parent_continuation_rate"])}） |')
    lines.extend(["", "最终全部样本的前 K 名来源计数（包含 Gold 先前已被剪枝的样本，仅描述最终组成）：", "",
        "| K | 匹配目标 | D1 首次偏离 | D2 首次偏离 | D3 首次偏离 |", "|---|---:|---:|---:|---:|"])
    for k, origins in summary["final_non_target_slots_by_origin"].items():
        lines.append(f'| {k} | ' + " | ".join(str(origins[str(o)]) for o in range(4)) + " |")
    lines.extend(["", "## 相对分数与连续竞争", "",
        "比较同层分数，以及同一父子分支的（非目标 − Gold）相对差变化，不把累计 Log Probability 的绝对上升当作级联。下表只计 Gold 父路径存在时、来自已经偏离目标父路径的实际存活子路径。扩大/缩小指该差值增加/减少（容差1e-6）；扩大也可能只是落后减少，并不一定已经领先。真正的连续领先另用父层与子层差值都严格大于0计数。", "",
        "| 层/分组 | 延续子路径数 | 优势扩大 / 缩小 / 近似不变 | 父层和子层均严格领先 Gold | 有连续领先竞争者的样本 |", "|---|---:|---|---:|---:|"])
    for d in ("2", "3"):
        for group in ("gold_parent_alive", "gold_survived", "first_pruned"):
            stat = depths[d]["continuing_branch_advantage"][group]
            lines.append(f'| D{d}/{group} | {stat["continued_paths"]} | {stat["expanded_relative_advantage"]} / {stat["shrunk_relative_advantage"]} / {stat["unchanged_relative_advantage"]} | '
                f'{stat["leads_parent_and_child_paths"]} | {stat["samples_with_continuous_leading_competitors"]}/{stat["samples"]} |')
    stat = depths["3"]["continuing_branch_advantage"]["gold_parent_alive"]
    lines.extend(["", f'D3 扩展前 Gold 父路径仍存在的样本中，{stat["samples_with_d1_wrong_leading_all_three"]} 条样本有 D1 就偏离的真实父子链连续三层严格领先 Gold；对应最终路径 {stat["d1_wrong_paths_lead_all_three"]} 条。D1 非目标链连续三层位于 Top-5 的最终路径有 {stat["d1_wrong_paths_in_top5_all_three"]} 条。', "",
        "## 控制局部预测难度后的占位比较", "",
        "局部排名分组按严格大于 Gold 的概率个数得到的排名下界；边界并列仍单列在 JSON。这里只做描述，不据此认定集中占位造成剪枝。完整 Margin 分组、15×15 父子排名转移矩阵和分布均保存在汇总 JSON。", "",
        "| 层 | 局部排名组 | 存活/剪枝样本数 | 非目标父名额占比：存活/剪枝 | 单个非目标父最大子数均值：存活/剪枝 |", "|---|---|---|---|---|"])
    for d in ("2", "3"):
        for group in ("1_5", "6_15", "16_50", "51_plus"):
            a = depths[d]["controlled_comparisons"][group]
            yes, no = a["survived"], a["pruned"]
            lines.append(f'| D{d} | {group} | {yes["samples"]}/{no["samples"]} | {percent(yes["non_target_parent_slot_share"])}/{percent(no["non_target_parent_slot_share"])} | '
                f'{value(yes["max_children_one_non_target_parent"].get("mean"))}/{value(no["max_children_one_non_target_parent"].get("mean"))} |')
    lines.extend(["", "## 单步 Oracle 竞争检查", "",
        "只在 D2/D3 首次剪枝且 Gold 父路径真实存在时，移除来自已偏离父路径、严格高于 Gold 的最高分 1/3/5 个候选；不足时移除实际数量。原分数完全不变。全部移除则排除所有非目标父路径的扩展。不是删除正确父路径下的错误 Token。", "",
        "确定恢复：剩余 ≥Gold 的候选总数 ≤15；可能恢复：严格 >Gold 的剩余候选少于15，但边界等分次序未知。区间下限为任何等分次序都能恢复的数量，上限含可能恢复的并列样本。若有限移除没有删掉任何候选，沿用实际未存活结果，不能靠重排并列制造恢复。", "",
        "| 首剪层 | 分母 | 移除1：确定～可能 | 移除3 | 移除5 | 排除全部非目标父扩展 |", "|---|---:|---|---|---|---|"])
    for d in ("2", "3"):
        result = depths[d]
        cells = [f'{result["oracle"][str(m)]["guaranteed_rescued"]}～{result["oracle"][str(m)]["possible_rescued_including_guaranteed"]}' for m in (1, 3, 5, "all")]
        lines.append(f'| D{d} | {result["first_prune_count"]} | ' + " | ".join(cells) + " |")
    oracle_totals = {str(m): {bound: sum(depths[d]["oracle"][str(m)][bound] for d in ("2", "3"))
        for bound in ("guaranteed_rescued", "possible_rescued_including_guaranteed")} for m in (1, 3, 5, "all")}
    lines.extend(["", "## 证据支持什么、不能证明什么", "",
        f'1. **直接观察**：真实轨迹能统计非目标路径的跨层延续和名额占用。在 Gold 父路径仍存在的 D2/D3 样本中分别延续 '
        f'{depths["2"]["occupancy"]["gold_parent_alive"]["non_target_parent_slots"]} / {depths["3"]["occupancy"]["gold_parent_alive"]["non_target_parent_slots"]} 条已偏离父路径的子分支。',
        f'2. **局部与全局竞争**：{b} 条 B 类样本的 Gold Token 局部排名上界不超过15，但未进入实际全局 Top-15。A 类、D1 初始失败和最终第11～15名不能合并归因为级联。',
        f'3. **固定分数下的竞争敏感性**：移除 1/3/5 个高分非目标父扩展后，确定恢复数分别为 '
        f'{oracle_totals["1"]["guaranteed_rescued"]}/{oracle_totals["3"]["guaranteed_rescued"]}/{oracle_totals["5"]["guaranteed_rescued"]}。'
        '这些是单步、标签已知的 Oracle 结果，不是 Recall@10 提升，也不代表重跑后续 Decoder 后仍能成功。',
        '4. **因果边界**：观察和单步重排不能证明 Teacher Forcing 或 Exposure Bias 是根本原因；局部概率质量、累计父分数、有限 Beam、并列以及最终排名均可能参与。非目标也不等于不相关商品。',
        ('5. **是否值得进入方法研究**：本次出现跨父竞争和可确定的单步恢复，支持继续研究无需标签的风险信号与竞争控制；仍须独立验证收益及代价，不能直接部署 Oracle。'
            if b and oracle_totals["5"]["guaranteed_rescued"] else
            '5. **是否值得进入方法研究**：本次结果尚不足以据单步恢复支持该方向，应先检查局部预测、最终排名及并列限制。'), "",
        "## 真实案例", "", "每例展示全部三层的 Gold 状态和关键保留路径：实际 Top-5、存活 Gold，以及高占位非目标父路径的部分子路径。原始分片可查询剩余保留路径；没有复制词表矩阵。", ""])
    lines.extend(case_text(summary["cases"], records, raw_root))
    if summary["case_categories_without_additional_unique_sample"]:
        lines.extend(["本范围内未找到额外独立样本的案例类别：" + "、".join(summary["case_categories_without_additional_unique_sample"]) + "。", ""])
    lines.extend(["## 输出与复现", "",
        f'结果目录：`{output_dir}`。`cascade_analysis_samples.parquet` 一行是 `sample_id + depth`，共 {len(records):,} 行。'
        '保留 Gold 状态、秩区间、分数、竞争者计数、名额统计、紧凑父索引/首次偏离深度及 Oracle 上下界，不保存完整词表矩阵。', "",
        "```powershell", "# 先用新的目录跑 100 条；通过后把 --max-samples 100 改为 --all", "$env:CUDA_VISIBLE_DEVICES='-1'",
        'python -m tiger_prefix_ft.cascade_analysis `',
        "  --max-samples 100 --output-dir out/tiger_prefix_ft/cascade_repeat100 `",
        "  --report-path docs/generated/tiger_prefix_ft_beauty_20261013/cascade_repeat100_report.md", "```", ""])
    summary["oracle_totals_d2_d3"] = oracle_totals
    question_titles = ("非目标前缀是否会持续扩展", "在 D2/D3 占据多少名额", "与 Gold 剪枝有什么关系",
        "有多少局部不差却全局被淘汰", "提前剪枝与最终排序分别多少", "是否持续具有相对优势",
        "单步 Oracle 能说明什么", "是否支持下一步研究")
    introduction = ["## 八个问题的直接回答", ""]
    for index, (title, answer) in enumerate(zip(question_titles, summary["conclusions"]), 1):
        introduction.extend([f"{index}. **{title}**：{answer}", ""])
    position = lines.index("## 研究问题与方法")
    lines[position:position] = introduction
    return "\n".join(lines)


def run(args):
    started = time.perf_counter()
    helpers = raw_helpers()
    import torch
    torch.set_num_threads(4)
    assert not args.output_dir.exists() and not args.report_path.exists(), "Use new output paths"
    manifest = json.loads((args.raw_root / "collection_manifest.json").read_text())
    assert manifest["state"] == "complete"
    inventory = {entry["path"]: entry for entry in json.loads((args.raw_root / "download_inventory.json").read_text())["files"]}
    shards = [entry for entry in manifest["shards"] if entry["split"] == "valid"]
    total_validation = sum(entry["samples"] for entry in shards)
    assert total_validation == 22363 and len(shards) == 22
    assert len(list((args.raw_root / "valid").glob("shard_*.npz"))) == len(shards)
    assert len(list((args.raw_root / "valid").glob("shard_*.jsonl.gz"))) == len(shards)
    sids = np.load(args.raw_root / "metadata/item_semantic_ids.npy", allow_pickle=False)
    reference = {r["sample_id"]: r for r in pq.read_table(args.diagnosis / "beam_diagnosis_samples.parquet",
        filters=[("model_name", "==", "original")]).to_pylist()}
    assert len(reference) == total_validation
    count = total_validation if args.all else min(args.max_samples, total_validation)
    assert count > 0
    args.output_dir.mkdir(parents=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    records, verified, sample_ids = [], [], []
    for shard in shards:
        if len(sample_ids) >= count:
            break
        paths = {"npz": next(f["path"] for f in shard["files"] if f["path"].endswith(".npz")),
                 "json": next(f["path"] for f in shard["files"] if f["path"].endswith(".jsonl.gz"))}
        for relative in paths.values():
            entry = inventory[relative]
            assert (args.raw_root / relative).stat().st_size == entry["bytes"]
            assert helpers.sha256(args.raw_root / relative) == entry["sha256"]
        with gzip.open(args.raw_root / paths["json"], "rt", encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream]
        assert len(metadata) == shard["samples"]
        take = min(len(metadata), count - len(sample_ids))
        with np.load(args.raw_root / paths["npz"], allow_pickle=False) as source:
            assert len(source["dataset_indices"]) == len(metadata)
            data = {key: source[key][:take].copy() for key in KEYS}
        integrity = helpers.verify_tensors(data)
        for i, sample in enumerate(metadata[:take]):
            assert sample["tensor_shard"] == paths["npz"] and sample["tensor_row"] == i
            assert sample["dataset_index"] == len(sample_ids)
            assert sample["sample_id"] == f'valid_{len(sample_ids):07d}'
            assert np.array_equal(sids[sample["target_item_id"]], sample["target_sid"])
            assert np.array_equal(sids[sample["history_item_ids"]], sample["history_sid"])
            history = (np.asarray(sample["history_sid"]) + [1, 257, 513]).reshape(-1).tolist()
            assert history == [t for t, mask in zip(sample["input_ids"], sample["attention_mask"]) if mask]
            rows, _ = inspect_sample(sample, data, i)
            ref = reference[sample["sample_id"]]
            for row in rows:
                d = row["depth"]
                assert row["target_tokens"] == ref["target_tokens"]
                assert row["gold_survived"] == ref[f"survived_depth{d}"]
                assert row["gold_rank"] == ref[f"gold_rank_depth{d}"]
                assert row["first_prune_depth"] == ref["first_prune_depth"]
                for key, reference_key in (("gold_candidate_score", f"gold_candidate_depth{d}"), ("gold_margin", f"gold_margin_depth{d}")):
                    actual, expected = row[key], ref[reference_key]
                    assert (actual is None) == (expected is None)
                    assert actual is None or abs(actual - expected) < 2e-5
            records.extend(rows)
            sample_ids.append(sample["sample_id"])
        verified.append({"npz": paths["npz"], "json": paths["json"], "samples": take,
                         "sha256_verified": True, "integrity": integrity})
        print(json.dumps({"verified_shard": paths["npz"], "samples_processed": len(sample_ids),
                          "score_abs_error": integrity["max_score_abs_error"]}), flush=True)
        del data, metadata
    assert len(sample_ids) == len(set(sample_ids)) == count
    summary = summarize(records, total_validation)
    if args.all:
        assert [summary["depths"][str(d)]["survival_count"] for d in (1, 2, 3)] == [13469, 3927, 2422]
        assert summary["first_prune_counts"] == {"0": 2422, "1": 8894, "2": 9542, "3": 1505}
        assert summary["final_gold_rank_groups"] == {"top1_5": 1333, "rank6_10": 662, "rank11_15": 427}
    summary["cases"], summary["case_categories_without_additional_unique_sample"] = choose_cases(records)
    if args.all:
        assert 8 <= len(summary["cases"]) <= 12
    summary.update(source_checkpoint=manifest["settings"]["sources"]["tiger_checkpoint"],
        verified_shards=verified, original_diagnosis_exact_for_survival_and_rank=True,
        code_sha256=helpers.sha256(Path(__file__)), raw_helper_sha256=helpers.sha256(Path(helpers.__file__)))
    sample_path = args.output_dir / "cascade_analysis_samples.parquet"
    pq.write_table(pa.Table.from_pylist(records), sample_path, compression="zstd")
    assert pq.ParquetFile(sample_path).metadata.num_rows == count * 3
    assert pq.read_table(sample_path).to_pylist() == records
    text = render_report(summary, records, args.raw_root, args.output_dir)
    args.report_path.write_text(text, encoding="utf-8")
    summary.update(state="complete", sample_rows=len(records), sample_file_sha256=helpers.sha256(sample_path),
                   report=str(args.report_path), seconds=time.perf_counter() - started)
    write_json(args.output_dir / "cascade_analysis_summary.json", summary)
    print(json.dumps({"state": "complete", "samples": count, "sample_rows": len(records),
                      "output_dir": str(args.output_dir), "report": str(args.report_path),
                      "seconds": summary["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--diagnosis", type=Path, default=DIAG)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-samples", type=int, default=100)
    limits.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
