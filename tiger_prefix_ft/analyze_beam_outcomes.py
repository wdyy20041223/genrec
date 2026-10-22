"""Offline analysis of the saved 2026-10-15 Beam-width experiments; no model run."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out/tiger_prefix_ft"
REPORT = ROOT / "docs/generated/tiger_prefix_ft_beauty_20261013"
FULL = BASE / "beauty_beam_30_30_30_eval_20261015/beam_evaluation_samples.parquet"
MIDDLE = BASE / "beauty_beam_15_30_15_eval_20261015/beam_evaluation_samples.parquet"
TINY = BASE / "beauty_beam_30_30_30_inference_20261015/predictions.jsonl"
OUTCOMES = ("d3_pruned", "rank16_30", "rank11_15", "top10")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(row):
    if not row["survived_depth3"]:
        return "d3_pruned"
    rank = row["gold_rank_depth3"]
    assert rank is not None and 1 <= rank <= 30
    return "top10" if rank <= 10 else "rank11_15" if rank <= 15 else "rank16_30"


def counts(rows):
    observed = Counter(outcome(row) for row in rows)
    return {key: observed[key] for key in OUTCOMES}


def index(rows, split, model, widths):
    result = {}
    for row in rows:
        if (row["split"], row["model_name"]) != (split, model):
            continue
        sid = row["sample_id"]
        assert sid.startswith(split + "_") and sid not in result
        assert len(row["target_tokens"]) == 3
        previous_survived, previous_score = True, 0.0
        first_prune = 0
        for d in (1, 2, 3):
            survived = row[f"survived_depth{d}"]
            rank, score = row[f"gold_rank_depth{d}"], row[f"gold_score_depth{d}"]
            candidate = row[f"gold_candidate_depth{d}"]
            assert not survived or previous_survived
            assert (rank is not None) == survived == (score is not None)
            assert (candidate is not None) == previous_survived
            if survived:
                assert 1 <= rank <= widths[d - 1] and math.isfinite(score)
                assert score == candidate and score <= previous_score + 1e-5
                previous_score = score
            elif not first_prune:
                first_prune = d
            previous_survived = survived
        assert row["first_prune_depth"] == first_prune
        assert row["hit_at_10"] == (row["survived_depth3"] and row["gold_rank_depth3"] <= 10)
        result[sid] = row
    assert len(result) == 22363
    return result


def detail(split, a, b, m):
    row = {"split": split, "sample_id": a["sample_id"], "target_tokens": a["target_tokens"],
           "baseline_first_prune_depth": a["first_prune_depth"], "beam30_outcome": outcome(b),
           "middle_first_prune_depth": m["first_prune_depth"]}
    for name, record in (("baseline", a), ("beam30", b), ("middle", m)):
        for d in (1, 2, 3):
            for kind in ("rank", "score"):
                row[f"{name}_gold_{kind}_depth{d}"] = record[f"gold_{kind}_depth{d}"]
    row["final_gold_score_delta"] = (b["gold_score_depth3"] - a["gold_score_depth3"]
                                      if a["survived_depth3"] and b["survived_depth3"] else None)
    return row


def write_csv(name, rows):
    with (REPORT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def main():
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in (FULL, MIDDLE, TINY)}
    rows, middle_rows = pq.read_table(FULL).to_pylist(), pq.read_table(MIDDLE).to_pylist()
    assert len(rows) == len(middle_rows) == 89452
    assert {r["model_name"] for r in rows} == {"original_free", "beam_30_30_30"}
    assert {r["model_name"] for r in middle_rows} == {"original_free", "beam_15_30_15"}
    assert {r["split"] for r in rows + middle_rows} == {"valid", "test"}
    predictions = {r["sample_id"]: r for r in map(json.loads, TINY.read_text(encoding="utf-8").splitlines())}
    report = {"source_sha256": sources, "no_model_run": True, "splits": {},
              "competitor_identity_limit": "Full-run Parquet stores gold statistics only, not other final candidate SID paths/scores. Original Beam-15 trajectories cannot identify Beam-30 competitors.",
              "middle_profile": "15-30-15: same D1 width as baseline, independently saved evaluation",
              "score_note": "Same weights; small FP16 numerical differences exist across Beam widths. Rank changes do not identify specific competing SID paths."}
    rescued_details, lost_details = [], []
    for split in ("valid", "test"):
        a = index(rows, split, "original_free", (15, 15, 15))
        b = index(rows, split, "beam_30_30_30", (30, 30, 30))
        m = index(middle_rows, split, "beam_15_30_15", (15, 30, 15))
        old_middle = index(middle_rows, split, "original_free", (15, 15, 15))
        assert a == old_middle and a.keys() == b.keys() == m.keys()
        assert all(a[k]["target_tokens"] == b[k]["target_tokens"] == m[k]["target_tokens"] for k in a)
        rescued = [k for k in a if not a[k]["survived_depth2"] and b[k]["survived_depth2"]]
        lost = [k for k in a if a[k]["hit_at_10"] and not b[k]["hit_at_10"]]
        gained = [k for k in a if not a[k]["hit_at_10"] and b[k]["hit_at_10"]]
        fate = counts([b[k] for k in rescued])
        assert sum(fate.values()) == len(rescued)
        deltas = [b[k]["gold_score_depth3"] - a[k]["gold_score_depth3"] for k in lost]
        assert all(b[k]["survived_depth3"] for k in lost)
        denominator = len(rescued) - fate["top10"]
        old_hits = sum(r["hit_at_10"] for r in a.values())
        new_hits = sum(r["hit_at_10"] for r in b.values())
        assert new_hits - old_hits == len(gained) - len(lost)
        summary = {
            "samples": len(a), "new_d2_count": len(rescued), "new_d2_outcome_counts": fate,
            "new_d2_outcome_percent": {k: v / len(rescued) * 100 for k, v in fate.items()},
            "new_d2_by_baseline_first_prune_depth": {str(d): counts([b[k] for k in rescued if a[k]["first_prune_depth"] == d]) for d in (1, 2)},
            "new_d2_fail_top10": denominator,
            "d3_pruned_share_of_new_d2_failures_percent": fate["d3_pruned"] / denominator * 100,
            "survived_outside_top10_share_of_new_d2_failures_percent": (fate["rank16_30"] + fate["rank11_15"]) / denominator * 100,
            "old_d2_survivors_lost": sum(a[k]["survived_depth2"] and not b[k]["survived_depth2"] for k in a),
            "lost_top10_count": len(lost), "lost_top10_outcomes": counts([b[k] for k in lost]),
            "lost_top10_baseline_rank_counts": dict(sorted(Counter(a[k]["gold_rank_depth3"] for k in lost).items())),
            "lost_top10_beam30_rank_counts": dict(sorted(Counter(b[k]["gold_rank_depth3"] for k in lost).items())),
            "lost_top10_with_d2_rank_above15": sum(b[k]["gold_rank_depth2"] > 15 for k in lost),
            "lost_top10_already_lost_with_middle_profile": sum(not m[k]["hit_at_10"] for k in lost),
            "lost_top10_gold_score_delta": {"min": min(deltas), "median": median(deltas), "max": max(deltas), "max_abs": max(map(abs, deltas)), "increased_count": sum(v > 0 for v in deltas)},
            "lost_top10_covered_by_saved_full30_predictions": sorted(set(lost) & predictions.keys()),
            "gained_top10_count": len(gained),
            "gained_top10_original_first_prune_counts": dict(sorted(Counter(a[k]["first_prune_depth"] for k in gained).items())),
            "gained_top10_from_new_d2": sum(k in set(rescued) for k in gained),
            "baseline_top10_hits": old_hits, "beam30_top10_hits": new_hits,
            "recall10_absolute_change_percentage_points": (new_hits - old_hits) / len(a) * 100,
            "recall10_relative_change_percent": (new_hits - old_hits) / old_hits * 100,
        }
        for k in rescued:
            rescued_details.append(detail(split, a[k], b[k], m[k]))
        for k in lost:
            lost_details.append(detail(split, a[k], b[k], m[k]))
        report["splits"][split] = summary
    assert sources == {str(p.relative_to(ROOT)): sha256(p) for p in (FULL, MIDDLE, TINY)}
    report.update(state="passed", source_files_unchanged=True,
                  new_d2_detail_rows=len(rescued_details), lost_top10_detail_rows=len(lost_details))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv("beam30_new_d2_outcomes_20261015.csv", rescued_details)
    write_csv("beam30_lost_top10_20261015.csv", lost_details)
    (REPORT / "beam30_outcomes_20261015.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
