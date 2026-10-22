"""CPU-only SID prefix analysis of an existing, aligned TIGER offline backup."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from threadpoolctl import threadpool_limits


def lcp_length(sid_i, sid_j):
    """Exact LCP of the first three semantic codes; ignore any suffix."""
    left, right = np.asarray(sid_i)[..., :3], np.asarray(sid_j)[..., :3]
    if left.shape != right.shape or left.shape[-1] != 3:
        raise ValueError("Both SID arrays must have the same shape and at least three codes")
    return np.logical_and.accumulate(left == right, axis=-1).sum(axis=-1)


def boundaries(values):
    if len(values) == 0:
        return np.array([0], dtype=np.int64)
    changed = np.any(values[1:] != values[:-1], axis=1)
    return np.r_[0, np.flatnonzero(changed) + 1, len(values)]


@dataclass
class PairSpace:
    """A bijection from [0,total) onto unordered pairs of one exact LCP."""
    order: np.ndarray
    left_positions: np.ndarray
    right_starts: np.ndarray
    cumulative_counts: np.ndarray

    @property
    def total(self):
        return int(self.cumulative_counts[-1]) if len(self.cumulative_counts) else 0

    def pairs_at(self, ranks):
        ranks = np.asarray(ranks, dtype=np.int64)
        if len(ranks) and (ranks.min() < 0 or ranks.max() >= self.total):
            raise ValueError("Pair rank is outside the population")
        rows = np.searchsorted(self.cumulative_counts, ranks, side="right")
        previous = np.r_[0, self.cumulative_counts[:-1]]
        right = self.right_starts[rows] + ranks - previous[rows]
        left_items = self.order[self.left_positions[rows]]
        right_items = self.order[right]
        return np.column_stack((np.minimum(left_items, right_items), np.maximum(left_items, right_items)))


def build_pair_spaces(sids, item_ids=None):
    """Index O(items) left rows, never enumerate all item pairs."""
    sids = np.asarray(sids)[:, :3]
    if sids.ndim != 2 or sids.shape[1] != 3:
        raise ValueError("Expected an item x semantic-code array")
    ids = np.arange(len(sids), dtype=np.int64) if item_ids is None else np.asarray(item_ids)
    if len(ids) != len(sids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Item identifiers must be unique and aligned")
    permutation = np.lexsort((sids[:, 2], sids[:, 1], sids[:, 0]))
    codes, order = sids[permutation], ids[permutation]
    spaces = []
    for k in range(4):
        left_parts, right_parts, count_parts = [], [], []
        parents = np.array([0, len(sids)]) if k == 0 else boundaries(codes[:, :k])
        for start, stop in zip(parents[:-1], parents[1:]):
            if k == 3:
                positions = np.arange(start, stop - 1, dtype=np.int64)
                left_parts.append(positions)
                right_parts.append(positions + 1)
                count_parts.append(stop - positions - 1)
            else:
                children = boundaries(codes[start:stop, k:k + 1]) + start
                for first, end in zip(children[:-2], children[1:-1]):
                    positions = np.arange(first, end, dtype=np.int64)
                    left_parts.append(positions)
                    right_parts.append(np.full(len(positions), end, dtype=np.int64))
                    count_parts.append(np.full(len(positions), stop - end, dtype=np.int64))
        concat = lambda arrays: np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        spaces.append(PairSpace(order, concat(left_parts), concat(right_parts),
                                np.cumsum(concat(count_parts), dtype=np.int64)))
    if sum(space.total for space in spaces) != len(sids) * (len(sids) - 1) // 2:
        raise AssertionError("LCP groups do not partition all unordered pairs")
    return spaces


def sample_pairs(space, count, rng):
    count = min(int(count), space.total)
    if count < 0:
        raise ValueError("Sample count must be nonnegative")
    ranks = np.arange(space.total, dtype=np.int64) if count == space.total else rng.choice(space.total, count, replace=False)
    pairs = space.pairs_at(ranks)
    if len(pairs):
        assert (pairs[:, 0] < pairs[:, 1]).all()
        assert len(np.unique(ranks)) == len(ranks)
    return pairs


def normalize_embeddings(embeddings):
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 0)
    normalized = np.full_like(values, np.nan)
    normalized[valid] = values[valid] / norms[valid, None]
    return normalized, valid


def semantic_cosine(normalized, item_i, item_j):
    values = np.einsum("ij,ij->i", normalized[item_i], normalized[item_j], optimize=False)
    return np.clip(values, -1.0, 1.0)


def build_train_item_users(sequences, item_count):
    """Use full_seq[:-2], matching the original leave-two-out training split."""
    item_rows, user_columns = [], []
    for user_index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.int64)
        if len(sequence) < 5 or (sequence < 0).any() or (sequence >= item_count).any():
            raise ValueError("Invalid source user sequence")
        train_items = np.unique(sequence[:-2])
        item_rows.extend(train_items)
        user_columns.extend([user_index] * len(train_items))
    values = np.ones(len(item_rows), dtype=np.int32)
    matrix = sparse.coo_matrix((values, (item_rows, user_columns)),
                               shape=(item_count, len(sequences)), dtype=np.int32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    assert np.all(matrix.data == 1)
    return matrix, np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)


def collaborative_cosine(common_users, degree_i, degree_j):
    valid = (degree_i > 0) & (degree_j > 0)
    result = np.full(len(common_users), np.nan)
    denominator = np.sqrt(degree_i[valid].astype(float) * degree_j[valid])
    result[valid] = common_users[valid] / denominator
    return result, valid


def cointeraction_rate(common_users, valid):
    return float(np.mean(common_users[valid] > 0)) if np.any(valid) else float("nan")


def numeric_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    return float(values.mean()), float(np.median(values)), float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_inputs(root):
    names = ["analysis/item_sid.parquet", "analysis/item_semantic_ids.npy",
             "analysis/user_sequences.parquet", "analysis/train_samples.parquet",
             "data/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet"]
    manifest = {entry["path"]: entry for entry in json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]}
    hashes = {}
    for name in names:
        hashes[name] = file_hash(root / name)
        if hashes[name] != manifest[name]["sha256"]:
            raise ValueError("Input does not match the cloud backup: " + name)
    items = pd.read_parquet(root / names[0]).sort_values("item_index").reset_index(drop=True)
    sids = np.load(root / names[1])[:, :3]
    users = pd.read_parquet(root / names[2]).sort_values("user_index").reset_index(drop=True)
    vectors = pd.read_parquet(root / names[-1])
    if not np.array_equal(items.item_index, np.arange(len(items))):
        raise ValueError("Item index mapping is not contiguous from zero")
    if not np.array_equal(items.embedding_row, items.item_index):
        raise ValueError("SID and embedding row mappings differ")
    if not np.array_equal(items.embedding_item_id, vectors.ItemID):
        raise ValueError("Embedding ItemID mapping does not match the SID export")
    if not (items.asin == items.embedding_asin).all() or not items.asin.is_unique:
        raise ValueError("ASIN mapping is ambiguous")
    if not np.array_equal(items[["sid_0", "sid_1", "sid_2"]].to_numpy(), sids):
        raise ValueError("SID mapping and saved SID tensor disagree")
    if not users.reviewer_id.is_unique or not np.array_equal(users.user_index, np.arange(len(users))):
        raise ValueError("User mapping is ambiguous")
    embedding = np.stack(vectors.embedding.to_numpy())
    normalized, embedding_valid = normalize_embeddings(embedding)
    sid_valid = np.isfinite(sids).all(axis=1) & (sids >= 0).all(axis=1) & (sids < 256).all(axis=1)
    eligible = np.flatnonzero(embedding_valid & sid_valid)
    item_users, degrees = build_train_item_users(users.item_indices, len(items))
    # Independently verify the selected training membership against saved samples.
    saved_train = pd.read_parquet(root / names[3], columns=["user_index", "target_item_index"])
    counts = np.array([len(seq) - 3 for seq in users.item_indices])
    expected_users = np.repeat(np.arange(len(users)), counts)
    expected_targets = np.concatenate([seq[1:-2] for seq in users.item_indices])
    if not np.array_equal(saved_train.user_index, expected_users) or not np.array_equal(saved_train.target_item_index, expected_targets):
        raise ValueError("Leave-two-out split does not match the original saved training samples")
    del vectors, embedding, saved_train, expected_users, expected_targets
    metadata = {
        "dataset": "Amazon 2014 Beauty", "source_commit": "9c6f2a918b4f6beaaada1b513b14978c56ba9f6e",
        "data_root": str(root), "input_sha256": hashes,
        "total_items": len(items), "items_with_sid": int(sid_valid.sum()),
        "items_with_semantic_embedding": int(embedding_valid.sum()),
        "items_with_interactions": int((degrees > 0).sum()),
        "items_used_in_evaluation": len(eligible), "users": len(users),
        "deduplicated_train_interactions": int(item_users.nnz),
        "training_samples_verified": int(counts.sum()),
        "embedding_dimension": normalized.shape[1], "semantic_sid_dimension": 3,
        "interaction_split": "full chronological user sequence excluding final validation and test events",
        "mapping_checks": "SID array, zero-based item index, embedding row, one-based ItemID, ASIN, user index and saved train targets verified",
    }
    return items, sids, normalized, eligible, item_users, degrees, metadata


def compute_group(pairs, lcp, normalized, item_users, degrees, batch_size):
    count = len(pairs)
    semantic, collaborative = np.empty(count), np.empty(count)
    common = np.empty(count, dtype=np.int32)
    valid = np.empty(count, dtype=bool)
    for first in range(0, count, batch_size):
        last = min(first + batch_size, count)
        left, right = pairs[first:last].T
        semantic[first:last] = semantic_cosine(normalized, left, right)
        # Rowwise sparse intersections only for these sampled pairs.
        shared = np.asarray(item_users[left].multiply(item_users[right]).sum(axis=1)).ravel()
        common[first:last] = shared
        collaborative[first:last], valid[first:last] = collaborative_cosine(shared, degrees[left], degrees[right])
        if first == 0 or last == count or first // 250000 != last // 250000:
            print(f"LCP={lcp} similarities {last:,}/{count:,}", flush=True)
    return {"pairs": pairs, "semantic": semantic, "collaborative": collaborative,
            "common": common, "valid": valid}


def make_summary(lcp, space, group, degrees, total_valid_pairs):
    mean_sem, median_sem, std_sem = numeric_stats(group["semantic"])
    mean_co, median_co, std_co = numeric_stats(group["collaborative"])
    valid = group["valid"]
    degree_pairs = degrees[group["pairs"]]
    popularity = np.sqrt(degree_pairs[:, 0].astype(float) * degree_pairs[:, 1])
    return {
        "lcp": lcp, "total_possible_pairs": space.total, "sampled_pairs": len(group["pairs"]),
        "sampling_fraction": len(group["pairs"]) / space.total if space.total else float("nan"),
        "mean_semantic_cosine": mean_sem, "median_semantic_cosine": median_sem, "std_semantic_cosine": std_sem,
        "total_possible_valid_collab_pairs": total_valid_pairs,
        "valid_collab_pairs": int(valid.sum()), "mean_collab_cosine": mean_co,
        "median_collab_cosine": median_co, "std_collab_cosine": std_co,
        "collab_zero_rate": float(np.mean(group["common"][valid] == 0)) if valid.any() else float("nan"),
        "cointeraction_rate": cointeraction_rate(group["common"], valid),
        "mean_pair_geometric_train_popularity": float(popularity[valid].mean()) if valid.any() else float("nan"),
        "median_pair_geometric_train_popularity": float(np.median(popularity[valid])) if valid.any() else float("nan"),
    }


def write_pair_details(groups, items, sids, degrees, output, batch_size):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sid_text = np.array(["[" + ",".join(map(str, sid)) + "]" for sid in sids], dtype=object)
    asins = items.asin.to_numpy()
    writer, header = None, True
    try:
        with (output / "sid_similarity_pairs.csv").open("w", encoding="utf-8", newline="") as csv:
            for k, group in enumerate(groups):
                for first in range(0, len(group["pairs"]), batch_size):
                    last = min(first + batch_size, len(group["pairs"]))
                    left, right = group["pairs"][first:last].T
                    frame = pd.DataFrame({
                        "item_i": left, "item_j": right, "asin_i": asins[left], "asin_j": asins[right],
                        "sid_i": sid_text[left], "sid_j": sid_text[right], "lcp": np.full(last - first, k, dtype=np.int8),
                        "semantic_cosine": group["semantic"][first:last],
                        "collaborative_cosine": group["collaborative"][first:last],
                        "common_user_count": np.where(group["valid"][first:last], group["common"][first:last], np.nan),
                        "item_i_interaction_count": degrees[left], "item_j_interaction_count": degrees[right],
                    })
                    frame.to_csv(csv, index=False, header=header, float_format="%.12g", na_rep="NaN")
                    header = False
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output / "sid_similarity_pairs.parquet", table.schema, compression="zstd")
                    writer.write_table(table)
                print(f"LCP={k} pair details saved", flush=True)
            if header:
                columns = ["item_i", "item_j", "asin_i", "asin_j", "sid_i", "sid_j", "lcp", "semantic_cosine",
                           "collaborative_cosine", "common_user_count", "item_i_interaction_count", "item_j_interaction_count"]
                pd.DataFrame(columns=columns).to_csv(csv, index=False)
    finally:
        if writer:
            writer.close()


def sample_correlations(groups):
    lcp = np.concatenate([np.full(len(group["pairs"]), k) for k, group in enumerate(groups)])
    semantic = np.concatenate([group["semantic"] for group in groups])
    collab = np.concatenate([group["collaborative"] for group in groups])
    def calculate(values):
        valid = np.isfinite(values)
        if valid.sum() < 2 or len(np.unique(lcp[valid])) < 2 or len(np.unique(values[valid])) < 2:
            return {"rho": None, "pairs": int(valid.sum())}
        return {"rho": float(stats.spearmanr(lcp[valid], values[valid]).statistic), "pairs": int(valid.sum())}
    return {"scope": "unweighted stratified sampled pairs; not all-pair population correlation",
            "semantic": calculate(semantic), "collaborative": calculate(collab)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from .paths import BASE, ROOT
    parser.add_argument("--data-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/tiger_prefix_ft/sid_similarity")
    parser.add_argument("--num-pairs-per-group", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.num_pairs_per_group <= 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error("Pair cap, batch size and thread count must be positive")
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if root == output or root in output.parents:
        parser.error("Use an analysis directory outside the original experiment backup")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("sid_similarity_summary.csv", "sid_similarity_pairs.csv", "analysis_metadata.json", "sid_similarity_report.md"):
        if (output / name).exists():
            parser.error("Analysis output already exists; choose a new --output-dir: " + name)
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        print("Checking input hashes, item alignment and training-only interactions", flush=True)
        items, sids, normalized, eligible, item_users, degrees, metadata = load_inputs(root)
        print("DATA " + json.dumps(metadata, ensure_ascii=False), flush=True)
        spaces = build_pair_spaces(sids[eligible], eligible)
        valid_ids = eligible[degrees[eligible] > 0]
        valid_spaces = build_pair_spaces(sids[valid_ids], valid_ids)
        groups, rows = [], []
        seeds = np.random.SeedSequence(args.seed).spawn(4)
        sampling_started = time.perf_counter()
        for k, space in enumerate(spaces):
            pairs = sample_pairs(space, args.num_pairs_per_group, np.random.default_rng(seeds[k]))
            if len(pairs) and not np.all(lcp_length(sids[pairs[:, 0]], sids[pairs[:, 1]]) == k):
                raise AssertionError("Sampled pair has the wrong exact LCP")
            print(f"LCP={k} population={space.total:,}; selected={len(pairs):,}", flush=True)
            group = compute_group(pairs, k, normalized, item_users, degrees, args.batch_size)
            groups.append(group)
            rows.append(make_summary(k, space, group, degrees, valid_spaces[k].total))
        metadata["similarity_seconds"] = time.perf_counter() - sampling_started
        summary = pd.DataFrame(rows)
        summary.to_csv(output / "sid_similarity_summary.csv", index=False, float_format="%.12g", na_rep="NaN")
        write_pair_details(groups, items, sids, degrees, output, args.batch_size)
        correlations = sample_correlations(groups)
        make_plots(summary, groups, output)
        metadata.update(
            random_seed=args.seed, num_pairs_per_group=args.num_pairs_per_group,
            batch_size=args.batch_size, cpu_thread_limit=args.threads,
            total_sampled_pairs=sum(len(group["pairs"]) for group in groups),
            sampling="Uniform without replacement in each exact LCP group using a bijective pair rank space",
            semantic_precision="float64 normalized original Sentence-T5 vectors",
            std_definition="sample standard deviation, ddof=1; fewer than two observations => NaN",
            correlations=correlations, created_at=datetime.now(timezone.utc).isoformat(),
            runtime_versions={"python": sys.version.split()[0], **{
                name: sys.modules[name].__version__
                for name in ("numpy", "pandas", "scipy", "pyarrow", "matplotlib")
            }},
            total_seconds=time.perf_counter() - started,
        )
        try:
            import psutil
            memory = psutil.Process().memory_info()
            metadata["peak_working_set_mib"] = getattr(memory, "peak_wset", memory.rss) / 1024 ** 2
        except ImportError:
            metadata["peak_working_set_mib"] = None
        write_report(summary, metadata, output, args)
        metadata["output_files"] = {
            name: {"bytes": (output / name).stat().st_size, "sha256": file_hash(output / name)}
            for name in ("sid_similarity_summary.csv", "sid_similarity_pairs.csv", "sid_similarity_pairs.parquet",
                         "lcp_similarity_comparison.png", "lcp_cointeraction_rate.png", "lcp_similarity_distribution.png", "sid_similarity_report.md")
            if (output / name).exists()
        }
        metadata["source_files"] = {
            name: {"sha256": file_hash(Path(__file__).resolve().parent / name)}
            for name in ("analyze_sid_similarity.py", "test_sid_similarity.py")
        }
        metadata["elapsed_through_output_hashes_seconds"] = time.perf_counter() - started
        (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(summary[["lcp", "sampled_pairs", "mean_semantic_cosine", "mean_collab_cosine", "cointeraction_rate"]].to_string(index=False), flush=True)
        print(f"ANALYSIS_COMPLETE pairs={metadata['total_sampled_pairs']:,} seconds={metadata['total_seconds']:.2f} peak_MiB={metadata['peak_working_set_mib']}", flush=True)


# Plotting and reporting helpers are kept below the numerical functions.

def make_plots(summary, groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#2478B4", "#DB782B")
    x = summary.lcp.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for column, label, color, marker in (
        ("mean_semantic_cosine", "Sentence-T5 semantic cosine", colors[0], "o"),
        ("mean_collab_cosine", "Training-interaction cosine", colors[1], "s"),
    ):
        values = summary[column].to_numpy()
        ax.plot(x, values, color=color, marker=marker, linewidth=2, label=label)
        for position, value in zip(x, values):
            if np.isfinite(value):
                ax.annotate(f"{value:.5f}", (position, value), xytext=(0, 9),
                            textcoords="offset points", ha="center", color=color)
    ax.set(xlabel="Exact longest common SID prefix (LCP)", ylabel="Mean cosine similarity",
           title="Beauty: semantic and collaborative similarity", ylim=(-0.03, 1.07), xticks=x)
    ax.set_xticklabels([f"{row.lcp}\nsem n={row.sampled_pairs:,}\ncollab n={row.valid_collab_pairs:,}"
                        for row in summary.itertuples()])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(output / "lcp_similarity_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = summary.cointeraction_rate.to_numpy()
    valid = np.isfinite(values)
    ax.bar(x[valid], values[valid], color=colors[1], width=0.6)
    for row in summary.itertuples():
        if np.isfinite(row.cointeraction_rate):
            ax.annotate(f"{row.cointeraction_rate:.2%}\nn={row.valid_collab_pairs:,}",
                        (row.lcp, row.cointeraction_rate), xytext=(0, 5),
                        textcoords="offset points", ha="center")
        else:
            ax.text(row.lcp, 0.03, "No valid pairs", ha="center")
    ax.set(xlabel="Exact longest common SID prefix (LCP)", ylabel="Co-interaction rate",
           title="At least one shared training user", ylim=(0, 1), xticks=x)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "lcp_cointeraction_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, field, color, label in zip(axes, ("semantic", "collaborative"), colors,
                                      ("Sentence-T5 semantic cosine", "Training-interaction cosine")):
        data, positions = [], []
        for k, group in enumerate(groups):
            values = group[field][np.isfinite(group[field])]
            if len(values):
                data.append(values)
                positions.append(k)
        if data:
            artists = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                                 showfliers=False, whis=(5, 95))
            for box in artists["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.65)
        ax.set(xticks=x, xlabel="Exact SID LCP", ylabel="Cosine similarity", title=label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Pair distributions: box=25th-75th percentiles, whiskers=5th-95th")
    fig.tight_layout()
    fig.savefig(output / "lcp_similarity_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary, metadata, output, args):
    import os

    def trend(column):
        values = summary[column].to_numpy()
        values = values[np.isfinite(values)]
        return len(values) > 1 and bool((np.diff(values) > 0).all())

    sem_rising = trend("mean_semantic_cosine")
    collab_rising = trend("mean_collab_cosine")
    co_rising = trend("cointeraction_rate")
    sem_text = "有样本组的均值随 LCP 严格上升，本次数据观察到层次化语义趋势。" if sem_rising else "均值没有随 LCP 严格上升，本次数据不支持简单的单调层次化语义假设。"
    collab_text = "平均协同余弦随 LCP 严格上升。" if collab_rising else "平均协同余弦未随 LCP 严格上升。"
    co_text = "至少一次共同交互的比例也严格上升。" if co_rising else "至少一次共同交互的比例未严格上升。"
    root_link = os.path.relpath(args.data_dir.resolve(), output).replace("\\", "/")
    rows = ["| LCP | 可用物品对 | 实际分析物品对 | 有效协同对 | Mean Semantic | Mean Collaborative | Co-interaction Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    details = ["| LCP | 语义中位数 | 语义标准差 | 协同中位数 | 协同标准差 | 协同零值比例 | 配对热门度均值 / 中位数 |",
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        rows.append(f"| {row.lcp} | {row.total_possible_pairs:,} | {row.sampled_pairs:,} | {row.valid_collab_pairs:,} | {row.mean_semantic_cosine:.8f} | {row.mean_collab_cosine:.8f} | {row.cointeraction_rate:.4%} |")
        details.append(f"| {row.lcp} | {row.median_semantic_cosine:.8f} | {row.std_semantic_cosine:.8f} | {row.median_collab_cosine:.8f} | {row.std_collab_cosine:.8f} | {row.collab_zero_rate:.4%} | {row.mean_pair_geometric_train_popularity:.3f} / {row.median_pair_geometric_train_popularity:.3f} |")
    data_stats = "\n".join([
        "| 数据统计 | 数量 |", "| --- | ---: |",
        f"| 总商品 / Total Items | {metadata['total_items']:,} |",
        f"| 有效 SID 商品 / Items with SID | {metadata['items_with_sid']:,} |",
        f"| 有效原始向量商品 / Items with Semantic Embedding | {metadata['items_with_semantic_embedding']:,} |",
        f"| 有训练交互商品 / Items with Interactions | {metadata['items_with_interactions']:,} |",
        f"| 基本评估商品集合 / Items used in evaluation | {metadata['items_used_in_evaluation']:,} |",
        f"| 用户 | {metadata['users']:,} |",
        f"| 训练 User-Item 去重交互 | {metadata['deduplicated_train_interactions']:,} |",
        f"| 与原训练划分核对的样本 | {metadata['training_samples_verified']:,} |",
    ])
    correlations = metadata["correlations"]
    zero_rates = ", ".join(f"LCP={row.lcp}: {row.collab_zero_rate:.2%}" for row in summary.itertuples())
    sem_range = (summary.mean_semantic_cosine.min(), summary.mean_semantic_cosine.max())
    co_range = (summary.mean_collab_cosine.min(), summary.mean_collab_cosine.max())
    full_groups = [str(row.lcp) for row in summary.itertuples()
                   if row.total_possible_pairs > 0 and row.sampled_pairs == row.total_possible_pairs]
    partial_groups = [str(row.lcp) for row in summary.itertuples()
                      if row.sampled_pairs > 0 and row.sampled_pairs < row.total_possible_pairs]
    coverage_text = " ".join([
        "LCP=" + ",".join(partial_groups) + " 的均值是组内均匀抽样估计。" if partial_groups else "",
        "LCP=" + ",".join(full_groups) + " 已覆盖全部可用商品对。" if full_groups else "",
    ]).strip()
    peak = metadata["peak_working_set_mib"]
    memory_text = f"{peak:.1f} MiB" if peak is not None else "未测量（缺少 psutil）"
    lines = [
        "# TIGER SID 语义与协同相似性分析：Amazon 2014 Beauty", "",
        f"本地 CPU 离线分析，记录时间 {metadata['created_at']}。每组最多分析 {args.num_pairs_per_group:,} 对，种子 {args.seed}；共分析 {metadata['total_sampled_pairs']:,} 对不同商品。", "",
        "## 数据来源与对齐", "",
        f"复用 [云端完整备份]({root_link}/offline_backup_verification.json)，代码提交 `{metadata['source_commit']}`。原始数据及 checkpoint 保持原样。", "",
        f"- SID：`{metadata['data_root']}/analysis/item_semantic_ids.npy` 和 `item_sid.parquet`；使用现有最终 RQ-VAE 导出的前三位语义码，当前数据实际就是三位 SID。",
        f"- 语义向量：`{metadata['data_root']}/data/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet`，768 维原始 Sentence-T5-XL 内容向量；没有使用 RQ-VAE 隐向量或量化向量。",
        f"- 交互：`{metadata['data_root']}/analysis/user_sequences.parquet`，只取每个用户按时间排序序列的 `sequence[:-2]`，去掉原划分中的验证与测试事件，User-Item 去重后构建稀疏二值矩阵。",
        "- 商品的零起始 `item_index` 对齐 SID 行与 embedding 行，向量中的一始 `ItemID`、ASIN、用户索引及原训练目标序列均已核对；五个输入文件重新通过备份 SHA-256 校验。具体哈希保存在 `analysis_metadata.json`。",
        "- 基本商品集合取有效 SID 与有限、非零范数原始向量的交集；有些商品没有训练交互，这些配对的协同相似度和共同用户数记为 NaN，不当作零。", "",
        data_stats, "", "## 配对抽样与计算", "",
        "LCP=0、1、2、3 是互不重叠的精确最长公共前缀分组，第四位消歧码不参与。每个组建立一个从整数 rank 到合法无序商品对的一一映射：一个左商品的 rank 数量等于该前缀下、下一级码不同的合法右商品数量；LCP=3 则枚举同码簇内不同商品的无序配对。", "",
        "在完整 rank 范围均匀无放回抽样，因此每个合法物品对等概率，不采用每簇等量采样，也不需要为组内均值再补权重。每组不超过配置上限；不足上限的组全部分析。所有配对满足 i<j，不重复，并已逐对断言精确 LCP。", "",
        "语义余弦使用 float64 归一化向量，在分块中做对应行点积。协同余弦为共同用户数 / sqrt(两侧训练用户数之积)，共同用户数通过选中配对的两个稀疏行逐元素相乘再求和得到。没有构建稠密 User-Item 矩阵或任何全量 Item-Item 相似度矩阵。", "",
        f"每个计算块 {args.batch_size:,} 对，CPU 线程上限 {args.threads}。标准差采用 ddof=1；没有有效样本的统计量为 NaN。Co-interaction Rate 分母只计两侧都有训练交互的有效配对。", "",
        "## 主要统计", "", "\n".join(rows), "", "\n".join(details), "",
        "配对热门度定义为 sqrt(两个商品各自训练交互用户数之积)，仅在有效协同配对上统计。" + coverage_text, "",
        "## 实际结论", "",
        f"**A：语义层次。** {sem_text} 这是一份数据集上的描述性结果，不等于对其他数据集的保证。", "",
        f"**B：协同关联。** {collab_text}{co_text} {'两项指标均呈现单调趋势，观察到共享前缀与训练共现的统计关联。' if collab_rising and co_rising else '两项协同指标未同时呈现单调趋势，H2 的严格版本未得到完整支持。'}", "",
        f"**C：两类相似性的一致性与尺度。** 语义均值范围为 {sem_range[0]:.6f}–{sem_range[1]:.6f}，协同均值范围为 {co_range[0]:.6f}–{co_range[1]:.6f}。语义相似不能直接替代用户共现；协同零值比例为 {zero_rates}。结合箱线图中的中位数与尾部，判断均值是否被少量共现配对影响。", "",
        "交互矩阵的稀疏性会使多数配对协同值为零；热门商品有更多共同用户机会，不同 LCP 组的商品热门度与分布也不同，因此共现变化不能单独归因到 SID 前缀。LCP=3 是量化碰撞，代表不同商品共享前三位码，不代表它们在严格商品级推荐中可以互换。", "",
        "即使存在语义、协同相关性，也只能说明统计关联，不能直接证明 RQ-VAE 在训练中学习了协同信息。共同用户共现不包含顺序或时间关系，不是完整序列协同指标。", "",
        "## 可视化", "",
        "![LCP 与语义、协同均值](lcp_similarity_comparison.png)", "",
        "![LCP 与共同交互比例](lcp_cointeraction_rate.png)", "",
        "![两类相似度分布](lcp_similarity_distribution.png)", "",
        "分布图盒体为 25%–75% 分位，须为 5%–95% 分位，未显示须外点；没有删改 CSV 中的尾部配对。", "",
        "## 相关性与效率", "",
        f"分层抽样样本上的 Spearman：语义 rho={correlations['semantic']['rho']}，协同 rho={correlations['collaborative']['rho']}。只用有效配对；没有进行总体相关性加权估计。这些 rho 不能当作全部商品对的总体相关系数，主要结论仍看各组均值与共现比例。", "",
        f"相似度采样及计算耗时 {metadata['similarity_seconds']:.2f} 秒，分析主体（含输入读取、配对明细写盘及出图）耗时 {metadata['total_seconds']:.2f} 秒。进程峰值工作集 {memory_text}。未训练模型、未使用 GPU、未重新生成 SID。", "",
        "## 运行命令与输出", "",
        "在项目根目录执行以下命令。以下是本次实际配置；复跑请选择新的输出目录，程序会拒绝覆盖已有结果。", "",
        "```powershell",
        'python -m tiger_prefix_ft.analyze_sid_similarity `',
        '  --data-dir "out/cloud_reproduction/tiger_beauty_fast_20261012_194804" `',
        f'  --output-dir "{os.path.relpath(output, Path.cwd()).replace(chr(92), chr(47))}" `',
        f"  --num-pairs-per-group {args.num_pairs_per_group} --seed {args.seed} --batch-size {args.batch_size} --threads {args.threads}",
        "```", "",
        "小规模边界测试：", "", "```powershell",
        'python -m unittest tiger_prefix_ft.test_sid_similarity -v',
        "```", "",
        f"所有分析输出统一位于 `{output}`：", "",
        "- [sid_similarity_summary.csv](sid_similarity_summary.csv)：全部要求的四组统计及有效协同总对数、抽样比例、热门度。",
        "- [sid_similarity_pairs.csv](sid_similarity_pairs.csv)：逐对明细，NaN 表示缺少有效训练交互；另有 [Parquet](sid_similarity_pairs.parquet)，适合更快读取。",
        "- [analysis_metadata.json](analysis_metadata.json)：输入、输出校验哈希、参数、环境版本、耗时与分层样本相关性。",
        "- 三张图和本报告保存在同一分析目录。实现代码与边界测试位于仓库的 tiger_prefix_ft/。", "",
    ]
    (output / "sid_similarity_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
