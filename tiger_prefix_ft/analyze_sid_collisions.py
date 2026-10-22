"""Count item SID collisions by the first three codes, without model inference."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


GROUP_COLUMNS = [
    "collision_group_id", "sid_0", "sid_1", "sid_2", "prefix_sid", "item_count",
    "collision_pair_count", "item_indices", "asins", "fourth_tokens",
    "fourth_token_counts", "distinct_fourth_tokens", "repeated_fourth_token_items",
    "remaining_collision_pairs_after_fourth",
]
ITEM_COLUMNS = [
    "collision_group_id", "collision_group_size", "item_index", "asin", "sid",
    "sid_0", "sid_1", "sid_2", "sid_3",
]
SIZE_COLUMNS = ["group_size", "group_count", "item_count", "collision_pair_count"]
TOKEN_COLUMNS = ["fourth_token", "item_count", "colliding_item_count", "collision_group_count"]
OUTPUT_NAMES = [
    "sid_collision_summary.json", "sid_collision_groups.csv", "sid_collision_items.csv",
    "sid_collision_size_distribution.csv", "sid_fourth_token_distribution.csv", "sid_collision_report.md",
]


def integer_codes(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError("SID must have shape (items, 3) or (items, 4)")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("SID codes must be numeric integers")
    if not np.isfinite(values).all() or (values < 0).any() or (values >= 2 ** 63).any():
        raise ValueError("SID codes must be finite nonnegative int64 values")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("SID codes must be integers")
    return values.astype(np.int64)


def parse_sid(value):
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else list(map(int, value.split("-")))
    return value


def load_sid_file(path):
    """Read the existing item mapping, or a row-aligned three/four-column array."""
    if path.suffix.lower() == ".npy":
        sids = integer_codes(np.load(path, allow_pickle=False))
        return sids, np.arange(len(sids)), [None] * len(sids), "NPY row index (zero-based)"
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"asin": "string"})
    else:
        raise ValueError("Supported inputs: .parquet, .csv, .npy")
    columns = [f"sid_{position}" for position in range(3)]
    if set(columns) <= set(frame.columns):
        if "sid_4" in frame.columns:
            raise ValueError("Only three semantic codes and at most one disambiguation code are supported")
        if "sid_3" in frame.columns:
            columns.append("sid_3")
        sids = integer_codes(frame[columns].to_numpy())
        if "sid" in frame and len(frame):
            declared = integer_codes(np.stack(frame.sid.map(parse_sid).to_numpy()))
            if not np.array_equal(declared, sids):
                raise ValueError("The sid field disagrees with the sid_0..sid_3 columns")
    elif "sid" in frame:
        if not len(frame):
            raise ValueError("Empty input needs explicit sid_0..sid_2 columns to determine SID width")
        sids = integer_codes(np.stack(frame.sid.map(parse_sid).to_numpy()))
    else:
        raise ValueError("Input needs sid_0..sid_2 columns, optional sid_3, or a sid list/string column")
    ids = frame.item_index.to_numpy() if "item_index" in frame else np.arange(len(frame))
    asins = [None if pd.isna(value) else str(value) for value in frame.asin] if "asin" in frame else [None] * len(frame)
    source = "item_index column" if "item_index" in frame else "input row index (zero-based)"
    return sids, ids, asins, source


def analyze_collisions(sids, item_indices=None, asins=None):
    sids = integer_codes(sids)
    total, width = sids.shape
    ids = np.arange(total) if item_indices is None else np.asarray(item_indices)
    if ids.ndim != 1 or len(ids) != total or not pd.Series(ids).is_unique:
        raise ValueError("item_index values must be unique and aligned with SID rows")
    if not np.issubdtype(ids.dtype, np.number) or not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("item_index values must be integers")
    ids = ids.astype(np.int64)
    labels = [None] * total if asins is None else list(asins)
    if len(labels) != total:
        raise ValueError("ASIN mapping is not aligned with SID rows")
    if any(pd.notna(label) and not isinstance(label, str) for label in labels):
        raise ValueError("ASIN identifiers must be strings to preserve leading zeros")

    prefixes, inverse, counts = np.unique(sids[:, :3], axis=0, return_inverse=True, return_counts=True)
    collision_indices = [index for index, count in enumerate(counts) if count > 1]
    collision_indices.sort(key=lambda index: (-int(counts[index]), tuple(prefixes[index])))
    # One stable sort builds every group membership; no item-pair matrix is constructed.
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    group_rows, item_rows = [], []
    suffix_collision_groups = Counter()
    for group_id, index in enumerate(collision_indices, start=1):
        positions = order[starts[index]:starts[index + 1]]
        positions = positions[np.argsort(ids[positions], kind="stable")]
        prefix = list(map(int, prefixes[index]))
        suffix = Counter(map(int, sids[positions, 3])) if width == 4 else None
        row = {
            "collision_group_id": group_id, **{f"sid_{i}": prefix[i] for i in range(3)},
            "prefix_sid": json.dumps(prefix), "item_count": len(positions),
            "collision_pair_count": len(positions) * (len(positions) - 1) // 2,
            "item_indices": json.dumps(ids[positions].tolist()),
            "asins": json.dumps([labels[position] for position in positions]),
            "fourth_tokens": json.dumps(sorted(suffix)) if suffix is not None else None,
            "fourth_token_counts": json.dumps(dict(sorted(suffix.items()))) if suffix is not None else None,
            "distinct_fourth_tokens": len(suffix) if suffix is not None else None,
            "repeated_fourth_token_items": sum(n for n in suffix.values() if n > 1) if suffix is not None else None,
            "remaining_collision_pairs_after_fourth": sum(n * (n - 1) // 2 for n in suffix.values()) if suffix is not None else None,
        }
        group_rows.append(row)
        if suffix is not None:
            suffix_collision_groups.update(suffix.keys())
        for position in positions:
            item_rows.append({
                "collision_group_id": group_id, "collision_group_size": len(positions),
                "item_index": int(ids[position]), "asin": labels[position],
                "sid": json.dumps(sids[position].tolist()),
                **{f"sid_{i}": int(sids[position, i]) for i in range(3)},
                "sid_3": int(sids[position, 3]) if width == 4 else None,
            })
    size_rows = [
        {"group_size": size, "group_count": groups, "item_count": size * groups,
         "collision_pair_count": size * (size - 1) // 2 * groups}
        for size, groups in sorted(Counter(map(int, counts)).items())
    ]
    token_rows = []
    if width == 4:
        full_counts = np.unique(sids, axis=0, return_counts=True)[1]
        all_tokens = Counter(map(int, sids[:, 3]))
        collided_tokens = Counter(map(int, sids[counts[inverse] > 1, 3]))
        token_rows = [
            {"fourth_token": token, "item_count": amount, "colliding_item_count": collided_tokens[token],
             "collision_group_count": suffix_collision_groups[token]}
            for token, amount in sorted(all_tokens.items())
        ]
    else:
        full_counts = None
    summary = {
        "collision_definition": "Different items with equal first three SID codes; the fourth code is excluded",
        "total_items": total, "input_sid_width": width, "semantic_sid_width": 3,
        "has_fourth_token": width == 4, "unique_semantic_prefixes": len(prefixes),
        "non_colliding_items": int((counts == 1).sum()),
        "collision_groups": len(collision_indices),
        "colliding_items": int(counts[counts > 1].sum()),
        "colliding_item_rate": float(counts[counts > 1].sum() / total) if total else None,
        "excess_items_beyond_one_per_prefix": total - len(prefixes),
        "collision_pairs": sum(int(n) * (int(n) - 1) // 2 for n in counts),
        "max_group_size": int(counts.max()) if len(counts) else 0,
        "remaining_collision_groups_after_fourth": int((full_counts > 1).sum()) if full_counts is not None else None,
        "remaining_colliding_items_after_fourth": int(full_counts[full_counts > 1].sum()) if full_counts is not None else None,
        "remaining_collision_pairs_after_fourth": sum(int(n) * (int(n) - 1) // 2 for n in full_counts) if full_counts is not None else None,
    }
    groups = pd.DataFrame(group_rows, columns=GROUP_COLUMNS)
    members = pd.DataFrame(item_rows, columns=ITEM_COLUMNS)
    distribution = pd.DataFrame(size_rows, columns=SIZE_COLUMNS)
    tokens = pd.DataFrame(token_rows, columns=TOKEN_COLUMNS)
    assert len(members) == summary["colliding_items"]
    assert distribution.item_count.sum() == total
    assert distribution.collision_pair_count.sum() == summary["collision_pairs"]
    return summary, groups, members, distribution, tokens


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_report(summary, groups, distribution, output):
    rate = f"{summary['colliding_item_rate']:.4%}" if summary["colliding_item_rate"] is not None else "无商品，比例未定义"
    fourth = ("输入有第四位；各组真实 token 值、频次及仍然重复的四位 SID 已统计。"
              if summary["has_fourth_token"] else
              "当前输入只有三位 SID，没有第四位消歧 token。第四位字段留空，尾码分布 CSV 只有表头；没有补造 0 或其他尾码。")
    lines = [
        "# SID 碰撞统计", "",
        "碰撞定义：两个不同商品的前三位 SID 完全相同。第四位是消歧 token，不参与这个分组条件。", "",
        f"输入：`{summary['input_file']}`；商品标识来自 {summary['item_identifier_source']}。", "",
        "| 指标 | 结果 |", "| --- | ---: |",
        f"| 总商品数 | {summary['total_items']:,} |",
        f"| 不同三位前缀数 | {summary['unique_semantic_prefixes']:,} |",
        f"| 碰撞组数（组大小至少 2） | {summary['collision_groups']:,} |",
        f"| 碰撞商品总数（计入每个碰撞组的全部成员） | {summary['colliding_items']:,} |",
        f"| 碰撞商品占总商品比例 | {rate} |",
        f"| 无碰撞商品数 | {summary['non_colliding_items']:,} |",
        f"| 每组只保留一件后多出的商品数 | {summary['excess_items_beyond_one_per_prefix']:,} |",
        f"| 碰撞无序商品对数 | {summary['collision_pairs']:,} |",
        f"| 最大组商品数 | {summary['max_group_size']:,} |", "",
        "例如一组 4 件商品：碰撞商品数为 4，多出商品数为 3，碰撞商品对数为 6。这三个指标含义不同。", "",
        fourth, "",
    ]
    if summary["has_fourth_token"]:
        lines.extend([
            f"加入真实第四位后，仍重复的完整 SID 组数为 {summary['remaining_collision_groups_after_fourth']:,}，"
            f"涉及商品 {summary['remaining_colliding_items_after_fourth']:,} 件，重复商品对 {summary['remaining_collision_pairs_after_fourth']:,} 对。", "",
        ])
    lines.extend(["## 分组大小分布", "", "组大小为 1 的行表示无碰撞的单商品组。", "",
                  "| 每组商品数 | 组数 | 涉及商品数 | 碰撞商品对数 |", "| ---: | ---: | ---: | ---: |"])
    for row in distribution.itertuples():
        lines.append(f"| {row.group_size} | {row.group_count:,} | {row.item_count:,} | {row.collision_pair_count:,} |")
    lines.extend(["", "## 最大的碰撞组（前 10 组）", "",
                  "| 前三位 SID | 商品数 | 实际第四位取值 |", "| --- | ---: | --- |"])
    for row in groups.head(10).itertuples():
        lines.append(f"| {row.prefix_sid} | {row.item_count} | {row.fourth_tokens if summary['has_fourth_token'] else '不存在'} |")
    if not len(groups):
        lines.extend(["", "没有碰撞组。"])
    lines.extend([
        "", "## 输出文件和运行命令", "",
        "- [sid_collision_groups.csv](sid_collision_groups.csv)：全部碰撞组，包含大小、前三位、商品索引/ASIN 列表、实际第四位取值及频次。",
        "- [sid_collision_items.csv](sid_collision_items.csv)：每件碰撞商品及其所属组、完整 SID 和第四位，便于逐件查看。",
        "- [sid_collision_size_distribution.csv](sid_collision_size_distribution.csv)：分组大小分布，包含无碰撞单商品组。",
        "- [sid_fourth_token_distribution.csv](sid_fourth_token_distribution.csv)：各第四位取值涉及的商品数和碰撞组数；没有第四位时为空表。",
        "- [sid_collision_summary.json](sid_collision_summary.json)：总统计、输入哈希及配置。", "",
        f"统计及结果写出耗时 {summary['elapsed_before_report_seconds']:.3f} 秒。仅使用 CPU 读取已有 SID，不训练模型。", "",
        "在项目根目录运行；默认读取本地 Beauty 备份。下面的命令可以更新本目录已生成的统计文件：", "",
        "```powershell",
        'python -m tiger_prefix_ft.analyze_sid_collisions --overwrite',
        "```", "",
        "指定自己的三位或四位 SID 文件（支持 NPY、Parquet、CSV）：", "", "```powershell",
        'python -m tiger_prefix_ft.analyze_sid_collisions `',
        '  --input "path/to/item_sids.npy" --output-dir "out/tiger_prefix_ft/my_sid_collisions"',
        "```", "",
        "表格输入使用 `sid_0`、`sid_1`、`sid_2` 和可选 `sid_3`，也支持 `sid` 列中的数组或连字符分隔字符串。商品 `item_index` 必须唯一；NPY 的商品标识为零起始行号。", "",
    ])
    (output / "sid_collision_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from .paths import ROOT as project
    parser.add_argument("--input", type=Path, default=project / "artifacts/beauty/item_semantic_ids.npy")
    parser.add_argument("--output-dir", type=Path, default=project / "out/tiger_prefix_ft/sid_collisions")
    parser.add_argument("--overwrite", action="store_true", help="Replace only previously generated collision statistics")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output_dir.resolve()
    for name in OUTPUT_NAMES:
        target = output / name
        if target == source:
            parser.error("Output must not replace the input SID file")
        if target.exists() and not args.overwrite:
            parser.error("Output exists; choose a new --output-dir or use --overwrite: " + str(target))
    started = time.perf_counter()
    sids, indices, asins, identifier_source = load_sid_file(source)
    summary, groups, members, distribution, tokens = analyze_collisions(sids, indices, asins)
    output.mkdir(parents=True, exist_ok=True)
    for frame, name in ((groups, OUTPUT_NAMES[1]), (members, OUTPUT_NAMES[2]),
                        (distribution, OUTPUT_NAMES[3]), (tokens, OUTPUT_NAMES[4])):
        frame.to_csv(output / name, index=False, encoding="utf-8-sig", na_rep="")
    summary.update(input_file=str(source), input_sha256=file_hash(source), item_identifier_source=identifier_source,
                   created_at=datetime.now(timezone.utc).isoformat(), output_directory=str(output),
                   elapsed_before_report_seconds=time.perf_counter() - started)
    write_report(summary, groups, distribution, output)
    summary["output_sha256"] = {name: file_hash(output / name) for name in OUTPUT_NAMES[1:]}
    summary["script_sha256"] = file_hash(Path(__file__))
    (output / OUTPUT_NAMES[0]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "total_items", "collision_groups", "colliding_items", "colliding_item_rate",
        "excess_items_beyond_one_per_prefix", "collision_pairs", "max_group_size", "has_fourth_token",
    )}, indent=2))
    print("Group-size distribution:\n" + distribution.to_string(index=False))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
