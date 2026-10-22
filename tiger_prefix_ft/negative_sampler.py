"""Simple, typed prefix sampling; no model, loss, or training code."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq

from .paths import ROOT, PREFIX_DATA as DEFAULT_DATA
OFFSETS = (1, 257, 513)


def position_valid(tokens):
    return 1 <= len(tokens) <= 3 and all(
        OFFSETS[i] <= int(token) < OFFSETS[i] + 256
        for i, token in enumerate(tokens)
    )


def shared_length(left, right):
    result = 0
    for a, b in zip(left, right):
        if a != b:
            break
        result += 1
    return result


class SIDCatalog:
    def __init__(self, rows):
        self.rows = rows
        complete = [tuple(row["offset_tokens"]) for row in rows]
        if len(set(complete)) != len(complete):
            raise ValueError("Catalog must contain unique full SIDs")
        if not complete or not all(position_valid(p) and len(p) == 3 for p in complete):
            raise ValueError("Invalid catalog Offset Token")
        self.prefixes = {d: tuple(sorted({p[:d] for p in complete})) for d in (1, 2, 3)}
        self.prefix_sets = {d: set(prefixes) for d, prefixes in self.prefixes.items()}
        self.children = {}
        self.by_first = {}
        for depth, prefixes in self.prefixes.items():
            children = defaultdict(list)
            first = defaultdict(list)
            for prefix in prefixes:
                children[prefix[:-1]].append(prefix[-1])
                first[prefix[0]].append(prefix)
            for parent, tokens in children.items():
                self.children[parent] = np.asarray(tokens, dtype=np.int64)
            self.by_first[depth] = dict(first)

    @classmethod
    def load(cls, path):
        return cls(pq.read_table(path).to_pylist())


class NegativeSampler:
    """Three Arrow tables + a sample lookup; Hard queries always use sample AND depth.

    The Hard table stays in Arrow buffers, rather than millions of Python dicts.
    Rows must be ordered by dataset_index/depth, as written by prepare_data.py.
    """

    def __init__(self, data_dir=DEFAULT_DATA, seed=42, include_gold_conditional=False):
        data_dir = Path(data_dir)
        self.catalog = SIDCatalog.load(data_dir / "sid_catalog.parquet")
        self.train = pq.read_table(data_dir / "train_samples.parquet", use_threads=False)
        self.hard = pq.read_table(data_dir / "hard_negatives.parquet", use_threads=False,
                                  read_dictionary=["sample_id", "source"])
        ids = self.train["sample_id"].to_pylist()
        self.sample_lookup = {name: i for i, name in enumerate(ids)}
        if len(self.sample_lookup) != len(ids):
            raise ValueError("Duplicate training sample ID")
        indices = self.hard["dataset_index"].to_numpy(zero_copy_only=False)
        depths = self.hard["depth"].to_numpy(zero_copy_only=False)
        self.hard_keys = indices * 3 + depths - 1
        if np.any(self.hard_keys[1:] < self.hard_keys[:-1]):
            raise ValueError("Hard Pool must be ordered by dataset_index/depth")
        self.rng = random.Random(seed)
        self.include_gold_conditional = include_gold_conditional

    def get_sample(self, sample_id):
        if sample_id not in self.sample_lookup:
            raise KeyError(f"Unknown Train sample: {sample_id}")
        return self.train.slice(self.sample_lookup[sample_id], 1).to_pylist()[0]

    def hard_for(self, sample_id, depth):
        if depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2, or 3")
        sample = self.get_sample(sample_id)
        key = sample["dataset_index"] * 3 + depth - 1
        start = int(np.searchsorted(self.hard_keys, key, side="left"))
        end = int(np.searchsorted(self.hard_keys, key, side="right"))
        rows = self.hard.slice(start, end - start).to_pylist()
        if any(r["sample_id"] != sample_id or r["depth"] != depth for r in rows):
            raise ValueError("Hard Pool sample_id/depth binding mismatch")
        return rows

    def sample(self, sample_id, depth, easy_num=2, medium_num=2, hard_num=4):
        if depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2, or 3")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0
               for n in (easy_num, medium_num, hard_num)):
            raise ValueError("Negative counts must be nonnegative integers")
        sample = self.get_sample(sample_id)
        target = tuple(sample["target_tokens"])
        if target not in self.catalog.prefix_sets[3]:
            raise ValueError("Target is missing from SID Catalog")
        positive = target[:depth]
        chosen = {positive}
        negatives = []
        rows = sorted(self.hard_for(sample_id, depth), key=lambda r: (-r["score"], r["negative_tokens"]))
        for row in rows:
            prefix = tuple(row["negative_tokens"])
            if len(prefix) != depth or prefix not in self.catalog.prefix_sets[depth] or prefix == positive:
                raise ValueError("Invalid Hard prefix")
        hard_rows = [r for r in rows if self.include_gold_conditional or r["source"] == "beam"]
        # Hard has priority when its prefix also qualifies as Medium or Easy.
        for row in hard_rows:
            if sum(n["type"] == "hard" for n in negatives) >= hard_num:
                break
            prefix = tuple(row["negative_tokens"])
            if prefix in chosen:
                continue
            chosen.add(prefix)
            negatives.append({"tokens": list(prefix), "type": "hard", "source": row["source"],
                              "cached_score": row["score"], "survived_beam": row["survived_beam"]})
        medium_pool = [] if depth == 1 else [
            prefix for prefix in self.catalog.by_first[depth].get(target[0], [])
            if prefix not in chosen
        ]
        for prefix in self.rng.sample(medium_pool, min(medium_num, len(medium_pool))):
            chosen.add(prefix)
            negatives.append({"tokens": list(prefix), "type": "medium"})
        universe = self.catalog.prefixes[depth]
        available = len(universe) - len(self.catalog.by_first[depth].get(target[0], []))
        available -= sum(p[0] != target[0] for p in chosen)
        easy_wanted = min(easy_num, available)
        # Uniform prefix sampling by rejection avoids copying the whole catalog per call.
        easy = []
        attempts = 0
        while len(easy) < easy_wanted and attempts < max(100, easy_wanted * 20):
            attempts += 1
            prefix = universe[self.rng.randrange(len(universe))]
            if prefix[0] == target[0] or prefix in chosen:
                continue
            chosen.add(prefix)
            easy.append(prefix)
        if len(easy) < easy_wanted:
            pool = [p for p in universe if p[0] != target[0] and p not in chosen]
            for prefix in self.rng.sample(pool, easy_wanted - len(easy)):
                chosen.add(prefix)
                easy.append(prefix)
        negatives.extend({"tokens": list(prefix), "type": "easy"} for prefix in easy)
        counts = {kind: sum(n["type"] == kind for n in negatives) for kind in ("easy", "medium", "hard")}
        requested = dict(easy=easy_num, medium=medium_num, hard=hard_num)
        return {"sample_id": sample_id, "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"], "ce_labels": list(target),
                "depth": depth, "positive": list(positive), "negatives": negatives,
                "requested_counts": requested, "actual_counts": counts,
                "shortfalls": {kind: requested[kind] - counts[kind] for kind in counts},
                "hard_source_policy": "beam_and_gold_conditional" if self.include_gold_conditional else "beam_only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--sample-id", default="train_0000000")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--easy-num", type=int, default=2)
    p.add_argument("--medium-num", type=int, default=2)
    p.add_argument("--hard-num", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-gold-conditional", action="store_true")
    a = p.parse_args()
    sampler = NegativeSampler(a.data_dir, a.seed, a.include_gold_conditional)
    print(json.dumps(sampler.sample(a.sample_id, a.depth, a.easy_num, a.medium_num, a.hard_num),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
