"""Raw recording and integrity checks; no error labels, mining or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 2
D, K, V, H = 3, 15, 769, 128
OFFSETS = np.array([1, 257, 513], dtype=np.int64)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def decoded(tokens):
    raw = np.asarray(tokens, dtype=np.int64) - OFFSETS[:len(tokens)]
    valid = bool(np.all((raw >= 0) & (raw < 256)))
    # Preserve both raw arithmetic and the original tokens, even if invalid.
    return raw.tolist() if valid else None, raw.tolist(), valid


class RawRecorder:
    """Copy original search decisions after actual top-k; never mutate inputs."""
    def __init__(self, batch_size, save_logits=True, save_hidden=True):
        self.batch_size, self.save_logits, self.save_hidden = batch_size, save_logits, save_hidden
        self.steps, self.pending_hidden = [], None

    def decoder_hook(self, module, inputs, output):
        # Last layer final-norm state for the current PARENT before extension.
        self.pending_hidden = output[0][:, -1, :].detach().float().cpu().numpy().copy()

    def __call__(self, **values):
        b, k = values["batch_size"], values["num_beams"]
        require((b, k) == (self.batch_size, K), "Unexpected beam shape")
        depth, first = values["depth"], values["first_beam_only"]
        require(depth == len(self.steps) + 1, "Recorder step order mismatch")

        def cpu(value, dtype=None):
            array = value.detach().cpu().numpy().copy()
            return array.astype(dtype, copy=False) if dtype else array

        def expanded(value):
            if first:
                # Exact distribution broadcast by original search; placeholder
                # copies are explicitly masked, not independent model forwards.
                return np.repeat(value[:, None, :], K, axis=1)
            return value.reshape(b, K, -1)

        active = np.ones((b, K), dtype=bool)
        if depth == 1:
            active[:, 1:] = False
        computed = np.ones_like(active)
        if first:
            computed[:, 1:] = False
        source = np.tile(np.arange(K, dtype=np.int16), (b, 1))
        if first:
            source.fill(0)
        parent_tokens = cpu(values["parent_tokens"], np.int16).reshape(b, K, depth)
        padded_parents = np.full((b, K, D), -1, dtype=np.int16)
        padded_parents[:, :, :depth] = parent_tokens
        step = {
            "next_token_logprobs": expanded(cpu(values["next_token_logprobs"], np.float32)),
            "beam_scores_before": cpu(values["parent_scores"], np.float32),
            "selected_flat_indices": cpu(values["selected_flat_indices"], np.int32),
            "selected_parent_indices": cpu(values["selected_flat_indices"] // V, np.int16),
            "selected_token_ids": cpu(values["selected_flat_indices"] % V, np.int16),
            "selected_cumulative_scores": cpu(values["selected_cumulative_scores"], np.float32),
            "active_beam_mask": active, "initial_placeholder_mask": ~active,
            "forward_evaluated_mask": computed, "logprob_source_parent_index": source,
            "parent_prefix_tokens_with_start": padded_parents,
        }
        if self.save_logits:
            step["next_token_logits"] = expanded(cpu(values["next_token_logits"], np.float32))
        if self.save_hidden:
            require(self.pending_hidden is not None, "Decoder hook did not run")
            step["parent_last_hidden_state"] = expanded(self.pending_hidden)
            self.pending_hidden = None
        self.steps.append(step)

    def tensors(self, generated):
        require(len(self.steps) == D, "Incomplete generation")
        result = {key: np.stack([step[key] for step in self.steps], axis=1) for key in self.steps[0]}
        result["final_tokens_with_start"] = generated.detach().cpu().numpy().astype(np.int16).reshape(
            self.batch_size, K, D + 1)
        result["final_cumulative_scores"] = result["selected_cumulative_scores"][:, -1].copy()
        return result


def verify_tensors(data):
    """Reconstruct all actual choices on CPU from FP32 matrices, without T5.

    For exactly tied scores, the recorded CUDA top-k order is the authoritative
    tie break; NumPy/CPU torch topk does not promise the same tied-token order.
    Verify the complete selected score multiset, all strictly better candidates,
    original flat indices, every prefix and final sequences.
    """
    logp = data["next_token_logprobs"]
    b = len(logp)
    require(logp.shape == (b, D, K, V) and logp.dtype == np.float32, "Wrong logprob shape/dtype")
    for key, value in data.items():
        if value.dtype.kind == "f":
            require(np.isfinite(value).all(), f"Non-finite tensor: {key}")
    normalization = float(np.max(np.abs(np.exp(logp.astype(np.float64)).sum(axis=-1) - 1)))
    require(normalization < 3e-6, "Unnormalized full-vocabulary probabilities")
    require(np.all(data["active_beam_mask"][:, 0, 0]), "Missing active initial parent")
    require(not np.any(data["active_beam_mask"][:, 0, 1:]), "Initial placeholders marked active")
    require(np.all(data["active_beam_mask"][:, 1:]), "Real parents discarded")
    require(np.array_equal(data["initial_placeholder_mask"], ~data["active_beam_mask"]), "Placeholder masks differ")
    require(np.all(data["beam_scores_before"][:, 0, 0] == 0), "Initial active score changed")
    require(np.all(data["beam_scores_before"][:, 0, 1:] == np.float32(-1e9)), "Placeholder scores changed")
    require(np.all(data["forward_evaluated_mask"][:, 0, 0]) and
            not np.any(data["forward_evaluated_mask"][:, 0, 1:]) and
            np.all(data["forward_evaluated_mask"][:, 1:]), "Wrong first_beam_only computation mask")
    require(np.all(data["logprob_source_parent_index"][:, 0] == 0), "Initial broadcast source differs")
    require(np.array_equal(data["logprob_source_parent_index"][:, 1:],
                           np.broadcast_to(np.arange(K), (b, D - 1, K))), "Parent source differs")
    require(np.array_equal(logp[:, 0], np.repeat(logp[:, 0, :1], K, axis=1)), "First distribution broadcast differs")
    paths = np.zeros((b, K, 1), dtype=np.int16)
    ties, cpu_order_differences, max_score_error = 0, 0, 0.0
    for step in range(D):
        require(np.array_equal(paths, data["parent_prefix_tokens_with_start"][:, step, :, :step + 1]),
                "Recorded parent path does not follow preceding choices")
        require(np.all(data["parent_prefix_tokens_with_start"][:, step, :, step + 1:] == -1), "Bad tensor prefix padding")
        if step:
            require(np.array_equal(data["beam_scores_before"][:, step], data["selected_cumulative_scores"][:, step - 1]),
                    "Parent score mismatch across steps")
        scores = (logp[:, step] + data["beam_scores_before"][:, step, :, None]).reshape(b, K * V)
        indices = data["selected_flat_indices"][:, step]
        parents, tokens = indices // V, indices % V
        require(np.all((indices >= 0) & (indices < K * V)), "Out-of-range flattened choice")
        require(np.array_equal(parents, data["selected_parent_indices"][:, step]) and
                np.array_equal(tokens, data["selected_token_ids"][:, step]), "Flat/parent/token indices disagree")
        selected = np.take_along_axis(scores, indices, axis=1)
        diff = float(np.max(np.abs(selected - data["selected_cumulative_scores"][:, step])))
        max_score_error = max(max_score_error, diff)
        require(diff == 0, "FP32 candidate score does not exactly reconstruct actual selected score")
        require(np.all(selected[:, :-1] >= selected[:, 1:]), "Actual beam ranks are not score-sorted")
        independent = np.sort(np.partition(scores, scores.shape[1] - K, axis=1)[:, -K:], axis=1)[:, ::-1]
        require(np.array_equal(independent, selected), "Actual choices are not global top-k")
        for row in range(b):
            require(len(set(indices[row].tolist())) == K, "Duplicate flat choice in top-k")
            threshold = selected[row, -1]
            strict = set(np.flatnonzero(scores[row] > threshold).tolist())
            require(strict.issubset(set(indices[row].tolist())), "Strictly higher-scoring candidate was lost")
            tied = len(np.flatnonzero(scores[row] == threshold)) > 1 or len(np.unique(selected[row])) < K
            ties += int(tied)
            # Unique top-k scores imply a unique ordering, already proven by the
            # independent score-multiset check. Sort the full row only for ties.
            if tied:
                cpu_top = np.argsort(-scores[row], kind="stable")[:K]
                cpu_order_differences += int(not np.array_equal(cpu_top, indices[row]))
        paths = np.concatenate([paths[np.arange(b)[:, None], parents], tokens[:, :, None]], axis=-1)
    require(np.array_equal(paths, data["final_tokens_with_start"]), "Reconstructed final tokens differ")
    require(np.array_equal(data["final_cumulative_scores"], data["selected_cumulative_scores"][:, -1]), "Final scores differ")
    if "gold_next_token_logprobs" in data:
        gold = data["gold_next_token_logprobs"]
        require(gold.shape == (b, D, V) and gold.dtype == np.float32, "Wrong gold probability shape")
        gold_norm = float(np.max(np.abs(np.exp(gold.astype(np.float64)).sum(-1) - 1)))
        require(gold_norm < 3e-6, "Gold distribution not normalized")
        chosen = np.take_along_axis(gold, data["gold_next_token_ids"][:, :, None], axis=-1).squeeze(-1)
        require(np.array_equal(chosen, data["gold_token_logprobs"]), "Gold target probability differs")
        require(np.allclose(np.cumsum(chosen, axis=1, dtype=np.float32), data["gold_cumulative_logprobs"], atol=2e-6, rtol=0),
                "Gold cumulative score differs")
        ranks = 1 + (gold > chosen[:, :, None]).sum(-1)
        require(np.array_equal(ranks, data["gold_token_ranks"]), "Gold rank differs")
        normalization = max(normalization, gold_norm)
    return {"samples": b, "reconstructed_steps": b * D, "max_score_abs_error": max_score_error,
            "max_probability_sum_error": normalization, "steps_with_score_ties": ties,
            "steps_cpu_stable_tie_order_differs": cpu_order_differences,
            "tie_rule": "Original recorded CUDA top-k order; full score multiset checked independently",
            "final_tokens_exact": True, "all_parent_paths_exact": True}


def add_readable_paths(records, data):
    for row, record in enumerate(records):
        steps = []
        for step in range(D):
            parents, survivors = [], []
            for rank in range(K):
                parent = data["parent_prefix_tokens_with_start"][row, step, rank, :step + 1].tolist()
                parents.append({"beam_rank": rank + 1, "parent_beam_idx": rank,
                                "prefix_tokens_with_start": parent, "prefix_tokens": parent[1:],
                                "cumulative_score": float(data["beam_scores_before"][row, step, rank]),
                                "active": bool(data["active_beam_mask"][row, step, rank]),
                                "initial_placeholder": bool(data["initial_placeholder_mask"][row, step, rank]),
                                "forward_evaluated": bool(data["forward_evaluated_mask"][row, step, rank]),
                                "distribution_source_parent_idx": int(data["logprob_source_parent_index"][row, step, rank])})
                idx = int(data["selected_parent_indices"][row, step, rank])
                token = int(data["selected_token_ids"][row, step, rank])
                prefix = data["parent_prefix_tokens_with_start"][row, step, idx, :step + 1].tolist() + [token]
                survivors.append({"beam_rank": rank + 1, "parent_beam_idx": idx, "selected_token_id": token,
                                  "selected_flat_index": int(data["selected_flat_indices"][row, step, rank]),
                                  "prefix_tokens_with_start": prefix, "prefix_tokens": prefix[1:],
                                  "token_logprob": float(data["next_token_logprobs"][row, step, idx, token]),
                                  "parent_cumulative_score": float(data["beam_scores_before"][row, step, idx]),
                                  "cumulative_score": float(data["selected_cumulative_scores"][row, step, rank])})
            steps.append({"depth": step + 1, "vocab_size": V,
                          "candidate_order": "parent-major, token-minor; flat_index=parent_idx*769+token_id",
                          "parents": parents, "surviving_beams": survivors})
        record["steps"] = steps
        finals = []
        for rank in range(K):
            full = data["final_tokens_with_start"][row, rank].tolist()
            raw, arithmetic, valid = decoded(full[1:])
            ancestor_ranks, position = [], rank
            for step in reversed(range(D)):
                ancestor_ranks.append(position + 1)
                position = int(data["selected_parent_indices"][row, step, position])
            finals.append({"beam_rank": rank + 1, "tokens_with_start": full, "sid_tokens": full[1:],
                           "raw_sid": raw, "raw_sid_arithmetic": arithmetic, "position_offset_valid": valid,
                           "cumulative_score": float(data["final_cumulative_scores"][row, rank]),
                           "beam_rank_at_each_depth": list(reversed(ancestor_ranks)),
                           "initial_parent_beam_idx": position})
        record["final_beams"] = finals
        record["gold_teacher_forced"] = [{
            "depth": step + 1, "gold_prefix_tokens": record["target_tokens"][:step],
            "decoder_prefix_with_start": [0] + record["target_tokens"][:step],
            "gold_next_token": int(data["gold_next_token_ids"][row, step]),
            "gold_token_logprob": float(data["gold_token_logprobs"][row, step]),
            "gold_cumulative_logprob": float(data["gold_cumulative_logprobs"][row, step]),
            "gold_token_rank": int(data["gold_token_ranks"][row, step]),
            "gold_equal_score_token_count": int(data["gold_equal_score_token_counts"][row, step]),
            "conditioning": "separate frozen teacher-forced forward, after free beam generation",
        } for step in range(D)]
