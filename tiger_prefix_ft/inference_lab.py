"""Isolated frozen-TIGER inference experiments. Existing defaults/files are untouched."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import calculate_pos_index
from .beam_diagnosis import BASE, ROOT, TaggedValidation, collate_tagged
from .finetune import file_sha256, make_loader


CHECKPOINT_HASH = "f1628a38656353e115b2739fc9c9c0ffdec74fcaf808cdbb6ef5ea615b0e4760"


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class CandidateRecorder:
    """Collect each selected path's OWN conditional log probabilities; no labels."""
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.paths, self.logps, self.depth_tokens = None, None, []

    def __call__(self, **t):
        b, k, d = t["batch_size"], t["num_beams"], t["depth"]
        parents = t["parent_tokens"].view(b, k, d)
        lp = t["next_token_logprobs"].view(b, 1 if t["first_beam_only"] else k, -1)
        v = lp.shape[-1]
        if self.paths is None:
            assert d == 1 and (parents[:, :, 0] == 0).all()
            previous = lp.new_zeros((b, k, 0))
        else:
            assert torch.equal(parents, self.paths)
            previous = self.logps

        def recover(flat):
            pi, token = flat // v, flat % v
            prefix = parents.gather(1, pi[:, :, None].expand(-1, -1, d))
            prior = previous.gather(1, pi[:, :, None].expand(-1, -1, d - 1))
            rows = torch.arange(b, device=parents.device)[:, None]
            next_lp = lp[rows, torch.zeros_like(pi) if t["first_beam_only"] else pi, token]
            return torch.cat((prefix, token[:, :, None]), -1), torch.cat((prior, next_lp[:, :, None]), -1)

        selected, components = recover(t["selected_flat_indices"])
        assert torch.allclose(components.sum(-1), t["selected_cumulative_scores"], atol=2e-6, rtol=1e-6)
        self.depth_tokens.append(selected[:, :, 1:].detach().cpu().numpy().astype(np.int16))
        if d == 3:
            candidates = (lp + t["parent_scores"][:, :, None]).view(b, -1)
            ranking = t.get("selection_candidate_scores", candidates).view(b, -1)
            _, flat = ranking.topk(self.pool_size, -1, sorted=True)
            # topk(k=30) and topk(k=256) can order equal FP16 scores differently.
            # Preserve the actual baseline selection/order, then append new paths.
            original = t["selected_flat_indices"]
            existing = (flat[:, :, None] == original[:, None, :]).any(-1)
            needed = self.pool_size - original.shape[1]
            extra_mask = ~existing & ((~existing).long().cumsum(-1) <= needed)
            extras = flat[extra_mask].view(b, needed)
            flat = torch.cat((original, extras), 1)
            scores = candidates.gather(1, flat)
            ranking_scores = ranking.gather(1, flat)
            paths, values = recover(flat)
            assert torch.equal(paths[:, :selected.shape[1]], selected)
            assert (ranking_scores[:, :-1] >= ranking_scores[:, 1:]).all()
            self.pool_tokens = paths[:, :, 1:].cpu().numpy().astype(np.int16)
            self.pool_logps = values.cpu().numpy()
            self.pool_scores = scores.cpu().numpy()
            self.pool_search_scores = ranking_scores.cpu().numpy()
        self.paths, self.logps = selected, components


@torch.inference_mode()
def alternative_search(tiger, ids, mask, widths, weights=(1., 1., 1.), sibling=0., penalty_depths=(2,), recorder=None, adjuster=None):
    """Weighted path selection; optional non-cumulative log sibling-rank penalty.

    Zero penalty/unit weights follow the original full-vocabulary Top-K exactly.
    Labels/targets never enter this function. Penalties change selection only;
    true per-token probabilities and cumulative model scores remain separate.
    """
    model = tiger.model
    encoded = model.encoder(input_ids=ids, attention_mask=mask)
    b, k, v = len(ids), widths[0], model.config.vocab_size
    paths = torch.full((b * k, 1), model.config.decoder_start_token_id, device=ids.device, dtype=torch.long)
    raw = torch.zeros(b, k, device=ids.device)
    raw[:, 1:] = -1e9
    weighted = raw.clone()
    components = raw.new_zeros((b, k, 0))
    cache, depth_tokens = None, []
    for step, keep in enumerate(widths):
        first = step == 0 and model.config.first_beam_only
        hidden = encoded[0].unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, encoded[0].shape[1], -1)
        expanded_mask = mask.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        out = model(encoder_outputs=encoded if first else (hidden,),
                    attention_mask=mask if first else expanded_mask,
                    decoder_input_ids=paths.view(b, k, -1)[:, 0] if first else paths if cache is None else paths[:, -1:],
                    use_cache=True, past_key_values=cache)
        cache = out.past_key_values
        if first:
            cache = tuple(tuple(None if pair is None else tuple(x.repeat_interleave(k, 0) for x in pair)
                                for pair in layer) for layer in cache)
        lp = out.logits[:, -1].float().log_softmax(-1).view(b, 1 if first else k, v)
        raw_candidates = raw[:, :, None] + lp
        selection = weighted[:, :, None] + weights[step] * lp
        if sibling and step + 1 in penalty_depths:
            ranks = lp.argsort(-1, descending=True).argsort(-1).float()
            selection = selection - sibling * ranks.log1p()
        if adjuster is not None:
            selection = selection + adjuster(step + 1, paths.view(b, k, -1), lp)
        _, flat = selection.reshape(b, -1).topk(keep, -1, sorted=True)
        if recorder is not None:
            recorder(depth=step + 1, batch_size=b, num_beams=k, first_beam_only=first,
                     parent_tokens=paths, parent_scores=raw, next_token_logprobs=lp,
                     selected_flat_indices=flat,
                     selected_cumulative_scores=raw_candidates.reshape(b, -1).gather(1, flat),
                     selection_candidate_scores=selection)
        parent, token = flat // v, flat % v
        rows = torch.arange(b, device=ids.device)[:, None]
        token_lp = lp[rows, torch.zeros_like(parent) if first else parent, token]
        components = torch.cat((components.gather(1, parent[:, :, None].expand(-1, -1, step)), token_lp[:, :, None]), -1)
        raw = raw_candidates.view(b, -1).gather(1, flat)
        # Do not carry sibling penalties forward into model probability scores.
        weighted = weighted.gather(1, parent) + weights[step] * token_lp
        global_parent = (rows * k + parent).reshape(-1)
        paths = torch.cat((paths[global_parent], token.reshape(-1, 1)), -1)
        depth_tokens.append(paths.view(b, keep, -1)[:, :, 1:])
        if step < 2:
            cache = model._reorder_cache(cache, global_parent)
        k = keep
    assert torch.allclose(raw, components.sum(-1), atol=2e-6, rtol=1e-6)
    return paths, components, raw, weighted, depth_tokens


@torch.inference_mode()
def generate_with_prior(tiger, input_ids, attention_mask, prior, widths=(15, 15, 15)):
    """Deployment entry: observed inputs only, one encoder, no trace collection."""
    if tiger.training:
        raise ValueError("Call tiger.eval() before inference")
    prior.prepare(input_ids, attention_mask)
    return alternative_search(tiger, input_ids, attention_mask, widths, adjuster=prior)[0]


def metrics_from_ranks(ranks):
    discount = (1 / torch.log2(torch.arange(2, 10002).float())).numpy()
    result = {}
    for k in (5, 10):
        hit = (ranks > 0) & (ranks <= k)
        result[f"Recall@{k}"] = float(hit.mean())
        result[f"NDCG@{k}"] = float(np.where(hit, discount[np.maximum(ranks - 1, 0)], 0).astype(np.float64).mean())
    return result


def ranks_for(tokens, targets):
    match = (tokens == targets[:, None, :]).all(-1)
    assert (match.sum(1) <= 1).all()
    return np.where(match.any(1), match.argmax(1) + 1, 0).astype(np.int16)


def collect(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.cuda.is_available()
    assert args.batch_size > 0 and args.pool_size >= max(args.widths)
    assert all(w > 0 for w in args.widths) and all(w > 0 for w in args.weights)
    calibrated = any(args.prior_strengths) or args.suppress_seen_singletons
    original = args.weights == [1., 1., 1.] and args.sibling == 0 and not calibrated
    checkpoint = BASE / "run/tiger/best_model.pt"
    before = file_sha256(checkpoint)
    assert before == CHECKPOINT_HASH
    config_path, sid_path = BASE / "analysis/model_config.json", BASE / "analysis/item_semantic_ids.npy"
    model = Tiger(json.loads(config_path.read_text())).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.cuda().set_sid_trie(None)
    sids = np.load(sid_path, allow_pickle=False)
    adjuster = None
    if calibrated:
        from .inference_priors import PrefixPrior
        if args.prior_train_stats is None:
            raise ValueError("Prefix calibration requires --prior-train-stats")
        with np.load(args.prior_train_stats, allow_pickle=False) as archive:
            statistics = {key: archive[key] for key in archive.files}
        adjuster = PrefixPrior(statistics, sids + np.array([1, 257, 513]), args.prior_strengths, args.suppress_seen_singletons, args.prior_history)
    source = BASE / f"analysis/{args.split}_samples.parquet"
    dataset = TaggedValidation(source, sids, args.max_samples)
    dataset.sample_ids = [f"{args.split}_{x.removeprefix('valid_')}" for x in dataset.sample_ids]
    loader = make_loader(dataset, args.batch_size, collate_tagged, False, 0, 42, True)
    args.output.mkdir(parents=True, exist_ok=False)
    chunks = {k: [] for k in ("tokens", "logps", "raw_scores", "search_scores", "depth_ranks", "targets", "histories")}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for bi, batch in enumerate(loader):
            if adjuster is not None:
                adjuster.prepare(batch["input_ids"], batch["attention_mask"])
            ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
            recorder = CandidateRecorder(args.pool_size)
            if original:
                model.model._beam_trace_recorder = recorder
                try:
                    generated = model.generate(ids, mask, num_beams=args.widths[-1], beam_sizes=args.widths)
                finally:
                    del model.model._beam_trace_recorder
            else:
                generated, _, _, _, _ = alternative_search(model, ids, mask, args.widths, args.weights, args.sibling, args.penalty_depths, recorder, adjuster)
            assert torch.equal(generated, recorder.paths.flatten(0, 1))
            tokens, lp, raw, score, depths = recorder.pool_tokens, recorder.pool_logps, recorder.pool_scores, recorder.pool_search_scores, recorder.depth_tokens
            del recorder
            targets = batch["labels"].numpy().astype(np.int16)
            dr = np.stack([ranks_for(paths, targets[:, :d]) for d, paths in enumerate(depths, 1)], 1)
            assert not ((dr[:, 1] > 0) & (dr[:, 0] == 0)).any()
            assert not ((dr[:, 2] > 0) & (dr[:, 1] == 0)).any()
            if bi == 0:
                expected = calculate_pos_index(torch.from_numpy(tokens[:, :10].astype(np.int64)), batch["labels"])
                assert np.array_equal(expected.any(1).numpy(), (ranks_for(tokens, targets) > 0) & (ranks_for(tokens, targets) <= 10))
            histories = np.zeros((len(ids), 50, 3), np.int16)
            for j, (history_ids, history_mask) in enumerate(zip(batch["input_ids"].tolist(), batch["attention_mask"].tolist())):
                history = np.asarray([t for t, active in zip(history_ids, history_mask) if active], np.int16).reshape(-1, 3)
                histories[j, -len(history):] = history
            for key, value in zip(chunks, (tokens, lp, raw, score, dr, targets, histories)):
                chunks[key].append(value)
            if bi % 80 == 0:
                print(json.dumps({"batch": bi, "samples": min((bi + 1) * args.batch_size, len(dataset)), "total": len(dataset)}), flush=True)
            del ids, mask, generated
    torch.cuda.synchronize()
    seconds, peak = time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**20
    arrays = {key: np.concatenate(parts) for key, parts in chunks.items()}
    ranks = ranks_for(arrays["tokens"], arrays["targets"])
    rows = [{"sample_id": sid, "target_tokens": target.tolist(), "rank": int(rank),
             **{f"depth{d}_rank": int(dr[d - 1]) for d in (1, 2, 3)}}
            for sid, target, rank, dr in zip(dataset.sample_ids, arrays["targets"], ranks, arrays["depth_ranks"])]
    np.savez_compressed(args.output / "candidates.npz", tokens=arrays["tokens"], logps=arrays["logps"],
                        raw_scores=arrays["raw_scores"], search_scores=arrays["search_scores"], depth_ranks=arrays["depth_ranks"])
    np.savez_compressed(args.output / "contexts.npz", sample_ids=np.array(dataset.sample_ids),
                        targets=arrays["targets"], histories=arrays["histories"])
    pq.write_table(pa.Table.from_pylist(rows), args.output / "samples.parquet", compression="zstd")
    assert file_sha256(checkpoint) == before
    meta = {"split": args.split, "samples": len(dataset), "widths": args.widths, "pool_size": args.pool_size,
            "weights": args.weights, "sibling": args.sibling, "penalty_depths": args.penalty_depths,
            "prior_strengths": args.prior_strengths, "suppress_seen_singletons": args.suppress_seen_singletons,
            "prior_history": args.prior_history,
            "prior_train_stats_sha256": file_sha256(args.prior_train_stats) if calibrated else None,
            "batch_size": args.batch_size, "metrics": metrics_from_ranks(ranks),
            "survival_counts": (arrays["depth_ranks"] > 0).sum(0).tolist(),
            "candidate_pool_target_count": int((ranks > 0).sum()), "seconds": seconds, "peak_gpu_mib": peak,
            "checkpoint_sha256": before, "checkpoint_unchanged": True, "sid_sha256": file_sha256(sid_path),
            "data_sha256": file_sha256(source), "gpu": torch.cuda.get_device_name(0), "precision": "fp16",
            "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "implementation_sha256": file_sha256(Path(__file__)), "no_training": True,
            "candidate_logps": "each candidate's own conditionals; start excluded; full-vocabulary normalization",
            "generation_has_no_target_argument": True, "state": "complete"}
    new_json(args.output / "summary.json", meta)
    print(json.dumps(meta), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p.add_argument("--widths", type=int, nargs=3, default=[15, 15, 15])
    p.add_argument("--weights", type=float, nargs=3, default=[1., 1., 1.])
    p.add_argument("--sibling", type=float, default=0.)
    p.add_argument("--penalty-depths", type=int, nargs="*", default=[2])
    p.add_argument("--prior-strengths", type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument("--prior-train-stats", type=Path)
    p.add_argument("--prior-history", type=int, choices=(1, 3), default=1)
    p.add_argument("--suppress-seen-singletons", action="store_true")
    p.add_argument("--pool-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int)
    collect(p.parse_args())


if __name__ == "__main__":
    main()
