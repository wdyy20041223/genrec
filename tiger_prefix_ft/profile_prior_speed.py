"""Measure the existing exact calibrated decoder, without changing old files."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import Subset

from cotiger.core.models.tiger import Tiger
from .beam_diagnosis import BASE, TaggedValidation, collate_tagged
from .finetune import make_loader
from .inference_lab import generate_with_prior, new_json
from .inference_priors import PrefixPrior, decode
from .inference_fast import CPUCompactPrior, CSRGPUPrior, fast_search
from .inference_rerank import lookup, sid_keys


from .paths import STUDY


class MeasuredPrior(PrefixPrior):
    def prepare(self, ids, mask):
        with getattr(self, 'timed_scope', record_function)("prior.prepare_D2H"):
            ids, mask = ids.cpu().numpy(), mask.cpu().numpy()
        with record_function("prior.prepare_CPU_context_and_seen"):
            self.contexts, self.seen = [], []
            for tokens, active in zip(ids, mask):
                history = tokens[active.astype(bool)].reshape(-1, 3)
                contexts, used = [], set()
                for age, context in enumerate(sid_keys(history[-self.history_items:])[::-1]):
                    if int(context) not in used:
                        contexts.append((int(context), .5 ** age))
                        used.add(int(context))
                self.contexts.append(contexts)
                keys = np.unique(sid_keys(history))
                sizes = lookup(self.catalog_keys, self.group_sizes, keys)
                self.seen.append(decode(keys[sizes == 1]))

    def __call__(self, depth, parent_paths, probabilities):
        if not self.strengths[depth - 1] and not (depth == 3 and self.suppress_seen):
            return 0.
        b, k, _ = parent_paths.shape
        with record_function("prior.allocate_CPU_adjustment"):
            adjustment = np.zeros((b, k, probabilities.shape[-1]), np.float32)
        with getattr(self, 'timed_scope', record_function)("prior.parent_D2H"):
            parents = parent_paths[:, :, 1:].detach().cpu().numpy()
        alpha = self.strengths[depth - 1]
        with record_function("prior.CPU_transition_lookup_python_loop"):
            for row, contexts in enumerate(self.contexts):
                for context, weight in contexts:
                    if alpha and context in self.by_depth[depth - 1]:
                        tokens, lifts = self.by_depth[depth - 1][context]
                        match = (parents[row, :, None, :] == tokens[None, :, :depth - 1]).all(-1)
                        beam, column = np.nonzero(match)
                        adjustment[row, beam, tokens[column, -1]] += alpha * weight * lifts[column]
        with record_function("prior.CPU_seen_filter"):
            if depth == 3 and self.suppress_seen:
                for row in range(b):
                    tokens = self.seen[row]
                    match = (parents[row, :, None, :] == tokens[None, :, :2]).all(-1)
                    beam, column = np.nonzero(match)
                    adjustment[row, beam, tokens[column, -1]] -= 1000.
        with getattr(self, 'timed_scope', record_function)("prior.adjustment_H2D"):
            return torch.from_numpy(adjustment).to(probabilities.device)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tiger = Tiger(json.loads((BASE / "analysis/model_config.json").read_text())).eval().requires_grad_(False).cuda()
    tiger.load_state_dict(torch.load(BASE / "run/tiger/best_model.pt", map_location="cpu", weights_only=True), strict=True)
    sids = np.load(BASE / "analysis/item_semantic_ids.npy", allow_pickle=False)
    with np.load(STUDY / "train_statistics.npz", allow_pickle=False) as f:
        stats = {k: f[k] for k in f.files}
    calls = []
    original_log1p = np.log1p
    def counted(x, *a, **kw):
        calls.append(np.size(x))
        return original_log1p(x, *a, **kw)
    with patch("numpy.log1p", counted):
        prior = MeasuredPrior(stats, sids + [1, 257, 513], [0., .4, .4], True, 3)
    if args.variant != 'reference':
        prior = CPUCompactPrior(prior) if args.variant == 'fast_cpu' else CSRGPUPrior(prior)
    initial_log_calls = len(calls)
    load_seconds = time.perf_counter() - started
    dataset = TaggedValidation(BASE / "analysis/valid_samples.parquet", sids, None)
    # Preserve the archived batch composition and padding shapes in FP16.
    indices = np.arange(256)
    batches = list(make_loader(Subset(dataset, indices), 64, collate_tagged, False, 0, 42, True))
    widths = [args.beam] * 3
    original = PrefixPrior(stats, sids + [1, 257, 513], [0., .4, .4], True, 3)
    phase = record_function
    def execute(prior):
        output = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for batch in batches:
                with phase("input.H2D"):
                    ids, mask = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
                with record_function("generation.total"):
                    if args.variant == 'reference' or prior is original:
                        generated = generate_with_prior(tiger, ids, mask, prior, widths)
                    else:
                        with phase('prior.prepare_compact'):
                            prior.prepare(ids, mask)
                        generated = fast_search(tiger, ids, mask, widths, prior)
                with phase("result.collect_D2H"):
                    output.append(generated.view(len(ids), args.beam, 4).cpu().numpy())
        return np.concatenate(output)
    expected = execute(original)
    with np.load(STUDY / f"production_benchmark_valid/prior{args.beam}_predictions.npz") as f:
        archived = f[f.files[0]]
    assert np.array_equal(expected[:, :, 1:], archived[indices])
    assert np.array_equal(execute(prior), expected), 'Warm implementation changed outputs'
    timers = []
    @contextmanager
    def scope(name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with record_function(name):
            start.record()
            try:
                yield
            finally:
                end.record()
                timers.append((name, start, end))
    phase = scope
    if isinstance(prior, MeasuredPrior):
        prior.timed_scope = scope
    elif hasattr(prior, 'reference'):
        prior.reference.timed_scope = scope
    if args.variant != 'reference':
        underlying_prior = prior
        class TracedPrior:
            def prepare(self, *a, **kw):
                return underlying_prior.prepare(*a, **kw)

            def __call__(self, *a, **kw):
                with scope('prior.compact_GPU_lookup'):
                    return underlying_prior(*a, **kw)
        prior = TracedPrior()
    forwards = []
    for module, name in [(tiger.model.encoder, "model.encoder"), (tiger.model, "model.decoder_and_logits")]:
        old = module.forward
        def wrap(*a, _old=old, _name=name, **kw):
            with scope(_name):
                return _old(*a, **kw)
        module.forward = wrap
        forwards.append((module, old))
    old_reorder = tiger.model._reorder_cache
    def reorder(*a, **kw):
        with scope("beam.KV_reorder"):
            return old_reorder(*a, **kw)
    tiger.model._reorder_cache = reorder
    tensor_methods = []
    for method, name in [('topk', 'beam.topk'), ('log_softmax', 'beam.fp32_logsoftmax'), ('gather', 'beam.gather')]:
        old_method = getattr(torch.Tensor, method)
        def tensor_wrap(self, *a, _old=old_method, _name=name, **kw):
            with scope(_name):
                return _old(self, *a, **kw)
        setattr(torch.Tensor, method, tensor_wrap)
        tensor_methods.append((method, old_method))
    old_add = torch.Tensor.__add__
    def add(self, *a, **kw):
        if sys._getframe(1).f_code.co_name in ('alternative_search', 'fast_search'):
            with scope('beam.score_and_index_add'):
                return old_add(self, *a, **kw)
        return old_add(self, *a, **kw)
    torch.Tensor.__add__ = add
    from . import inference_fast
    old_self_reorder = inference_fast.reorder_self_cache
    def self_reorder(*a, **kw):
        with scope('beam.self_only_KV_reorder'):
            return old_self_reorder(*a, **kw)
    inference_fast.reorder_self_cache = self_reorder
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    activities = [ProfilerActivity.CPU]
    if ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activities.append(ProfilerActivity.CUDA)
    started = time.perf_counter()
    try:
        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof, patch("numpy.log1p", counted):
            actual = execute(prior)
            torch.cuda.synchronize()
    finally:
        for module, old in forwards:
            module.forward = old
        tiger.model._reorder_cache = old_reorder
        for method, old_method in tensor_methods:
            setattr(torch.Tensor, method, old_method)
        torch.Tensor.__add__ = old_add
        inference_fast.reorder_self_cache = old_self_reorder
    elapsed = time.perf_counter() - started
    assert np.array_equal(actual, expected)
    events = [{"name": item.key, "count": item.count, "cpu_total_ms": item.cpu_time_total/1000,
               "cpu_self_ms": item.self_cpu_time_total/1000, "cuda_total_ms": item.device_time_total/1000,
               "cuda_self_ms": item.self_device_time_total/1000} for item in prof.key_averages()]
    event_timing = {}
    for name, a, b in timers:
        event_timing[name] = event_timing.get(name, 0.) + a.elapsed_time(b)
    prof.export_chrome_trace(str(args.output / "trace.json"))
    started = time.perf_counter()
    np.savez_compressed(args.output / "predictions.npz", tokens=actual, dataset_indices=indices)
    save_seconds = time.perf_counter()-started
    new_json(args.output / "profile.json", dict(state="passed", variant=args.variant, samples=256, batch_size=64, beam=args.beam,
        all_beams_and_order_match_archived=True, instrumentation_matches_original=True,
        profiler_activities=[str(x) for x in activities], loading_seconds=load_seconds,
        profiled_seconds=elapsed, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_gpu_mib=torch.cuda.max_memory_reserved()/2**20,
        log1p_calls_initialization=initial_log_calls, log1p_calls_decoding=len(calls)-initial_log_calls,
        cuda_event_elapsed_ms=event_timing, events=events, result_save_seconds=save_seconds,
        note="Profile overhead is excluded from later benchmarks. CUDA event spans include stream idle. Device totals require CUDA profiler activity; zero device totals with CPU-only activity are unavailable, not zero GPU work. Inclusive CPU/CUDA scopes must not be summed."))
    print(json.dumps(dict(profiled_seconds=elapsed, cuda_event_elapsed_ms=event_timing,
        important_events=[e for e in events if e['name'].startswith(('prior.', 'model.', 'beam.', 'result.', 'input.')) or e['name'] in ['aten::topk','aten::allclose','aten::item','aten::_to_copy','cudaMemcpyAsync','cudaStreamSynchronize']]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam", type=int, choices=(15, 30), default=30)
    parser.add_argument('--variant', choices=('reference','fast_cpu','fast_gpu'), default='reference')
    run(parser.parse_args())
