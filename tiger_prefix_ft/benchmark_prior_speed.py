"""Identical-batch, archived-output-checked engineering benchmarks on Validation."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import psutil
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import evaluate
from .beam_diagnosis import BASE, TaggedValidation, collate_tagged
from .finetune import file_sha256, make_loader
from .inference_fast import CPUCompactPrior, CSRGPUPrior, fast_search, reorder_self_cache
from .inference_lab import CHECKPOINT_HASH, alternative_search, new_json
from .inference_priors import PrefixPrior
from .inference_weight_cache import FrozenLinearCastCache
from .profile_prior_speed import STUDY


class Observed(torch.nn.Module):
    def __init__(self, tiger, prior, beam, engine):
        super().__init__()
        self.tiger, self.prior, self.beam, self.engine = tiger, prior, beam, engine
        self.predictions = []
        self.phase_seconds = dict(prepare=0., decode_and_gpu_wait=0., result_collection=0.)

    def generate(self, input_ids, attention_mask, num_beams):
        ids, mask = input_ids, attention_mask
        assert num_beams == self.beam
        start = time.perf_counter()
        if self.prior is not None:
            self.prior.prepare(ids, mask)
        ready = time.perf_counter()
        if self.engine == 'original':
            result = self.tiger.generate(ids, mask, self.beam)
        elif self.engine == 'kv_only':
            reorder = self.tiger.model._reorder_cache
            self.tiger.model._reorder_cache = reorder_self_cache
            try:
                result = alternative_search(self.tiger, ids, mask, [self.beam]*3, adjuster=self.prior)[0]
            finally:
                self.tiger.model._reorder_cache = reorder
        elif self.engine == 'reference':
            result = alternative_search(self.tiger, ids, mask, [self.beam]*3, adjuster=self.prior)[0]
        else:
            result = fast_search(self.tiger, ids, mask, [self.beam]*3, self.prior)
        # Collection is identical for all variants. It is also the batch fence.
        launched = time.perf_counter()
        output = result.view(len(ids), self.beam, 4)[:, :, 1:].cpu().numpy().astype(np.int16)
        self.predictions.append(output)
        done = time.perf_counter()
        self.phase_seconds['prepare'] += ready-start
        self.phase_seconds['decode_and_gpu_wait'] += launched-ready
        self.phase_seconds['result_collection'] += done-launched
        return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source_paths = [BASE/'run/tiger/best_model.pt', BASE/'analysis/item_semantic_ids.npy',
                    BASE/'analysis/model_config.json', BASE/'analysis/valid_samples.parquet', STUDY/'train_statistics.npz',
                    Path(__file__), Path(__file__).with_name('inference_fast.py'), Path(__file__).with_name('inference_lab.py'),
                    Path(__file__).with_name('inference_priors.py'), Path(__file__).with_name('inference_weight_cache.py')]
    sources = {str(p):file_sha256(p) for p in source_paths}
    snapshot = args.output/'code'
    snapshot.mkdir()
    for source in source_paths:
        if source.suffix == '.py':
            shutil.copyfile(source, snapshot/source.name)
    assert sources[str(source_paths[0])] == CHECKPOINT_HASH
    started = time.perf_counter()
    tiger = Tiger(json.loads(source_paths[2].read_text())).eval().requires_grad_(False).cuda()
    tiger.load_state_dict(torch.load(source_paths[0], weights_only=True, map_location='cpu'), strict=True)
    sids = np.load(source_paths[1], allow_pickle=False)
    with np.load(STUDY/'train_statistics.npz', allow_pickle=False) as archive:
        stats = {k:archive[k] for k in archive.files}
    dataset = TaggedValidation(source_paths[3], sids, args.max_samples)
    loader = make_loader(dataset, 64, collate_tagged, False, 0, 42, True)
    first = next(iter(loader))
    ids, mask = first['input_ids'].cuda(), first['attention_mask'].cuda()
    loading_seconds = time.perf_counter()-started
    summaries = []
    occurrences = {}
    for variant in args.variants:
        prior_started = time.perf_counter()
        reference = None if variant == 'original' else PrefixPrior(stats, sids+[1,257,513], [0.,.4,.4], True, 3)
        prior = reference
        if 'cpu' in variant:
            prior = CPUCompactPrior(reference)
        elif 'gpu' in variant:
            prior = CSRGPUPrior(reference)
            reference = None
        engine = 'original' if variant == 'original' else 'kv_only' if variant.endswith('kv_only') else 'fast' if variant.startswith('fast_') else 'reference'
        linear_cache = FrozenLinearCastCache(tiger) if variant.endswith('_linear') else None
        setup_seconds = time.perf_counter()-prior_started
        adapter = Observed(tiger, prior, args.beam, engine).eval()
        # Separate the first CUDA call from warmed full-dataset timing.
        cold_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            adapter.generate(ids, mask, args.beam)
        torch.cuda.synchronize()
        first_call_seconds = time.perf_counter()-cold_started
        # Real-data correctness before any full-dataset benchmark of this variant.
        baseline = STUDY / f'production_benchmark_valid/{"original" if variant=="original" else "prior"}{args.beam}_predictions.npz'
        with np.load(baseline, allow_pickle=False) as archive:
            expected = archive['tokens'][:len(dataset)]
        assert np.array_equal(adapter.predictions[0], expected[:len(ids)]), (variant, 'first batch changed')
        warmup_seconds = None
        if args.warmup_full:
            adapter.predictions.clear()
            warmup_started = time.perf_counter()
            evaluate(adapter, loader, torch.device('cuda'), args.beam, True, 'fp16', f'{variant} full warmup')
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter()-warmup_started
            assert np.array_equal(np.concatenate(adapter.predictions), expected)
        for _ in range(args.repeats):
            repeat = occurrences.get(variant, 0)
            occurrences[variant] = repeat + 1
            adapter.predictions.clear()
            adapter.phase_seconds = dict.fromkeys(adapter.phase_seconds, 0.)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            process = psutil.Process()
            rss_start = process.memory_info().rss
            rss = [rss_start]
            stopped = threading.Event()
            def sample_rss():
                while not stopped.wait(.025):
                    rss.append(process.memory_info().rss)
            sampler = None
            if args.sample_rss:
                sampler = threading.Thread(target=sample_rss, daemon=True)
                sampler.start()
            started, cpu_started = time.perf_counter(), time.process_time()
            metrics = evaluate(adapter, loader, torch.device('cuda'), args.beam, True, 'fp16', f'{variant} b{args.beam} r{repeat}')
            torch.cuda.synchronize()
            seconds, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu_started
            peak = torch.cuda.max_memory_allocated()/2**20
            stopped.set()
            if sampler is not None:
                sampler.join()
            rss_end = process.memory_info().rss
            actual = np.concatenate(adapter.predictions)
            previous = json.loads((STUDY/f'production_benchmark_valid/{"original" if variant=="original" else "prior"}{args.beam}.json').read_text())
            if args.max_samples is None:
                assert all(abs(metrics[key]-value)<1e-7 for key,value in previous['metrics'].items())
            top10_different = np.any(actual[:, :10] != expected[:, :10], axis=(1,2))
            all_different = np.any(actual != expected, axis=(1,2))
            item = dict(variant=variant, beam=args.beam, repeat=repeat, samples=len(dataset), batch_size=64,
                seconds=seconds, cpu_process_seconds=cpu_seconds, samples_per_second=len(dataset)/seconds,
                ms_per_sample=seconds/len(dataset)*1000, ms_per_batch=seconds/len(loader)*1000,
                peak_gpu_mib=peak, rss_start_mib=rss_start/2**20, rss_end_mib=rss_end/2**20,
                peak_rss_mib=max(rss)/2**20 if args.sample_rss else None,
                periodic_rss_sampling=args.sample_rss,
                full_dataset_warmup_seconds=warmup_seconds,
                peak_reserved_gpu_mib=torch.cuda.max_memory_reserved()/2**20,
                prior_setup_seconds=setup_seconds, first_cuda_call_seconds=first_call_seconds, model_data_loading_seconds=loading_seconds,
                cached_linear_mib=0. if linear_cache is None else linear_cache.bytes/2**20,
                mismatched_top10_samples=int(top10_different.sum()), mismatched_all_beams_samples=int(all_different.sum()),
                exact_top10_sid_and_order=not bool(top10_different.any()), metrics=metrics, cpu_wall_phase_seconds=adapter.phase_seconds.copy(),
                phase_note='Host scopes overlap asynchronous CUDA; collection may include remaining GPU wait. Use profiler/events for attribution.')
            save_started = time.perf_counter()
            if repeat == 0:
                np.savez_compressed(args.output/f'{variant}_predictions.npz', tokens=actual, sample_ids=np.array(dataset.sample_ids))
            item['result_save_seconds'] = time.perf_counter()-save_started
            new_json(args.output/f'{variant}_{repeat}.json', item)
            summaries.append(item)
            print(json.dumps(item), flush=True)
            assert not top10_different.any(), (variant, 'Top10 changed; NOT a lossless candidate')
            assert not all_different.any(), (variant, 'A lower-ranked Beam changed')
        if linear_cache is not None:
            linear_cache.close()
        del adapter, prior, reference, linear_cache
        gc.collect()
        torch.cuda.empty_cache()
    assert all(file_sha256(Path(p))==h for p,h in sources.items())
    new_json(args.output/'summary.json', dict(state='passed', source_sha256=sources, profiles=summaries,
        gpu=torch.cuda.get_device_name(0), gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
        torch=torch.__version__, cuda=torch.version.cuda, precision='fp16 model; fp32 log_softmax and scores',
        weights_frozen=all(not p.requires_grad for p in tiger.parameters()),
        timing_scope='Same original evaluator, full batches, same final CPU output capture, model/prior setup and saving excluded; first call recorded separately.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--beam', type=int, choices=(15,30), default=15)
    p.add_argument('--variants', nargs='+', choices=['original','reference','cpu_table','gpu_csr','gpu_csr_kv_only','fast_cpu_table','fast_gpu_csr','fast_cpu_linear','fast_gpu_linear'], default=['reference','fast_cpu_table','fast_gpu_csr'])
    p.add_argument('--max-samples', type=int)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--warmup-full', action='store_true')
    p.add_argument('--sample-rss', action='store_true', help='Optional diagnostic thread; excluded from primary timing by default')
    args=p.parse_args()
    assert args.repeats > 0
    run(args)
