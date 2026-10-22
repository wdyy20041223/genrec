"""Evaluate the frozen Beauty TIGER and CoTIGER with the original metrics."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import evaluate
from .data import ValidationDataset, collate_validation, make_loader
from .decoding import CPUCompactPrior, CSRGPUPrior, generate_fast
from .prior import PrefixPrior
from .reference import reference_search


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


class Generator(torch.nn.Module):
    def __init__(self, model, prior, beam, engine):
        super().__init__()
        self.model, self.prior, self.beam, self.engine = model, prior, beam, engine
        self.predictions = []

    def generate(self, input_ids, attention_mask, num_beams):
        assert num_beams == self.beam
        if self.prior is None:
            output = self.model.generate(input_ids, attention_mask, num_beams)
        elif self.engine == 'reference':
            self.prior.prepare(input_ids, attention_mask)
            output = reference_search(self.model, input_ids, attention_mask,
                                      (num_beams,) * 3, adjuster=self.prior)[0]
        else:
            output = generate_fast(self.model, input_ids, attention_mask,
                                   self.prior, widths=(num_beams,) * 3)
        # All methods collect exactly the same final output on CPU.
        self.predictions.append(output.view(len(input_ids), num_beams, 4)
                                [:, :, 1:].cpu().numpy().astype(np.int16))
        return output


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the verified FP16 evaluation protocol')
    if args.batch_size < 1 or args.repeats < 1 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError('Batch size, repeats and sample limit must be positive')
    if args.check_reference and (args.batch_size != 64 or
            args.max_samples is not None and args.max_samples % 64):
        raise ValueError('Archived comparisons require batch64 and whole batches, or the complete split')
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    required = ['best_model.pt', 'model_config.json', 'item_semantic_ids.npy', 'valid_samples.parquet']
    if args.method == 'cotiger':
        required.append('train_statistics.npz')
    missing = [name for name in required if not (args.artifacts / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Beauty assets: {missing}. Download the release bundle first.')
    torch.set_num_threads(2)
    config = json.loads((args.artifacts / 'model_config.json').read_text(encoding='utf-8'))
    if config['sem_id_dim'] != 3 or config['vocab_size'] != 769:
        raise ValueError('This protocol requires the original three-token Beauty checkpoint')
    checkpoint_hash = file_sha256(args.artifacts / 'best_model.pt')
    model = Tiger(config).eval().requires_grad_(False).cuda()
    model.load_state_dict(torch.load(args.artifacts / 'best_model.pt',
                                    map_location='cpu', weights_only=True), strict=True)
    sids = np.load(args.artifacts / 'item_semantic_ids.npy', allow_pickle=False)
    prior = None
    if args.method == 'cotiger':
        with np.load(args.artifacts / 'train_statistics.npz', allow_pickle=False) as archive:
            stats = {key: archive[key] for key in archive.files}
        prior = PrefixPrior(stats, sids + [1, 257, 513], [0., .4, .4], True, 3)
        if args.engine == 'fast':
            prior = (CSRGPUPrior if args.prior_backend == 'gpu' else CPUCompactPrior)(prior)
    dataset = ValidationDataset(args.artifacts / 'valid_samples.parquet', sids, args.max_samples)
    loader = make_loader(dataset, args.batch_size, collate_validation, False, 0, 42, True)
    adapter = Generator(model, prior, args.beam, args.engine).eval()
    args.output.mkdir(parents=True)
    batch = next(iter(loader))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        adapter.generate(batch['input_ids'].cuda(), batch['attention_mask'].cuda(), args.beam)
    torch.cuda.synchronize()
    adapter.predictions.clear()
    if args.warmup_full:
        evaluate(adapter, loader, torch.device('cuda'), args.beam, True, 'fp16', 'Full warmup')
        adapter.predictions.clear()
    reference = None
    if args.check_reference:
        name = 'prior' if args.method == 'cotiger' else 'original'
        with np.load(args.artifacts / f'reference/{name}{args.beam}_predictions.npz', allow_pickle=False) as archive:
            reference = archive['tokens'][:len(dataset)]
    runs = []
    for repeat in range(args.repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = evaluate(adapter, loader, torch.device('cuda'), args.beam, True, 'fp16', args.method)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        predictions = np.concatenate(adapter.predictions)
        if reference is not None and not np.array_equal(predictions, reference):
            raise AssertionError('Ranked SID outputs differ from the archived reference')
        runs.append(dict(repeat=repeat,seconds=elapsed,samples_per_second=len(dataset)/elapsed,
                         peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,metrics=metrics,
                         all_beams_exact=reference is not None))
        if repeat == 0:
            np.savez_compressed(args.output / 'predictions.npz', tokens=predictions,
                                sample_ids=np.array(dataset.sample_ids))
        adapter.predictions.clear()
    assert file_sha256(args.artifacts / 'best_model.pt') == checkpoint_hash
    result = dict(method=args.method,engine=args.engine if prior is not None else 'original',
                  beam=args.beam,samples=len(dataset),batch_size=args.batch_size,
                  prior_backend=args.prior_backend if prior is not None else None,
                  checkpoint_sha256=checkpoint_hash,split='valid',sid_length=3,
                  precision='FP32 parameters, FP16 autocast, FP32 LogSoftmax/raw scores',
                  timing='Warmed original evaluator and identical CPU output capture; excludes loading and saving',
                  runs=runs)
    with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts/beauty'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=('tiger','cotiger'),default='cotiger')
    parser.add_argument('--engine',choices=('fast','reference'),default='fast')
    parser.add_argument('--prior-backend',choices=('gpu','cpu'),default='gpu')
    parser.add_argument('--beam',type=int,choices=(15,30),default=15)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--warmup-full',action='store_true')
    parser.add_argument('--check-reference',action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
