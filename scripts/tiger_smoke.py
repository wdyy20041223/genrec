"""Bounded Beauty smoke run: two RQ-VAE steps and two TIGER steps.

Uses a tiny isolated fixture made from existing Beauty reviews and cached RPG
Sentence-T5-XL vectors. It never encodes the full catalog or runs a benchmark.
The fixture checks pipeline wiring; its metrics are not reproduction results.
"""
import argparse
import gzip
import json
import subprocess
import sys
import time
import hashlib
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(output):
    source = ROOT / 'dataset/amazon'
    reviews_path = source / 'raw/beauty/reviews_Beauty_5.json.gz'
    embedding_path = source / 'processed/beauty/sentence-t5-xl.sent_emb.npy'
    if not reviews_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError('Smoke run requires existing Beauty reviews and cached Sentence-T5-XL vectors. No download is performed.')
    records = []
    original_order = OrderedDict()
    by_user = defaultdict(list)
    with gzip.open(reviews_path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index >= 8192:
                break
            row = json.loads(line)
            if not row.get('asin') or not row.get('reviewerID'):
                continue
            original_order.setdefault(row['asin'], len(original_order))
            records.append(row)
            by_user[row['reviewerID']].append(row)
    users = [u for u, rows in by_user.items() if len(rows) >= 7][:8]
    if len(users) < 8 or len(original_order) < 320:
        raise RuntimeError('The bounded review prefix does not contain enough fixture data.')
    selected = []
    for user in users:
        selected.extend(by_user[user][:7])
    catalog = OrderedDict()
    for row in selected:
        catalog.setdefault(row['asin'], len(catalog))
    for asin in original_order:
        if len(catalog) >= 320:
            break
        catalog.setdefault(asin, len(catalog))
    vectors = np.load(embedding_path, mmap_mode='r')
    embeddings = np.asarray(vectors[[original_order[asin] for asin in catalog]], dtype=np.float32)
    raw = output / 'data/amazon/raw/beauty'
    processed = output / 'data/amazon/processed/beauty'
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    with gzip.open(raw / 'reviews_Beauty_5.json.gz', 'wt', encoding='utf-8') as stream:
        for row in selected:
            stream.write(json.dumps(row) + '\n')
    # Cached embeddings make metadata/text encoding unnecessary in this fixture.
    with gzip.open(raw / 'meta_Beauty.json.gz', 'wt', encoding='utf-8'):
        pass
    pd.DataFrame({'ItemID': np.arange(1, len(embeddings) + 1),
                  'embedding': list(embeddings)}).to_parquet(processed / 'item_emb_sentence-t5-xl.parquet', index=False)
    provenance = {'users': len(users), 'review_rows': len(selected), 'items': len(catalog),
                  'input_dim': embeddings.shape[1], 'source_prefix_limit': 8192,
                  'source_reviews': str(reviews_path), 'source_embeddings': str(embedding_path),
                  'catalog_asins_in_order': list(catalog), 'quality_evaluation': False}
    (output / 'fixture.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return output / 'data/amazon', provenance


def run_stage(module, config, overrides, output):
    command = [sys.executable, '-u', '-m', module, config, '--split', 'beauty']
    for binding in overrides:
        command.extend(['--gin', binding])
    (output / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    started = time.perf_counter()
    with (output / 'console.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    return time.perf_counter() - started


def micro_timing(report_path):
    # Two input histories, one warmup and three measurements per execution path.
    from cotiger.core.data.schemas import SeqData
    from cotiger.core.models.tiger import Tiger
    from cotiger.core.trainers.tiger_trainer import t5_collate, autocast_context
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        return {'skipped': 'CUDA is unavailable'}
    config = dict(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
                  num_heads=6, d_kv=64, dropout_rate=0.0, vocab_size=769, sem_id_dim=3)
    torch.manual_seed(42)
    reference = Tiger({**config, 'first_beam_only': False}).to(device)
    optimized = Tiger({**config, 'use_sdpa': True}).to(device)
    optimized.load_state_dict(reference.state_dict())
    samples = [SeqData(0, [12, 45, 7] * 6, [15, 18, 2]),
               SeqData(1, [13, 42, 6] * 3, [14, 19, 3])]
    timings = {}
    for name, model, amp, dynamic in [('fp32_fixed_eager', reference, False, False),
                                      ('fp16_dynamic_sdpa', optimized, True, True)]:
        reference.zero_grad(set_to_none=True)
        optimized.zero_grad(set_to_none=True)
        batch = t5_collate(samples, max_history_tokens=150, dynamic_padding=dynamic, pad_to_multiple_of=8)
        inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask', 'labels')}
        milliseconds = []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(4):
            model.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(device, amp, 'fp16'):
                loss, _ = model(**inputs)
            loss.backward()
            end.record()
            end.synchronize()
            if not torch.isfinite(loss):
                raise FloatingPointError('Micro timing loss is not finite.')
            if iteration:
                milliseconds.append(start.elapsed_time(end))
        timings[name] = {'batch_size': 2, 'history_tokens': inputs['input_ids'].shape[1],
                         'forward_backward_ms': milliseconds, 'median_ms': float(np.median(milliseconds)),
                         'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Micro timing gradients are not finite.'
        model.eval()
        generation_ms = []
        with torch.inference_mode():
            for iteration in range(4):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with autocast_context(device, amp, 'fp16'):
                    generated = model.generate(inputs['input_ids'], inputs['attention_mask'], num_beams=15)
                end.record()
                end.synchronize()
                assert generated.shape == (30, 4)
                if iteration:
                    generation_ms.append(start.elapsed_time(end))
        timings[name]['beam15_generation_ms'] = generation_ms
        timings[name]['beam15_generation_median_ms'] = float(np.median(generation_ms))
    timings['device'] = torch.cuda.get_device_name(0)
    timings['scope'] = 'Tiny timing only; not a V100 throughput estimate or accuracy comparison.'
    report_path.write_text(json.dumps(timings, indent=2), encoding='utf-8')
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()
    output = (ROOT / (args.output or f'out/tiger_speed_smoke/run_{time.strftime("%Y%m%d_%H%M%S")}')).resolve()
    # Refuse paths outside the checkout and existing runs; never overwrite artifacts.
    output.relative_to(ROOT)
    document_dir = ROOT / 'docs/generated'
    report_prefix = f'tiger-smoke-{output.name.removeprefix("run_")}'
    report_path = document_dir / f'{report_prefix}-report.json'
    timing_path = document_dir / f'{report_prefix}-micro-timing.json'
    for path in (report_path, timing_path):
        if path.exists():
            raise FileExistsError(f'Document already exists: {path}')
    document_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    data, fixture = make_fixture(output)
    rq_dir = output / 'rqvae'
    tiger_dir = output / 'tiger'
    rq_dir.mkdir()
    tiger_dir.mkdir()
    common = [f'train.dataset_folder={data.as_posix()!r}', 'train.wandb_logging=False',
              'train.num_workers=0']
    rq_seconds = run_stage('cotiger.core.trainers.rqvae_trainer', 'config/tiger/amazon/rqvae.gin',
                          common + ['train.epochs=2', 'train.warmup_epochs=0',
                                    'train.use_kmeans_init=False',
                                    f'train.save_dir_root={rq_dir.as_posix()!r}'], rq_dir)
    checkpoint = rq_dir / 'checkpoint_epoch_1.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 1 and state['scaler'], 'Final RQ-VAE checkpoint/scaler is missing.'
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    assert any(torch.count_nonzero(values.get('exp_avg', torch.tensor(0))) > 0
               for values in state['optimizer']['state'].values()), 'RQ-VAE optimizer never updated.'
    tiger_seconds = run_stage('cotiger.core.trainers.tiger_trainer', 'config/tiger/amazon/tiger.gin',
                             common + ['train.epochs=1', 'train.batch_size=16', 'train.infer_batch_size=4',
                                       'train.max_train_samples=32', 'train.max_eval_samples=8',
                                       f'train.pretrained_rqvae_path={checkpoint.as_posix()!r}',
                                       f'train.save_dir_root={tiger_dir.as_posix()!r}'], tiger_dir)
    result = json.loads((tiger_dir / 'results.json').read_text(encoding='utf-8'))
    assert result['epochs_completed'] == 1 and result['train_samples'] == 32
    assert result['best_epoch_zero_based'] == 0 and result['final_test']
    assert result['optimizer_updates'] > 0, 'All TIGER steps were skipped by GradScaler.'
    tiger_weights = torch.load(tiger_dir / 'best_model.pt', map_location='cpu', weights_only=True)
    assert all(torch.isfinite(value).all() for value in tiger_weights.values())
    timing = None if args.skip_timing else micro_timing(timing_path)
    report = {'status': 'passed', 'fixture': {k: v for k, v in fixture.items() if k != 'catalog_asins_in_order'},
              'rqvae_optimizer_steps_requested': 2, 'tiger_optimizer_steps_requested': 2,
              'rqvae_seconds_including_startup': rq_seconds, 'tiger_seconds_including_startup': tiger_seconds,
              'production_codebook_size': 256, 'production_tiger_architecture': True,
              'bulk_kmeans_init_skipped': True, 'full_sentence_encoding_skipped': True,
              'rqvae_weights_finite': True, 'rqvae_optimizer_updated': True,
              'tiger_weights_finite': True, 'tiger': result, 'micro_timing': timing,
              'accuracy_claim': 'No full-data training or accuracy conclusion.'}
    source_files = ['cotiger/core/trainers/rqvae_trainer.py', 'cotiger/core/trainers/tiger_trainer.py',
                    'cotiger/core/models/rqvae.py', 'cotiger/core/models/tiger.py', 'cotiger/core/modules/t5.py',
                    'cotiger/core/data/amazon.py', 'config/tiger/amazon/rqvae.gin', 'config/tiger/amazon/tiger.gin']
    report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(report_path),
                      'rqvae_seconds': rq_seconds, 'tiger_seconds': tiger_seconds, 'timing': timing}, indent=2))


if __name__ == '__main__':
    main()
