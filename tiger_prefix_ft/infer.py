"""Run saved Beauty TIGER without Trie, retaining 15 prefixes at each depth by default."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from .beam_diagnosis import BASE, ROOT, TaggedValidation, collate_tagged
from .finetune import file_sha256, make_loader, write_json


def run(args):
    if args.max_samples < 1 or args.batch_size < 1 or min(args.beam_sizes) < 1:
        raise ValueError("Sample count, batch size and Beam size must be positive")
    beam_size = args.beam_sizes[-1]
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    config = json.loads((args.analysis_dir / "model_config.json").read_text())
    sid_path = args.analysis_dir / "item_semantic_ids.npy"
    sids = np.load(sid_path, allow_pickle=False)
    catalog = {tuple(row) for row in (sids.astype(np.int64) + np.array([1, 257, 513])).tolist()}
    checkpoint_hash = file_sha256(args.checkpoint)
    model = Tiger(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False).set_sid_trie(None)
    dataset = TaggedValidation(args.analysis_dir / "valid_samples.parquet", sids, args.max_samples)
    loader = make_loader(dataset, args.batch_size, collate_tagged, False, 0, 42, device.type == "cuda")
    args.output_dir.mkdir(parents=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generated_count, samples, illegal = 0, 0, 0
    started = time.perf_counter()
    context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and args.precision == "fp16" else nullcontext()
    with torch.inference_mode(), context, (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for batch in loader:
            predicted = model.generate(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                       beam_size, beam_sizes=args.beam_sizes)
            predicted = predicted.view(len(batch["sample_ids"]), beam_size, 4)[:, :, 1:].cpu().tolist()
            for sample_id, target, paths in zip(batch["sample_ids"], batch["labels"].tolist(), predicted):
                illegal += sum(tuple(path) not in catalog for path in paths)
                assert len({tuple(path) for path in paths}) == beam_size, "Duplicate Beam paths"
                output.write(json.dumps({"sample_id": sample_id, "target_tokens": target, "predicted_tokens": paths}) + "\n")
                generated_count += len(paths)
                samples += 1
    assert file_sha256(args.checkpoint) == checkpoint_hash
    report = {"state": "complete", "split": "valid", "samples": samples, "generated_sids": generated_count,
              "all_generated_sids_legal": illegal == 0, "illegal_generated_sids": illegal,
              "unique_paths_per_sample": beam_size, "beam_size": beam_size,
              "beam_sizes": args.beam_sizes, "trie_enabled": False, "sid_length": config["sem_id_dim"],
              "catalog_items": len(sids), "unique_catalog_sids": len(catalog),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
              "sid_mapping_sha256": file_sha256(sid_path), "checkpoint_unchanged": True,
              "scoring": "original full-vocabulary cumulative log probabilities; no Trie or position constraints",
              "device": str(device), "precision": args.precision, "seconds": time.perf_counter() - started,
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
    write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=BASE / "analysis")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "run/tiger/best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/tiger_prefix_ft/beauty_beam_15_15_15_inference_20261015")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-sizes", type=int, nargs=3, default=[15, 15, 15])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "no"), default="fp16")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
