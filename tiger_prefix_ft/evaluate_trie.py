"""Compare original Beauty TIGER with a SID Trie or different per-depth Beam widths."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.modules.sid_trie import SIDTrie
from cotiger.core.trainers.tiger_trainer import evaluate
from .beam_diagnosis import BASE, ROOT, ObservedModel, TaggedValidation, collate_tagged, paired, sample_schema, summarize
from .finetune import file_sha256, make_loader, write_json


class CatalogObserved(ObservedModel):
    def __init__(self, tiger, name, catalog, beam_sizes=None):
        super().__init__(tiger, name, beam_sizes)
        self.catalog = catalog

    def generate(self, input_ids, attention_mask, num_beams):
        generated = super().generate(input_ids, attention_mask, num_beams)
        paths = generated.view(len(input_ids), num_beams, 4)[:, :, 1:].cpu().tolist()
        for record, beams in zip(self.records[-len(input_ids):], paths):
            record["illegal_prediction_count"] = sum(tuple(path) not in self.catalog for path in beams)
        return generated


def run(args):
    if args.beam_sizes is not None and args.beam_sizes[-1] < 10:
        raise ValueError("Recall@10 comparison requires final Beam width >= 10")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    checkpoint = BASE / "run/tiger/best_model.pt"
    previous = json.loads((BASE / "run/tiger/results.json").read_text())
    config = json.loads((BASE / "analysis/model_config.json").read_text())
    assert previous["epochs_completed"] == 100
    checkpoint_hash = file_sha256(checkpoint)
    assert checkpoint_hash == "f1628a38656353e115b2739fc9c9c0ffdec74fcaf808cdbb6ef5ea615b0e4760"
    sid_path = BASE / "analysis/item_semantic_ids.npy"
    sids = np.load(sid_path, allow_pickle=False)
    trie = SIDTrie.from_semantic_ids(sids) if args.beam_sizes is None else None
    candidate_name = "original_trie" if args.beam_sizes is None else "beam_" + "_".join(map(str, args.beam_sizes))
    prefix = "trie_evaluation" if args.beam_sizes is None else "beam_evaluation"
    variants = (("original_free", None, None), (candidate_name, trie, args.beam_sizes))
    catalog = {tuple(row) for row in (sids.astype(np.int64) + np.array([1, 257, 513])).tolist()}
    model = Tiger(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to("cuda").eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / f"{prefix}_summary.json"
    report = {"state": "running", "scope": "full_evaluation" if args.all else "smoke",
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
              "training_epochs_completed": previous["epochs_completed"],
              "selected_epoch_one_based": previous["best_epoch_zero_based"] + 1,
              "checkpoint_selection": "validation-best from the original 100-epoch run; not a separate epoch-100 final snapshot",
              "baseline_beam_sizes": [15, 15, 15], "sid_length": 3, "batch_size": args.batch_size, "precision": "fp16",
              "comparison_beam_sizes": args.beam_sizes or [15, 15, 15],
              "comparison_uses_trie": trie is not None,
              "gpu": torch.cuda.get_device_name(0),
              "total_gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "sid_mapping_sha256": file_sha256(sid_path), "catalog_items": len(sids),
              "unique_catalog_sids": len(catalog), "legal_prefix_counts": trie.prefix_counts if trie is not None else None,
              "scoring": "original full-vocabulary cumulative log probabilities",
              "splits": {}, "no_training": True}
    write_json(summary_path, report)
    all_records = []
    started = time.perf_counter()
    for split in ("valid", "test"):
        source = BASE / f"analysis/{split}_samples.parquet"
        dataset = TaggedValidation(source, sids, None if args.all else args.max_samples)
        dataset.sample_ids = [f"{split}_{sample_id.removeprefix('valid_')}" for sample_id in dataset.sample_ids]
        if args.all:
            assert len(dataset) == 22363
        loader = make_loader(dataset, args.batch_size, collate_tagged, False, 0, 42, True)
        result = {"samples": len(dataset), "source_sha256": file_sha256(source),
                  "historical_baseline": previous["best_valid" if split == "valid" else "final_test"], "models": {}}
        report["splits"][split] = result
        records_by_mode = {}
        for name, constraint, widths in variants:
            final_size = widths[-1] if widths is not None else 15
            model.set_sid_trie(constraint)
            observed = CatalogObserved(model, name, catalog, widths).eval()
            torch.cuda.reset_peak_memory_stats()
            phase_started = time.perf_counter()
            metrics = evaluate(observed, observed.batches(loader), torch.device("cuda"), final_size,
                               amp=True, mixed_precision_type="fp16", desc=f"{split} {name} ({len(dataset)})")
            records = observed.records
            assert [row["sample_id"] for row in records] == dataset.sample_ids
            illegal = sum(row["illegal_prediction_count"] for row in records)
            if constraint is not None:
                assert illegal == 0
            summary = summarize(records, metrics, final_size)
            summary.update(seconds=time.perf_counter() - phase_started,
                           beam_sizes=widths or [15, 15, 15], trie_enabled=constraint is not None,
                           peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           illegal_generated_sids=illegal, total_generated_sids=len(records) * final_size,
                           first_batch_passive_replay_equal=observed.reference_checked)
            result["models"][name] = summary
            records_by_mode[name] = records
            all_records.extend({"split": split, **row} for row in records)
            write_json(summary_path, report)
            print(json.dumps({"split": split, "model": name, **summary}), flush=True)
        free, constrained = result["models"]["original_free"], result["models"][candidate_name]
        result["metric_delta_percentage_points"] = {key: 100 * (constrained["metrics"][key] - value)
            for key, value in free["metrics"].items()}
        before, after = records_by_mode["original_free"], records_by_mode[candidate_name]
        result["paired"] = paired(before, after)
        result["paired"].update(
            recovered_top10=sum(not a["hit_at_10"] and b["hit_at_10"] for a, b in zip(before, after)),
            lost_top10=sum(a["hit_at_10"] and not b["hit_at_10"] for a, b in zip(before, after)))
        if args.all:
            result["free_delta_from_historical"] = {key: value - result["historical_baseline"][key]
                for key, value in free["metrics"].items()}
        write_json(summary_path, report)
    schema = sample_schema().append(pa.field("split", pa.string())).append(pa.field("illegal_prediction_count", pa.int8()))
    sample_path = args.output_dir / f"{prefix}_samples.parquet"
    pq.write_table(pa.Table.from_pylist(all_records, schema=schema), sample_path, compression="zstd")
    assert pq.read_table(sample_path).to_pylist() == all_records
    assert file_sha256(checkpoint) == checkpoint_hash
    report.update(state="complete", seconds=time.perf_counter() - started,
                  sample_rows=len(all_records), checkpoint_unchanged=True, samples_sha256=file_sha256(sample_path),
                  source_sha256={name: file_sha256(ROOT / name) for name in (
                      "tiger_prefix_ft/evaluate_trie.py", "tiger_prefix_ft/beam_diagnosis.py",
                      "cotiger/core/models/tiger.py", "cotiger/core/modules/t5.py", "cotiger/core/modules/sid_trie.py",
                      "cotiger/core/trainers/tiger_trainer.py")})
    write_json(summary_path, report)
    print(json.dumps({"state": "complete", "samples": len(all_records), "seconds": report["seconds"],
                      "summary": str(summary_path.resolve())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-samples", type=int, default=64)
    limits.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-sizes", type=int, nargs=3,
                        help="Compare with these widths, disabling Trie in both groups")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
