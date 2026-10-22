"""Read-only, per-depth TIGER Beam diagnosis on Beauty Validation."""
import argparse
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import calculate_pos_index, evaluate
from .finetune import EvaluationDataset, collate_evaluation, file_sha256, make_loader, write_json

from .paths import ROOT, BASE
REPORTS = ROOT / "docs/generated/tiger_prefix_ft_beauty_20261013"


class PrefixRecorder:
    """Observe actual Top-K decisions; never retain full-vocabulary tensors."""
    def __init__(self, targets):
        self.targets = targets
        self.steps = []
        self.previous_tokens = None
        self.previous_scores = None

    def __call__(self, **trace):
        depth, size, beams = trace["depth"], trace["batch_size"], trace["num_beams"]
        assert depth == len(self.steps) + 1 and depth <= 3
        parents = trace["parent_tokens"].view(size, beams, depth)
        parent_scores = trace["parent_scores"]
        scores = trace["selected_cumulative_scores"]
        probabilities = trace["next_token_logprobs"]
        vocab = probabilities.shape[-1]
        parent_index = trace["selected_flat_indices"] // vocab
        tokens = trace["selected_flat_indices"] % vocab
        rows = torch.arange(size, device=parents.device)
        assert (parents[:, :, 0] == 0).all() and torch.isfinite(scores).all()
        assert (scores[:, :-1] >= scores[:, 1:]).all()
        if self.previous_tokens is not None:
            assert torch.equal(parents, self.previous_tokens), "Beam parent paths changed"
            assert torch.equal(parent_scores, self.previous_scores), "Parent scores differ"
        else:
            assert (parent_index == 0).all(), "Inactive initial Beam selected"
        if trace["first_beam_only"]:
            probabilities = probabilities.view(size, 1, vocab)
            selected_logp = probabilities[:, 0].gather(1, tokens)
        else:
            probabilities = probabilities.view(size, beams, vocab)
            selected_logp = probabilities[rows[:, None], parent_index, tokens]
        recovered_scores = parent_scores.gather(1, parent_index) + selected_logp
        assert torch.equal(recovered_scores, scores), "Wrong-depth cumulative score"
        chosen_parents = parents.gather(1, parent_index[:, :, None].expand(-1, -1, depth))
        selected = torch.cat((chosen_parents, tokens[:, :, None]), dim=-1)
        match = (selected[:, :, 1:] == self.targets[:, None, :depth]).all(-1)
        assert (match.sum(1) <= 1).all()
        survived = match.any(1)
        rank = match.long().argmax(1) + 1
        gold_score = scores[rows, rank - 1]

        gold_parent = (parents[:, :, 1:] == self.targets[:, None, :depth - 1]).all(-1)
        if depth == 1:
            gold_parent[:, 1:] = False  # Only initial Beam 0 is active.
        parent_alive = gold_parent.any(1)
        if self.steps:
            assert torch.equal(parent_alive, self.steps[-1]["survived"])
            assert not (survived & ~self.steps[-1]["survived"]).any(), "Gold path resurrected"
        candidate = scores.new_full((size,), float("nan"))
        valid_rows = parent_alive.nonzero(as_tuple=True)[0]
        parent = gold_parent.long().argmax(1)[valid_rows]
        probability_parent = torch.zeros_like(parent) if trace["first_beam_only"] else parent
        candidate[valid_rows] = (parent_scores[valid_rows, parent] +
            probabilities[valid_rows, probability_parent, self.targets[valid_rows, depth - 1]])
        assert torch.isfinite(candidate[parent_alive]).all()
        assert torch.equal(candidate[survived], gold_score[survived])
        if self.steps:
            previous = self.steps[-1]["score"]
            assert (gold_score[survived] <= previous[survived] + 1e-5).all()
        self.steps.append({"survived": survived, "rank": rank, "score": gold_score,
                           "candidate": candidate, "margin": candidate - scores[:, -1]})
        self.previous_tokens, self.previous_scores = selected, scores

    def finish(self, generated, sample_ids, model_name):
        assert len(self.steps) == 3
        assert torch.equal(generated, self.previous_tokens.flatten(0, 1)), "Final Beam reconstruction differs"
        final = generated.view(len(sample_ids), -1, 4)[:, :, 1:]
        hits = calculate_pos_index(final, self.targets, maxk=final.shape[1])
        assert torch.equal(hits.any(1), self.steps[-1]["survived"])
        values = [{key: value.detach().cpu().tolist() for key, value in step.items()} for step in self.steps]
        targets, hit10 = self.targets.cpu().tolist(), hits[:, :10].any(1).cpu().tolist()
        result = []
        for row, sample_id in enumerate(sample_ids):
            alive = [step["survived"][row] for step in values]
            record = {"sample_id": sample_id, "model_name": model_name, "target_tokens": targets[row],
                      "first_prune_depth": next((d for d, ok in enumerate(alive, 1) if not ok), 0),
                      "hit_at_10": hit10[row]}
            for depth, step in enumerate(values, 1):
                ok = step["survived"][row]
                record[f"survived_depth{depth}"] = ok
                record[f"gold_rank_depth{depth}"] = step["rank"][row] if ok else None
                record[f"gold_score_depth{depth}"] = step["score"][row] if ok else None
                for key in ("candidate", "margin"):
                    value = step[key][row]
                    record[f"gold_{key}_depth{depth}"] = value if math.isfinite(value) else None
            result.append(record)
        return result


class TaggedValidation(EvaluationDataset):
    def __init__(self, path, item_sids, limit):
        super().__init__(path, item_sids, max_history_items=50, limit=limit)
        indices = pq.read_table(path, columns=["sample_index"])["sample_index"].to_pylist()[:len(self)]
        self.sample_ids = [f"valid_{index:07d}" for index in indices]
        assert len(set(self.sample_ids)) == len(self)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["sample_id"] = self.sample_ids[index]
        return row


def collate_tagged(rows):
    batch = collate_evaluation(rows)
    batch["sample_ids"] = [row["sample_id"] for row in rows]
    return batch


class ObservedModel(torch.nn.Module):
    def __init__(self, tiger, name, beam_sizes=None):
        super().__init__()
        self.tiger, self.name, self.records = tiger, name, []
        self.beam_sizes = beam_sizes
        self.reference_checked = False

    def batches(self, loader):
        for batch in loader:
            self.sample_ids, self.labels = batch["sample_ids"], batch["labels"]
            yield batch

    def generate(self, input_ids, attention_mask, num_beams):
        assert num_beams >= 10 and not hasattr(self.tiger.model, "_beam_trace_recorder")
        options = {"num_beams": num_beams}
        if self.beam_sizes is not None:
            assert self.beam_sizes[-1] == num_beams
            options["beam_sizes"] = self.beam_sizes
        # One extra first-batch replay proves that observing does not change output.
        reference = None
        if not self.reference_checked:
            reference = self.tiger.generate(input_ids, attention_mask, **options)
        recorder = PrefixRecorder(self.labels.to(input_ids.device))
        self.tiger.model._beam_trace_recorder = recorder
        try:
            generated = self.tiger.generate(input_ids, attention_mask, **options)
        finally:
            del self.tiger.model._beam_trace_recorder
        if reference is not None:
            assert torch.equal(reference, generated), "Recorder changed generated Beams"
            self.reference_checked = True
        self.records.extend(recorder.finish(generated, self.sample_ids, self.name))
        return generated


def summarize(records, metrics, final_beam_size=15):
    total = len(records)
    survival = {str(d): sum(row[f"survived_depth{d}"] for row in records) for d in (1, 2, 3)}
    pruned = {str(d): sum(row["first_prune_depth"] == d for row in records) for d in (0, 1, 2, 3)}
    assert sum(pruned.values()) == total and pruned["0"] == survival["3"]
    ranks = [row["gold_rank_depth3"] for row in records]
    discounts = (1 / torch.log2(torch.arange(2, final_beam_size + 2).float())).tolist()
    for k in (5, 10):
        recall = sum(rank is not None and rank <= k for rank in ranks) / total
        ndcg = sum(discounts[rank - 1] for rank in ranks if rank is not None and rank <= k) / total
        assert math.isclose(recall, metrics[f"Recall@{k}"], abs_tol=1e-7)
        assert math.isclose(ndcg, metrics[f"NDCG@{k}"], abs_tol=1e-7)
    return {"samples": total, "prefix_survival_counts": survival,
            "prefix_survival_rates": {d: count / total for d, count in survival.items()},
            "first_prune_counts": pruned, "first_prune_rates": {d: count / total for d, count in pruned.items()},
            "conditional_survival_rates": {str(d): survival[str(d)] / incoming if incoming else None
                for d, incoming in ((1, total), (2, survival["1"]), (3, survival["2"]))},
            "metrics": metrics,
            "final_beam_size": final_beam_size,
            "target_in_final_top15_rate": sum(rank is not None and rank <= 15 for rank in ranks) / total,
            "target_in_final_beam_rate": survival["3"] / total,
            "checks": {"parent_paths_and_scores": True, "final_beams_reconstructed": True,
                       "monotone_gold_survival": True, "metrics_match_original_evaluator": True}}


def paired(original, modified):
    assert [row["sample_id"] for row in original] == [row["sample_id"] for row in modified]
    lost = [b for a, b in zip(original, modified) if a["survived_depth3"] and not b["survived_depth3"]]
    recovered = [a for a, b in zip(original, modified) if not a["survived_depth3"] and b["survived_depth3"]]
    top15 = lambda row: row["gold_rank_depth3"] is not None and row["gold_rank_depth3"] <= 15
    return {"original_keeps_modified_prunes": len(lost), "original_prunes_modified_keeps": len(recovered),
            "net_final_beam_hits": len(recovered) - len(lost),
            "net_top15_hits": sum(top15(row) for row in modified) - sum(top15(row) for row in original),
            "lost_targets_first_prune_depth": {str(d): sum(r["first_prune_depth"] == d for r in lost) for d in (1, 2, 3)},
            "recovered_targets_original_first_prune_depth": {str(d): sum(r["first_prune_depth"] == d for r in recovered) for d in (1, 2, 3)}}


def sample_schema():
    fields = [pa.field("sample_id", pa.string()), pa.field("model_name", pa.string()),
              pa.field("target_tokens", pa.list_(pa.int16(), 3)),
              pa.field("first_prune_depth", pa.int8()), pa.field("hit_at_10", pa.bool_())]
    for d in (1, 2, 3):
        fields.extend([pa.field(f"survived_depth{d}", pa.bool_()), pa.field(f"gold_rank_depth{d}", pa.int8()),
                       pa.field(f"gold_score_depth{d}", pa.float32()),
                       pa.field(f"gold_candidate_depth{d}", pa.float32()), pa.field(f"gold_margin_depth{d}", pa.float32())])
    return pa.schema(fields)


def run(args):
    config = json.loads((args.analysis_dir / "model_config.json").read_text())
    ce = json.loads(args.ce_report.read_text())
    h6 = json.loads(args.h6_report.read_text())
    assert ce["best_epoch"] == 9 and h6["best_epoch"] == 1
    assert config == ce["model_config"] == h6["model_config"]
    assert config["sem_id_dim"] == 3 and config["vocab_size"] == 769
    assert config["pad_token_id"] == config["eos_token_id"] == 0
    if "_beam_trace_recorder" not in inspect.getsource(Tiger(config).model._beam_search):
        raise RuntimeError("T5 passive Beam hook is missing; use the source copy described in beam_diagnosis.md")
    models = [("original", args.original_checkpoint, 0),
              ("ce_only_best", args.finetune_dir / "ce_only/best_model.pt", ce["best_epoch"]),
              ("h6_epoch1", args.finetune_dir / "h6_mixed/best_model.pt", 1),
              ("h6_epoch3", args.finetune_dir / "h6_mixed/checkpoint_epoch_3.pt", 3),
              ("h6_epoch10", args.finetune_dir / "h6_mixed/checkpoint_epoch_10.pt", 10)]
    for _, path, _ in models:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("Sample limit and batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new diagnosis directory: {args.output_dir}")
    path = args.analysis_dir / "valid_samples.parquet"
    total_validation = pq.ParquetFile(path).metadata.num_rows
    assert total_validation == 22363
    sids = np.load(args.analysis_dir / "item_semantic_ids.npy", allow_pickle=False)
    assert sids.shape == (12101, 3)
    dataset = TaggedValidation(path, sids, None if args.all else args.max_samples)
    # Check all selected labels against the existing item mapping and offsets.
    metadata = pq.read_table(path, columns=["target_item_index", "target_sid"]).slice(0, len(dataset)).to_pylist()
    for row in metadata:
        assert np.array_equal(sids[row["target_item_index"]], row["target_sid"])
        assert all(0 <= code < 256 for code in row["target_sid"])
    device = torch.device(args.device)
    amp = args.precision == "fp16" and device.type == "cuda"
    loader = make_loader(dataset, args.batch_size, collate_tagged, False, args.workers, 43, device.type == "cuda")
    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "beam_diagnosis_summary.json"
    sample_path = args.output_dir / "beam_diagnosis_samples.parquet"
    report = {"state": "running", "split": "valid", "samples_per_model": len(dataset),
              "validation_total": total_validation, "scope": "full_validation" if len(dataset) == total_validation else "local_smoke",
              "beam_size": 15, "sid_length": 3, "precision": args.precision,
              "batch_size": args.batch_size, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
              "validation_file_sha256": file_sha256(path), "sid_mapping_sha256": file_sha256(args.analysis_dir / "item_semantic_ids.npy"),
              "models": {}, "paired_vs_original": {},
              "h6_training_loss": [{k: row[k] for k in ("epoch", "loss", "loss_ce", "loss_nce")} for row in h6["epochs"]]}
    write_json(summary_path, report)
    all_records, records_by_model = [], {}
    for name, checkpoint, epoch in models:
        started = time.perf_counter()
        original_hash = file_sha256(checkpoint)
        tiger = Tiger(config)
        tiger.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        tiger.to(device).requires_grad_(False).eval()
        model = ObservedModel(tiger, name).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = evaluate(model, model.batches(loader), device, 15, amp=amp,
                           mixed_precision_type="fp16", desc=f"Diagnose {name} ({len(dataset)} valid)")
        records = model.records
        assert [row["sample_id"] for row in records] == dataset.sample_ids
        assert all(row["target_tokens"] == dataset[i]["labels"] for i, row in enumerate(records))
        result = summarize(records, metrics)
        result.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=original_hash, epoch=epoch,
                      seconds=time.perf_counter() - started, first_batch_passive_replay_equal=model.reference_checked)
        if device.type == "cuda":
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        assert file_sha256(checkpoint) == original_hash
        report["models"][name] = result
        all_records.extend(records)
        records_by_model[name] = records
        if name != "original":
            report["paired_vs_original"][name] = paired(records_by_model["original"], records)
        write_json(summary_path, report)
        print(json.dumps({"model": name, **result}), flush=True)
        del model, tiger
        if device.type == "cuda":
            torch.cuda.empty_cache()
    temporary = sample_path.with_suffix(".parquet.partial")
    table = pa.Table.from_pylist(all_records, schema=sample_schema())
    pq.write_table(table, temporary, compression="zstd")
    assert pq.read_table(temporary).to_pylist() == all_records
    temporary.replace(sample_path)
    report.update(state="complete", sample_rows=len(all_records), samples_sha256=file_sha256(sample_path),
                  no_training=True, no_test_data=True, scores_from_current_models=True)
    write_json(summary_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=BASE / "analysis")
    parser.add_argument("--original-checkpoint", type=Path, default=BASE / "run/tiger/best_model.pt")
    parser.add_argument("--finetune-dir", type=Path, default=ROOT / "out/tiger_prefix_ft/beauty_finetune_20261014_232805")
    parser.add_argument("--ce-report", type=Path, default=REPORTS / "cloud_ce_only_20261014_232805.json")
    parser.add_argument("--h6-report", type=Path, default=REPORTS / "cloud_h6_20261014_232805.json")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-samples", type=int, default=100)
    limits.add_argument("--all", action="store_true", help="Full Validation; intended for the cloud run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "no"), default="fp16")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/tiger_prefix_ft/beauty_beam_diagnosis_smoke100_20261015")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
