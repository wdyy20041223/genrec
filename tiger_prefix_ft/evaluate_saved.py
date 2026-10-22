"""Evaluate the actual epoch 3/5/10 weights after all four runs complete."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import evaluate
from .finetune import EvaluationDataset, collate_evaluation, file_sha256, make_loader, write_json


def run(args):
    directories = {"h2": "h2_mixed", "h4": "h4_mixed", "h6": "h6_mixed", "ce_only": "ce_only"}
    reports = {name: json.loads((args.run_dir / directory / "result.json").read_text())
               for name, directory in directories.items()}
    base = reports["h4"]
    for report in reports.values():
        assert report["state"] == "complete" and report["epoch_budgets"] == [3, 5, 10]
        assert report["model_config"] == base["model_config"]
        assert report["source_checkpoint_sha256"] == base["source_checkpoint_sha256"]
        assert report["beam_size"] == base["beam_size"]
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda")
    item_sids = np.load(args.analysis_dir / "item_semantic_ids.npy", allow_pickle=False)
    test = EvaluationDataset(args.analysis_dir / "test_samples.parquet", item_sids, max_history_items=50)
    assert len(test) == base["test_samples"] == 22363
    loader = make_loader(test, 256, collate_evaluation, False, 4, base["seed"] + 2, True)
    model = Tiger(base["model_config"]).to(device).requires_grad_(False).eval()
    result = {"state": "running", "test_samples": len(test), "beam_size": base["beam_size"],
              "precision": base["precision"], "baseline": base["baseline"], "groups": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    for name, directory in directories.items():
        result["groups"][name] = {}
        for epoch in (3, 5, 10):
            checkpoint = args.run_dir / directory / f"checkpoint_epoch_{epoch}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            metrics = evaluate(model, loader, device, base["beam_size"],
                               amp=base["precision"] == "fp16", mixed_precision_type="fp16",
                               desc=f"Test {name} actual epoch {epoch}")
            valid = next(row["valid"] for row in reports[name]["epochs"] if row["epoch"] == epoch)
            result["groups"][name][str(epoch)] = {
                "epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                "valid": valid, "test": metrics,
            }
            write_json(args.output, result)
            print(json.dumps({"profile": name, "actual_epoch": epoch, "test": metrics}), flush=True)
    result["state"] = "complete"
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
