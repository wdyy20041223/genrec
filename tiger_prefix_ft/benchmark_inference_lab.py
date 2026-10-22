"""Profile retained label-free inference APIs with the original Validation evaluator."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from cotiger.core.models.tiger import Tiger
from cotiger.core.trainers.tiger_trainer import evaluate
from .beam_diagnosis import BASE, TaggedValidation, collate_tagged
from .finetune import file_sha256, make_loader
from .inference_lab import CHECKPOINT_HASH, generate_with_prior, new_json
from .inference_priors import PrefixPrior


class Profiled(torch.nn.Module):
    def __init__(self, tiger, widths, prior=None):
        super().__init__()
        self.tiger, self.widths, self.prior = tiger, widths, prior
        self.predictions = []

    def generate(self, input_ids, attention_mask, num_beams):
        assert num_beams == self.widths[-1]
        if self.prior is None:
            result = self.tiger.generate(input_ids, attention_mask, num_beams)
        else:
            result = generate_with_prior(self.tiger, input_ids, attention_mask, self.prior, self.widths)
        self.predictions.append(result.view(len(input_ids), num_beams, 4)[:, :, 1:].cpu().numpy().astype(np.int16))
        return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = BASE / "run/tiger/best_model.pt"
    before = file_sha256(checkpoint)
    assert before == CHECKPOINT_HASH
    tiger = Tiger(json.loads((BASE / "analysis/model_config.json").read_text())).eval().requires_grad_(False).cuda()
    tiger.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
    sids = np.load(BASE / "analysis/item_semantic_ids.npy", allow_pickle=False)
    dataset = TaggedValidation(BASE / "analysis/valid_samples.parquet", sids, None)
    assert len(dataset) == 22363
    loader = make_loader(dataset, 64, collate_tagged, False, 0, 42, True)
    with np.load(args.experiment / "train_statistics.npz", allow_pickle=False) as archive:
        statistics = {k: archive[k] for k in archive.files}
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for name, widths, calibrated, cache_name in (
        ("original15", [15, 15, 15], False, "raw15_pool256"),
        ("original30", [30, 30, 30], False, "raw30_pool256"),
        ("prior15", [15, 15, 15], True, "prior15_h3_d2_04_d3_04"),
        ("prior30", [30, 30, 30], True, "prior_h3_d2_04_d3_04"),
    ):
        prior = PrefixPrior(statistics, sids + np.array([1, 257, 513]), [0., .4, .4], True, 3) if calibrated else None
        model = Profiled(tiger, widths, prior).eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = evaluate(model, loader, torch.device("cuda"), widths[-1], True, "fp16", f"production {name}")
        torch.cuda.synchronize()
        seconds, peak = time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**20
        actual = np.concatenate(model.predictions)
        cache_path = args.experiment / "valid" / cache_name
        with np.load(cache_path / "candidates.npz", allow_pickle=False) as archive:
            expected = archive["tokens"][:, :widths[-1]]
        assert np.array_equal(actual, expected), f"Non-recording API changed {name} predictions"
        previous = json.loads((cache_path / "summary.json").read_text())
        for key, value in metrics.items():
            assert abs(value - previous["metrics"][key]) < 1e-7, key
        np.savez_compressed(args.output / f"{name}_predictions.npz", tokens=actual)
        item = {"name": name, "samples": len(dataset), "widths": widths, "prior_strengths": [0., .4, .4] if calibrated else None,
                "prior_history": 3 if calibrated else None, "batch_size": 64, "metrics": metrics,
                "seconds": seconds, "peak_gpu_mib": peak, "all_sample_predictions_match_recorded_run": True,
                "original_evaluator": True, "full_vocabulary_trace_disabled": True,
                "final_token_output_collection_enabled_equally": True}
        new_json(args.output / f"{name}.json", item)
        summaries.append(item)
        print(json.dumps(item), flush=True)
        del model
    assert file_sha256(checkpoint) == before
    new_json(args.output / "summary.json", {"state": "passed", "split": "valid", "profiles": summaries,
             "checkpoint_unchanged": True, "all_parameters_frozen": all(not p.requires_grad for p in tiger.parameters()),
             "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
             "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
             "timing_scope": "warm loaded model, original evaluator, same final-token capture; excludes model/stats loading and file compression"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
