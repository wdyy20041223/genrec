"""Verify and unpack a TIGER analysis backup using hashes and file metadata."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import pyarrow.parquet as pq


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    root = args.backup_dir.resolve()
    archive_path = root / "offline_bundle.tar.gz"
    actual_archive_hash = sha256(archive_path)
    if actual_archive_hash != args.archive_sha256.lower():
        raise RuntimeError("Downloaded archive checksum mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            target = root.joinpath(*name.parts).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Archive entry escapes backup directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if target.exists():
                    raise RuntimeError(f"Refusing to overwrite backup file: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
            else:
                raise RuntimeError(f"Unsupported archive entry type: {member.name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checked = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing or wrong-size file: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {entry['path']}")
        checked.append(entry["path"])
    analysis = root / "analysis"
    summary = json.loads((analysis / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "complete"
    sids = np.load(analysis / "item_semantic_ids.npy", mmap_mode="r")
    assert sids.shape == (12101, 3)
    assert bool(((sids >= 0) & (sids < 256)).all())
    epochs = np.load(analysis / "rqvae_checkpoint_epochs.npy")
    trajectory = np.load(analysis / "rqvae_checkpoint_semantic_ids.npy", mmap_mode="r")
    assert epochs.tolist() == list(range(200, 2001, 200))
    assert trajectory.shape == (10, 12101, 3)
    assert np.array_equal(trajectory[-1], sids)
    assert len(list((root / "run/rqvae").glob("checkpoint_epoch_*.pt"))) == 10
    assert (root / "run/tiger/best_model.pt").is_file()
    assert (root / "source_snapshot.zip").is_file()
    counts = {}
    for name, expected in (("train", 131413), ("valid", 22363), ("test", 22363)):
        count = pq.ParquetFile(analysis / f"{name}_samples.parquet").metadata.num_rows
        assert count == expected
        counts[name + "_samples"] = count
    for name in ("valid", "test"):
        pred = np.load(analysis / f"{name}_prediction_tokens.npy", mmap_mode="r")
        assert pred.shape == (22363, 15, 3)
        count = pq.ParquetFile(analysis / f"{name}_predictions.parquet").metadata.num_rows
        assert count == 22363 * 15
        assert pq.ParquetFile(analysis / f"{name}_user_metrics.parquet").metadata.num_rows == 22363
        assert max(abs(v) for v in summary["predictions"][name]["difference_from_saved_metrics"].values()) <= 2e-7
        counts[name + "_prediction_rows"] = count
    verification = {
        "state": "verified", "verified_at": datetime.now(timezone.utc).isoformat(),
        "backup_dir": str(root), "archive_sha256": actual_archive_hash,
        "files_verified": len(checked),
        "file_bytes_verified": sum(entry["bytes"] for entry in manifest["files"]),
        "row_counts": counts, "item_count": 12101,
        "rqvae_checkpoints": 10, "sid_shape": list(sids.shape),
        "source_commit": manifest["source_commit"],
        "prediction_metrics_match_saved_results": True,
        "shutdown_requested": True,
    }
    (root / "offline_backup_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
