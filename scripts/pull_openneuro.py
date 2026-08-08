#!/usr/bin/env python3
"""Pull EEG datasets from OpenNeuro for ZUNA1.1 pretraining.

Targets (all public, no DUA required):
  ds005505  HBN RestingState     136 subj × 129ch × 500Hz  ~103 GB   (S3)
  ds005385  Resting-state EEG    608 subj ×  64ch × ?Hz     ~74 GB   (S3)
  ds004362  PhysioNet MMIDB      109 subj ×  64ch × 160Hz   ~11 GB   (S3)
  ds003775  SRM Resting-state      ? subj ×   ?ch × ?Hz        ? GB   (S3)

Usage:
  python scripts/pull_openneuro.py --dataset ds005505 --subjects 10 --output data/hbn
  python scripts/pull_openneuro.py --dataset ds004362 --output data/mmidb
  python scripts/pull_openneuro.py --dataset ds005385 --subjects 50 --output data/resting

Without --subjects, pulls ALL available subjects for that dataset.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

# ── dataset registry ──────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "ds005505": {
        "name": "HBN RestingState EEG (cmi_bids_R1)",
        "bucket": "https://fcp-indi.s3.amazonaws.com",
        "prefix": "data/Projects/HBN/BIDS_EEG/cmi_bids_R1/",
        "task_filter": "task-RestingState",
        "extensions": (".set", ".json", ".tsv"),
        "n_subjects_available": 136,
    },
    "ds004362": {
        "name": "PhysioNet EEG Motor Movement/Imagery",
        "bucket": "https://openneuro.s3.amazonaws.com",
        "prefix": "ds004362/",
        "task_filter": None,  # pull all tasks
        "extensions": (".edf", ".json", ".tsv"),
        "n_subjects_available": 109,
    },
    "ds005385": {
        "name": "Resting-state EEG (608 subjects)",
        "bucket": "https://openneuro.s3.amazonaws.com",
        "prefix": "ds005385/",
        "task_filter": "task-rest",  # resting-state only
        "extensions": (".edf", ".set", ".json", ".tsv"),
        "n_subjects_available": 608,
    },
    "ds003775": {
        "name": "SRM Resting-state EEG",
        "bucket": "https://openneuro.s3.amazonaws.com",
        "prefix": "ds003775/",
        "task_filter": "task-rest",
        "extensions": (".edf", ".set", ".json", ".tsv"),
        "n_subjects_available": None,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull OpenNeuro EEG datasets")
    p.add_argument("--dataset", type=str, required=True,
                   choices=list(DATASETS.keys()))
    p.add_argument("--subjects", type=int, default=0,
                   help="Number of subjects to pull (0=all)")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory root")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download threads")
    p.add_argument("--dry-run", action="store_true",
                   help="List files without downloading")
    return p.parse_args()


def list_s3_prefix(bucket: str, prefix: str) -> list[str]:
    """List all keys under an S3 prefix via HTTP (public bucket, no auth)."""
    keys = []
    marker = ""
    while True:
        url = f"{bucket}/?prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        # Parse XML-like listing
        text = resp.text
        for line in text.split("<Key>")[1:]:
            key = line.split("</Key>")[0]
            keys.append(key)

        # Check for more
        if "<IsTruncated>true</IsTruncated>" in text:
            for line in text.split("<NextMarker>")[1:]:
                marker = line.split("</NextMarker>")[0]
                break
        else:
            break

    return keys


def discover_files(
    dataset: str,
    subjects: int = 0,
) -> list[tuple[str, str]]:
    """Discover files to download. Returns list of (s3_key, local_rel_path)."""
    ds = DATASETS[dataset]
    bucket = ds["bucket"]
    prefix = ds["prefix"]
    task_filter = ds.get("task_filter")
    exts = ds["extensions"]

    print(f"Listing {ds['name']}...")
    all_keys = list_s3_prefix(bucket, prefix)

    # Find subject directories
    subject_dir_names = sorted({
        k.split("/")[1] if dataset != "ds005505" else k.split("/")[5]
        for k in all_keys
        if "/eeg/" in k or k.endswith((".edf", ".set"))
    })

    if subjects > 0:
        subject_dir_names = subject_dir_names[:subjects]

    print(f"  {len(subject_dir_names)} subjects selected (of {len(subject_dir_names)} available)")

    # Filter to matching files
    matches = []
    for sub in subject_dir_names:
        for k in all_keys:
            if sub not in k:
                continue
            if "/eeg/" not in k and not any(k.endswith(ext) for ext in (".edf", ".set")):
                continue

            filename = k.split("/")[-1]
            if task_filter and task_filter not in filename:
                continue
            if not any(k.endswith(ext) for ext in exts):
                continue

            # Compute relative path
            rel = k[len(prefix):]
            matches.append((k, rel))

    return matches


def download_file(
    bucket: str,
    s3_key: str,
    dest: Path,
    skip_existing: bool = True,
) -> tuple[str, int, str]:
    """Download one file. Returns (filename, size_bytes, status)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dest.exists() and dest.stat().st_size > 100:
        return (s3_key, dest.stat().st_size, "skipped")

    try:
        url = f"{bucket}/{s3_key}"
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()

        with open(dest, "wb") as f:
            f.writelines(resp.iter_content(chunk_size=8 * 1024 * 1024))

        return (s3_key, dest.stat().st_size, "ok")
    except Exception as e:
        return (s3_key, 0, f"error: {e}")


def main() -> None:
    args = parse_args()
    ds = DATASETS[args.dataset]
    output_root = Path(args.output)

    # Discover files
    files = discover_files(args.dataset, subjects=args.subjects)
    total_gb = 0

    if args.dry_run:
        print(f"\nWould download {len(files)} files:")
        for s3_key, rel in files[:20]:
            print(f"  {rel}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return

    print(f"Downloading {len(files)} files to {output_root}...")

    # Sequential download (S3 doesn't love too many parallel connections)
    ok = skip = err = 0
    t0 = time.time()
    bucket = ds["bucket"]

    for i, (s3_key, rel) in enumerate(files):
        dest = output_root / rel
        _, size, status = download_file(bucket, s3_key, dest)
        total_gb += size / 1e9

        if status == "ok":
            ok += 1
        elif status == "skipped":
            skip += 1
        else:
            err += 1
            print(f"  [{i+1}/{len(files)}] {rel} — {status}", file=sys.stderr)

        if (i + 1) % 50 == 0 or i == len(files) - 1:
            elapsed = time.time() - t0
            rate = total_gb / max(elapsed, 1) * 3600
            print(f"  [{i+1}/{len(files)}] {total_gb:.1f} GB "
                  f"({rate:.0f} GB/h) — {ok} ok, {skip} skipped, {err} err",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total_gb:.1f} GB in {elapsed/60:.0f} min "
          f"({ok} ok, {skip} skipped, {err} errors)")


if __name__ == "__main__":
    main()
