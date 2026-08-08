#!/usr/bin/env python3
"""Preprocess TUH EEG Corpus into ZUNA1.1-compatible format.

Converts EDF files -> 256Hz resampled numpy memmaps + metadata JSON files.
Output format matches ZUNA's EEGDataset_v3 requirements.

Usage:
    python scripts/preprocess_tuh.py --input data/tuh_raw --output data/tuh_processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from eeg_medical.data import preprocess_recording


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess TUH EEG for ZUNA1.1")
    p.add_argument("--input", type=str, required=True, help="Directory with TUH EDF files")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--segment-duration", type=float, default=30.0)
    p.add_argument("--min-quality", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Max recordings to process (0=all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(input_dir.rglob("*.edf"))
    if not edf_files:
        print(f"No EDF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        edf_files = edf_files[: args.limit]

    print(f"Found {len(edf_files)} EDF files")
    print(f"Output: {output_dir}")
    print(f"Segment duration: {args.segment_duration}s")
    print(f"Min quality: {args.min_quality}")

    total_segments = 0
    failed = 0

    # Process sequentially for now (parallel needs GPU-safe memory management)
    for edf_path in tqdm(edf_files, desc="Preprocessing"):
        try:
            outputs = preprocess_recording(
                edf_path,
                output_dir,
                segment_duration=args.segment_duration,
                min_quality=args.min_quality,
            )
            total_segments += len(outputs)
        except Exception as e:
            print(f"  Failed {edf_path.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {total_segments} segments from {len(edf_files) - failed}/{len(edf_files)} recordings")


if __name__ == "__main__":
    main()
