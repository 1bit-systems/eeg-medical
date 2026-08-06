#!/usr/bin/env python3
"""Evaluate a fine-tuned ZUNA1.1 model on clinical benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --task seizure_detection
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --task all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import torch

from eeg_medical.evaluation import run_benchmark
from eeg_medical.data import read_tuh_edf, resample_to_256hz, segment_eeg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned ZUNA1.1 on clinical tasks")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="all",
                   choices=["all", "seizure_detection", "artifact_removal", "montage_reconstruction"])
    p.add_argument("--data", type=str, default="data/tuh_processed")
    p.add_argument("--output", type=str, default="eval_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    # TODO: load model from checkpoint
    # model = load_zuna_checkpoint(ckpt_path)

    results = {"checkpoint": str(ckpt_path), "benchmarks": {}}
    tasks = (
        ["seizure_detection", "artifact_removal", "montage_reconstruction"]
        if args.task == "all"
        else [args.task]
    )

    for task in tasks:
        print(f"\nRunning benchmark: {task}")
        # TODO: run actual inference + benchmark
        # dummy = model.reconstruct(test_segment)
        # result = run_benchmark(task, dummy, test_segment, metadata)
        # results["benchmarks"][task] = result
        results["benchmarks"][task] = {"status": "scaffold — evaluation pending"}

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
