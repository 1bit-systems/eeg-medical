#!/usr/bin/env python3
"""Evaluate a fine-tuned ZUNA1.1 model on clinical benchmarks.

The benchmark functions themselves live in eeg_medical/evaluation/benchmarks.py
(see tests/test_benchmarks.py for usage). Wiring them to a live checkpoint
requires the reconstruction inference path, which is not implemented yet —
this script fails loudly instead of writing placeholder results.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --task seizure_detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    print(
        "Evaluation is not wired yet: there is no inference path that turns a "
        "checkpoint into reconstructions. Use the benchmark functions directly "
        "from eeg_medical.evaluation (see tests/test_benchmarks.py).",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
