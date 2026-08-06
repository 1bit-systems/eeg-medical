#!/usr/bin/env python3
"""Fine-tune ZUNA1.1 on TUH clinical EEG data.

Usage:
    python scripts/finetune.py --config configs/tuh_clinical_finetune.yaml

This script wraps ZUNA1.1's Lingua training framework, swapping in clinical
EEG data and adding medical evaluation benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add zuna's training framework to path
# (requires zuna installed: pip install zuna)
try:
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import (
        DecoderTransformer,
        EncoderTransformer,
        DecoderTransformerArgs,
    )
except ImportError:
    print(
        "ZUNA training framework not found. Install with: pip install zuna",
        file=sys.stderr,
    )
    sys.exit(1)

from eeg_medical.training.config import FinetuneConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ZUNA1.1 on clinical EEG")
    p.add_argument("--config", type=str, default="configs/tuh_clinical_finetune.yaml")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--dry-run", action="store_true", help="Validate config and data without training")
    p.add_argument("--num-gpus", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)

    print(f"Config loaded: {args.config}")
    print(f"  Model: {config.pretrained_model}")
    print(f"  Data: {config.data_dir}")
    print(f"  LR: {config.learning_rate}, Steps: {config.total_steps}")
    print(f"  LoRA: {'enabled' if config.use_lora else 'disabled'}")

    if config.use_lora:
        print(f"    rank={config.lora_rank}, alpha={config.lora_alpha}")

    if args.dry_run:
        print("\n[Dry run] Validating data directory...")
        data_path = Path(config.data_dir)
        if data_path.exists():
            json_files = list(data_path.glob("*.json"))
            print(f"  Found {len(json_files)} processed segments in {config.data_dir}")
        else:
            print(f"  WARNING: {config.data_dir} does not exist yet.")
            print("  Run preprocessing first: python scripts/preprocess_tuh.py")
        return

    # TODO: Full training loop integrating ZUNA's Lingua trainer
    # The training loop uses ZUNA's existing config_bci.yaml structure
    # but with our clinical data paths and evaluation benchmarks.
    #
    # Key integration points:
    # 1. Replace DataConfig.data_dir with clinical data path
    # 2. Add clinical evaluation callbacks (seizure detection, artifact removal)
    # 3. Optional LoRA adapters for parameter-efficient fine-tuning
    # 4. WandB logging with clinical metrics
    print("\nTraining not yet wired — scaffolding in progress.")
    print("Check eeg_medical/training/ for the training loop skeleton.")


if __name__ == "__main__":
    main()
