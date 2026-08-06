"""EEG Medical — fine-tuning loop for ZUNA1.1 on clinical EEG."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning ZUNA1.1 on clinical EEG."""

    # Model
    pretrained_model: str = "Zyphra/ZUNA1.1"
    init_ckpt_path: Optional[str] = None  # local checkpoint override

    # Data
    data_dir: str = "data/tuh_processed"
    segment_duration: str = "30_seconds"
    min_quality_any: float = 0.1
    min_quality_mean: float = 0.2

    # Clinical tasks
    clinical_tasks: list[str] = field(default_factory=lambda: ["denoising", "reconstruction"])
    # seizure_detection, artifact_removal, montage_upsample

    # Training
    batch_size: int = 1
    target_packed_seqlen: int = 22000
    learning_rate: float = 1e-4  # lower than ZUNA1.1's 5e-4 since we're fine-tuning
    warmup_steps: int = 500
    total_steps: int = 50000
    grad_acc_steps: int = 4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    lr_min_ratio: float = 0.01

    # Distributed
    fsdp_type: str = "no_shard"
    compile: bool = True
    model_dtype: str = "bf16"

    # Checkpointing
    checkpoint_every: int = 2000
    eval_every: int = 5000
    keep_checkpoints: int = 5

    # Logging
    wandb_project: str = "eeg-medical"
    log_every: int = 10

    # LoRA (optional parameter-efficient fine-tuning)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class ClinicalEvalConfig:
    """Configuration for clinical evaluation benchmarks."""

    tasks: list[str] = field(default_factory=lambda: [
        "seizure_detection",
        "artifact_removal",
        "montage_reconstruction",
    ])
    data_dir: str = "data/tuh_processed"
    eval_split: str = "test"  # train/val/test or a specific subject list
    num_batches: int = 100
    batch_size: int = 1
    target_packed_seqlen: int = 15000
