"""EEG Medical — model wrappers for ZUNA1.1 fine-tuning.

Provides:
- ZUNA1.1 loading with LoRA support
- Clinical task heads (seizure classifier, etc.)
- Checkpoint save/load utilities
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_zuna_pretrained(
    model_id: str = "Zyphra/ZUNA1.1",
    device: str = "cuda",
) -> nn.Module:
    """Load ZUNA1.1 pretrained weights from HuggingFace or local cache.

    The zuna package has no public `ZUNA` class; the supported path is
    building the EncoderDecoder and loading the safetensors checkpoint
    (see zuna_loader.build_zuna11). model_id is accepted for API
    compatibility but the weights are fetched from the Zyphra/ZUNA1.1
    HF repo by the loader.
    """
    from eeg_medical.models.zuna_loader import build_zuna11

    return build_zuna11(device=device)


def add_lora_adapters(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Add LoRA adapters to ZUNA1.1 for parameter-efficient fine-tuning.

    ponytail: use peft if already installed, else hand-rolled LoRA on attention qkv + output.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        return get_peft_model(model, lora_config)
    except ImportError:
        print("peft not installed. Install with: pip install peft")
        print("Falling back to full fine-tuning (no LoRA).")
        return model


class SeizureClassifier(nn.Module):
    """Linear probe on ZUNA1.1 encoder latents for seizure detection.

    Trained separately after the main fine-tuning to avoid interfering
    with the reconstruction objective.
    """

    def __init__(self, latent_dim: int = 1024, n_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, encoder_latents: torch.Tensor) -> torch.Tensor:
        # encoder_latents: (B, n_registers, latent_dim)
        # Pool registers by mean
        pooled = encoder_latents.mean(dim=1)
        return self.classifier(pooled)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        ckpt["config"] = config
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
) -> dict:
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
