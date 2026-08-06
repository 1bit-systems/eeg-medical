"""Full ZUNA1.1 fine-tuning loop for clinical EEG.

Wires up Zyphra's Lingua training framework with:
- EEGDataset_v3 (or clinical variant) for streaming data
- EEGProcessor for diffusion flow-matching noise/target prep
- EncoderDecoder model forward pass
- Distributed training (FSDP, compile, mixed precision)
- Clinical evaluation callbacks
- Checkpoint save/load via DCP (torch distributed checkpoint)

Usage:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/tuh_clinical_finetune.yaml

or single-GPU:
    python scripts/train.py --config configs/tuh_clinical_finetune.yaml --num-gpus 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# ZUNA imports (available via pip install zuna)
from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import (
    DecoderTransformerArgs,
    EncoderDecoder,
)
from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import (
    EEGDataset_v3,
    EEGProcessor,
)
from zuna.inference.AY2l.lingua.lingua.distributed import (
    setup_torch_distributed,
    parallelize_model,
    DistributedArgs,
)
from zuna.inference.AY2l.lingua.lingua.optim import build_optimizer, OptimArgs
from zuna.inference.AY2l.lingua.lingua.checkpoint import CheckpointManager, CheckpointArgs
from zuna.inference.AY2l.lingua.lingua.logger import MetricLogger, LoggingArgs, Logger

from eeg_medical.training.config import FinetuneConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ZUNA1.1 on clinical EEG")
    p.add_argument("--config", type=str, default="configs/tuh_clinical_finetune.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    return p.parse_args()


def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize torch distributed process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def build_model(config: FinetuneConfig, device: torch.device) -> EncoderDecoder:
    """Build ZUNA1.1 EncoderDecoder from pretrained weights with optional LoRA."""
    # Model args from ZUNA1.1 defaults (380M params, dim=1024, n_layers=16)
    model_args = DecoderTransformerArgs(
        dim=1024,
        n_layers=16,
        head_dim=64,
        input_dim=32,           # 32 time points per token (0.125s @ 256Hz)
        encoder_input_dim=32,
        encoder_output_dim=32,
        encoder_latent_downsample_factor=1,
        encoder_sliding_window=65536,
        sliding_window=65536,
        xattn_sliding_window=65536,
        max_seqlen=256,
        max_chans=512,
        rope_dim=4,             # 4D-RoPE (x, y, z, tc)
        rope_theta=10000.0,
        ape_dim=0,
        num_fine_time_pts=32,
        model_dtype=config.model_dtype,
        stft_global_sigma=0.1,
        kept_token_loss_weight=0.1,
        huber_c=None,           # MSE by default
    )

    model = EncoderDecoder(model_args)
    model.to(device)

    # Load pretrained weights if available
    if config.init_ckpt_path:
        print(f"Loading pretrained weights from {config.init_ckpt_path}")
        ckpt = torch.load(config.init_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Apply LoRA adapters
    if config.use_lora:
        from eeg_medical.models.zuna_wrapper import add_lora_adapters
        model = add_lora_adapters(
            model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

    return model


def build_dataloader(
    config: FinetuneConfig,
    rank: int,
    world_size: int,
    split: str = "train",
) -> EEGDataset_v3:
    """Build EEGDataset_v3 for streaming clinical EEG data.

    For clinical data, we wrap the TUH-processed memmaps using the ZUNA v3 format.
    The dataset handles: quality filtering, channel dropout, z-scoring, packing.
    """
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import BCIDatasetArgs

    data_args = BCIDatasetArgs(
        data_dir=str(Path(config.data_dir).absolute()),
        use_v3=True,
        filter_version=["v2_notch", "v3_bandpass"],
        min_quality_any=config.min_quality_any,
        min_quality_mean=config.min_quality_mean,
        sample_duration_str=config.segment_duration,
        do_avg_ref=True,
        z_score_type="across_channel",
        batch_size=config.batch_size,
        target_packed_seqlen=config.target_packed_seqlen,
        data_norm=10.0,
        data_clip=1.0,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 4,
        token_dropout_prob=0.99,
        dropout_scheme="mix-4-dropouts-train",
        randomly_permute_sequence=True,
        num_bins_discretize_xyz_chan_pos=100,
        chan_pos_xyz_extremes_type="twelves",
        sample_rate=256,
        shuffle=True,
    )

    return EEGDataset_v3(data_args, rank=rank, world_size=world_size)


def train_step(
    model: EncoderDecoder,
    processor: EEGProcessor,
    batch: dict,
    device: torch.device,
    autocast_ctx,
) -> tuple[torch.Tensor, dict]:
    """Single training step with flow-matching objective.

    Flow: batch eeg -> EEGProcessor.process (creates noise+target at timestep t)
                         -> model.forward (encoder + decoder)
                         -> MSE loss on target
    """
    eeg = batch["eeg"].to(device)           # (B, seqlen, ch*ft)
    chan_pos = batch["chan_pos"].to(device)  # (B, seqlen, 3)
    tok_idx = batch.get("tok_idx", None)
    if tok_idx is not None:
        tok_idx = tok_idx.to(device)

    # EEGProcessor handles: timestep sampling, noise mixing, channel masking
    proc = processor.process(
        eeg_signal=eeg,
        chan_pos=chan_pos,
        tok_idx=tok_idx,
    )

    with autocast_ctx:
        # EncoderDecoder forward: predicts flow target (noise - signal)
        logits, losses = model.forward(
            tokens=proc["decoder_input"],
            cross_attended=proc["encoder_input"],
            timeD=proc["t"],
            seq_lens=batch["seq_lens"].to(device),
            cross_seq_lens=batch["seq_lens"].to(device),
            target=proc["target"],
            tok_idx=tok_idx,
            cross_tok_idx=tok_idx,
            do_idx=proc.get("do_idx"),
            pad_mask=proc.get("pad_mask"),
        )

    return losses["decoder_rf_loss"], losses


def train(
    rank: int,
    world_size: int,
    config: FinetuneConfig,
    output_dir: Path,
) -> None:
    """Main training loop (one process per GPU)."""
    if world_size > 1:
        setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # Build model
    model = build_model(config, device)

    # FSDP + compile
    dist_args = DistributedArgs(
        fsdp_type=config.fsdp_type,
        compile=config.compile,
        model_dtype=config.model_dtype,
        tp_size=1,
    )
    if world_size > 1:
        model = parallelize_model(model, dist_args, device)

    # Optimizer
    optim_args = OptimArgs(
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.optimizer_beta1 if hasattr(config, "optimizer_beta1") else 0.9,
        beta2=config.optimizer_beta2 if hasattr(config, "optimizer_beta2") else 0.95,
        clip=config.clip if hasattr(config, "clip") else 1.0,
        scheduler=config.scheduler,
        warmup=config.warmup_steps,
        lr_min_ratio=config.lr_min_ratio,
        use_ema=config.use_ema if hasattr(config, "use_ema") else True,
        ema_decay=config.ema_decay if hasattr(config, "ema_decay") else 0.9999,
    )
    optimizer, scheduler = build_optimizer(model, optim_args)

    # Checkpoint manager
    ckpt_args = CheckpointArgs(
        dump_every=config.checkpoint_every,
        dump_keep=config.keep_checkpoints,
        eval_every=config.eval_every,
        eval_keep=-1,
        init_ckpt_path=config.init_ckpt_path,
        load_optimizer_state=True,
    )
    ckpt_mgr = CheckpointManager(
        path=output_dir,
        args=ckpt_args,
        model=model,
        optimizer=optimizer,
    )

    # Data
    train_ds = build_dataloader(config, rank, world_size, split="train")
    processor = EEGProcessor(
        stft_global_sigma=0.1,
        data_norm=10.0,
        data_clip=1.0,
        diffusion_noise_schedule="linear",
        masked_in_decoder=False,
    )

    # Mixed precision
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if config.model_dtype == "bf16" else nullcontext()

    # Logger
    if is_main:
        logger = MetricLogger(project=config.wandb_project)
    else:
        logger = None

    # Training loop
    step = ckpt_mgr.train_state.get("step", 0)
    total_steps = config.total_steps
    grad_acc = config.grad_acc_steps

    model.train()
    train_iter = iter(train_ds)
    t_start = time.time()
    accumulated_loss = 0.0

    while step < total_steps:
        optimizer.zero_grad()

        for micro_step in range(grad_acc):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter)

            loss, loss_dict = train_step(model, processor, batch, device, autocast_ctx)
            (loss / grad_acc).backward()
            accumulated_loss += loss.item()

        # Gradient clipping
        if hasattr(optim_args, "clip") and optim_args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_args.clip)

        optimizer.step()
        scheduler.step()
        step += 1

        # Logging
        if is_main and step % config.log_every == 0:
            elapsed = time.time() - t_start
            tok_per_sec = (config.target_packed_seqlen * grad_acc * config.log_every) / max(elapsed, 1e-3)
            avg_loss = accumulated_loss / (config.log_every * grad_acc)

            print(
                f"[step {step:06d}/{total_steps}] "
                f"loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"{tok_per_sec:.0f} tok/s  "
                f"({elapsed:.0f}s)"
            )
            if logger:
                logger.log({
                    "train/loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/tok_per_sec": tok_per_sec,
                    "train/step": step,
                })
            accumulated_loss = 0.0
            t_start = time.time()

        # Checkpoint
        if step % config.checkpoint_every == 0:
            ckpt_mgr.save(step, train_state={"step": step, "config": config.__dict__})
            if is_main:
                print(f"  Checkpoint saved at step {step}")

        # Clinical evaluation
        if step % config.eval_every == 0 and is_main:
            model.eval()
            # TODO: run clinical benchmarks (seizure detection, artifact removal)
            # See eeg_medical/evaluation/benchmarks.py
            model.train()

    # Final save
    if is_main:
        ckpt_mgr.save(step, train_state={"step": step, "config": config.__dict__})
        print(f"Training complete. Final checkpoint at step {step}")

    if world_size > 1:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config.to_yaml(output_dir / "resolved_config.yaml")

    world_size = args.num_gpus

    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            args=(world_size, config, output_dir),
            nprocs=world_size,
            join=True,
        )
    else:
        train(rank=0, world_size=1, config=config, output_dir=output_dir)


if __name__ == "__main__":
    main()
