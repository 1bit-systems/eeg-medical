"""Load ZUNA1.1 pretrained weights into EncoderDecoder model.

The HuggingFace checkpoint uses keys prefixed with 'model.' —
we strip that prefix to match the local EncoderDecoder state_dict.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn


def load_zuna11_weights(
    model: nn.Module,
    repo_id: str = "Zyphra/ZUNA1.1",
    filename: str = "model-00001-of-00001.safetensors",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Download and load ZUNA1.1 weights into an EncoderDecoder model.

    Args:
        model: EncoderDecoder instance (already constructed with correct args)
        repo_id: HuggingFace repo
        filename: safetensors file
        device: target device

    Returns:
        model with loaded weights
    """
    path = hf_hub_download(repo_id, filename)
    ckpt = load_file(path)

    # Strip 'model.' prefix from checkpoint keys
    state_dict = {k.removeprefix("model."): v for k, v in ckpt.items()}

    # Load with strict=False — the model may have extra keys (e.g. EMA, dropout_vec)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:5]:
            print(f"    {k}")

    model.to(device)
    return model


def build_zuna11(
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build ZUNA1.1 EncoderDecoder and load pretrained weights.

    One-liner to get a ready-to-use/fine-tune model.
    """
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import (
        DecoderTransformerArgs,
        EncoderDecoder,
    )

    args = DecoderTransformerArgs(
        dim=1024,
        n_layers=16,
        head_dim=64,
        input_dim=32,
        encoder_input_dim=32,
        encoder_output_dim=32,
        encoder_latent_downsample_factor=1,
        encoder_sliding_window=65536,
        sliding_window=65536,
        xattn_sliding_window=65536,
        max_seqlen=256,
        max_chans=512,
        rope_dim=4,
        rope_theta=10000.0,
        ape_dim=0,
        tok_idx_type="{x,y,z,tc}",  # ZUNA1.1 uses 4D-RoPE (no explicit ch dim)
        num_fine_time_pts=32,
        model_dtype="bf16",
        stft_global_sigma=0.1,
        kept_token_loss_weight=0.1,
    )

    model = EncoderDecoder(args)
    load_zuna11_weights(model, device=device)
    return model
