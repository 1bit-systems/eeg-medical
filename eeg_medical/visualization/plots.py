"""EEG Medical — visualization utilities for clinical EEG."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    ch_names: Optional[list[str]] = None,
    sfreq: int = 256,
    n_channels: int = 8,
    title: str = "EEG Reconstruction",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot original vs reconstructed EEG for visual comparison.

    Args:
        original: (n_ch, n_samples) ground truth
        reconstructed: (n_ch, n_samples) model output
        ch_names: optional channel names
        sfreq: sampling frequency for time axis
        n_channels: max channels to plot
        title: plot title
        save_path: if provided, save figure to disk
    """
    n_ch = min(original.shape[0], n_channels)
    time = np.arange(original.shape[1]) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i in range(n_ch):
        ax = axes[i]
        ax.plot(time, original[i], "k-", alpha=0.6, linewidth=0.5, label="Original")
        ax.plot(time, reconstructed[i], "r-", alpha=0.6, linewidth=0.5, label="Reconstructed")
        label = ch_names[i] if ch_names and i < len(ch_names) else f"Ch {i+1}"
        ax.set_ylabel(label)
        ax.set_xlim(time[0], time[-1])

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_nmse_heatmap(
    nmse_per_ch: np.ndarray,
    ch_names: Optional[list[str]] = None,
    title: str = "Per-Channel NMSE",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot per-channel NMSE as a topographic heatmap (simplified).

    For proper topomaps, use MNE's plot_topomap with channel positions.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    n_ch = len(nmse_per_ch)
    ax.bar(range(n_ch), nmse_per_ch, color="steelblue", alpha=0.8)
    ax.axhline(y=np.mean(nmse_per_ch), color="red", linestyle="--", label=f"Mean: {np.mean(nmse_per_ch):.4f}")

    if ch_names and len(ch_names) >= n_ch:
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(ch_names[:n_ch], rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("NMSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
