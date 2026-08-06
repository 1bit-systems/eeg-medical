"""EEG Medical — clinical evaluation benchmarks for fine-tuned ZUNA1.1.

Each benchmark evaluates a clinically meaningful task:
1. Seizure detection — classify ictal vs interictal segments
2. Artifact removal — reconstruct clean EEG from artifact-corrupted inputs
3. Montage reconstruction — predict dense montage from sparse clinical montage
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import numpy as np


def load_clinical_metadata(meta_dir: str | Path) -> dict[str, dict]:
    """Load TUH clinical metadata (seizure labels, montage info, etc.).

    Returns dict mapping recording_id -> metadata dict.
    """
    meta_dir = Path(meta_dir)
    metadata = {}
    for f in meta_dir.glob("*.json"):
        rec_id = f.stem.rsplit("_seg", 1)[0]
        if rec_id not in metadata:
            metadata[rec_id] = json.loads(f.read_text())
    return metadata


def seizure_detection_benchmark(
    reconstructed: np.ndarray,
    original: np.ndarray,
    metadata: dict,
) -> dict:
    """Evaluate seizure detection from reconstructed EEG.

    Uses reconstructed vs original NMSE as a proxy for "how well clinical
    features are preserved." Lower NMSE on ictal segments = better preservation
    of seizure morphology.

    Args:
        reconstructed: (n_ch, n_samples) model output
        original: (n_ch, n_samples) ground truth
        metadata: clinical metadata with seizure labels

    Returns:
        dict with per-channel NMSE, overall NMSE, seizure flag
    """
    nmse_per_ch = np.mean((reconstructed - original) ** 2, axis=1) / (
        np.var(original, axis=1) + 1e-8
    )

    is_seizure = metadata.get("has_seizure", False)

    return {
        "nmse_mean": float(np.mean(nmse_per_ch)),
        "nmse_per_channel": nmse_per_ch.tolist(),
        "is_seizure": is_seizure,
        "task": "seizure_detection",
    }


def artifact_removal_benchmark(
    clean: np.ndarray,
    artifact_corrupted: np.ndarray,
    reconstructed: np.ndarray,
) -> dict:
    """Evaluate artifact removal quality.

    How well does the model reconstruct clean signal from artifact-corrupted input?

    Args:
        clean: original clean EEG
        artifact_corrupted: EEG with simulated artifacts (muscle, eye-blink, electrode pop)
        reconstructed: model's denoised output

    Returns:
        dict with improvement metrics
    """
    # NMSE before (artifact vs clean)
    nmse_before = np.mean((artifact_corrupted - clean) ** 2) / (
        np.var(clean) + 1e-8
    )
    # NMSE after (reconstructed vs clean)
    nmse_after = np.mean((reconstructed - clean) ** 2) / (
        np.var(clean) + 1e-8
    )
    # Signal-to-noise ratio improvement
    snr_improvement_db = 10 * np.log10(nmse_before / (nmse_after + 1e-8))

    return {
        "nmse_before": float(nmse_before),
        "nmse_after": float(nmse_after),
        "snr_improvement_db": float(snr_improvement_db),
        "task": "artifact_removal",
    }


def montage_reconstruction_benchmark(
    predicted_dense: np.ndarray,
    true_dense: np.ndarray,
    n_sparse_channels: int,
) -> dict:
    """Evaluate montage upsampling from sparse to dense.

    How well does the model predict 128-channel EEG from N sparse channels?

    Args:
        predicted_dense: model's predicted dense montage (n_dense, n_samples)
        true_dense: ground truth dense montage
        n_sparse_channels: number of input channels

    Returns:
        dict with per-region NMSE
    """
    n_dense = predicted_dense.shape[0]
    nmse_per_ch = np.mean((predicted_dense - true_dense) ** 2, axis=1) / (
        np.var(true_dense, axis=1) + 1e-8
    )

    # Group channels into brain regions (rough 10-20 groupings)
    regions = _default_regions()
    region_nmse = {}
    for region, indices in regions.items():
        if max(indices) < n_dense:
            region_nmse[region] = float(np.mean(nmse_per_ch[indices]))

    return {
        "nmse_mean": float(np.mean(nmse_per_ch)),
        "n_input_channels": n_sparse_channels,
        "n_output_channels": n_dense,
        "region_nmse": region_nmse,
        "task": "montage_reconstruction",
    }


def _default_regions() -> dict[str, list[int]]:
    """Default 10-20 channel groupings by brain region."""
    return {
        "frontal": list(range(0, 32)),      # Fp1, Fp2, F3, F4, Fz, etc.
        "central": list(range(32, 64)),     # C3, C4, Cz
        "temporal": list(range(64, 96)),    # T3, T4, T5, T6
        "parietal": list(range(96, 112)),   # P3, P4, Pz
        "occipital": list(range(112, 128)), # O1, O2
    }


def run_benchmark(
    benchmark_name: str,
    reconstructed: np.ndarray,
    original: np.ndarray,
    metadata: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Run a named clinical benchmark.

    Args:
        benchmark_name: "seizure_detection", "artifact_removal", or "montage_reconstruction"
        reconstructed: model output
        original: ground truth
        metadata: optional clinical metadata
    """
    benchmarks = {
        "seizure_detection": seizure_detection_benchmark,
        "artifact_removal": artifact_removal_benchmark,
        "montage_reconstruction": montage_reconstruction_benchmark,
    }

    fn = benchmarks.get(benchmark_name)
    if fn is None:
        available = list(benchmarks.keys())
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {available}")

    return fn(reconstructed, original, **(metadata or {}), **kwargs)
