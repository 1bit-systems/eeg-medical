"""EEG Medical — basic tests."""

import numpy as np
import pytest

from eeg_medical.evaluation.benchmarks import (
    seizure_detection_benchmark,
    artifact_removal_benchmark,
)


def test_seizure_detection_benchmark():
    """Seizure detection benchmark should return lower NMSE for similar signals."""
    n_ch, n_samp = 19, 2560
    original = np.random.randn(n_ch, n_samp).astype(np.float32) * 20  # µV scale

    # Good reconstruction (low noise added)
    good = original + np.random.randn(n_ch, n_samp).astype(np.float32) * 2
    # Bad reconstruction (high noise added)
    bad = original + np.random.randn(n_ch, n_samp).astype(np.float32) * 20

    good_result = seizure_detection_benchmark(good, original, {"has_seizure": False})
    bad_result = seizure_detection_benchmark(bad, original, {"has_seizure": False})

    assert good_result["nmse_mean"] < bad_result["nmse_mean"]
    assert good_result["task"] == "seizure_detection"


def test_artifact_removal_benchmark():
    """Artifact removal should show SNR improvement."""
    n_ch, n_samp = 19, 2560
    clean = np.random.randn(n_ch, n_samp).astype(np.float32) * 20
    artifact = clean.copy()
    artifact[0] += 200  # large artifact on channel 0

    # Good denoising
    good_denoised = clean + np.random.randn(n_ch, n_samp).astype(np.float32) * 2

    result = artifact_removal_benchmark(clean, artifact, good_denoised)
    assert result["snr_improvement_db"] > 0
    assert result["task"] == "artifact_removal"
