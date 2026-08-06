"""EEG Medical — evaluation package init."""

from eeg_medical.evaluation.benchmarks import (
    run_benchmark,
    seizure_detection_benchmark,
    artifact_removal_benchmark,
    montage_reconstruction_benchmark,
)

__all__ = [
    "run_benchmark",
    "seizure_detection_benchmark",
    "artifact_removal_benchmark",
    "montage_reconstruction_benchmark",
]
