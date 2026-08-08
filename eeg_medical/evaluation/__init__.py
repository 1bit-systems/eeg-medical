"""EEG Medical — evaluation package init."""

from eeg_medical.evaluation.benchmarks import (
    artifact_removal_benchmark,
    montage_reconstruction_benchmark,
    run_benchmark,
    seizure_detection_benchmark,
)

__all__ = [
    "artifact_removal_benchmark",
    "montage_reconstruction_benchmark",
    "run_benchmark",
    "seizure_detection_benchmark",
]
