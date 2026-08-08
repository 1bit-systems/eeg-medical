"""EEG Medical — data package init."""

from eeg_medical.data.clinical_dataset import (
    ClinicalEEGDataset,
    prepare_clinical_data,
)
from eeg_medical.data.tuh import (
    preprocess_recording,
    quality_score,
    read_tuh_edf,
    resample_to_256hz,
    segment_eeg,
)

__all__ = [
    "ClinicalEEGDataset",
    "prepare_clinical_data",
    "preprocess_recording",
    "quality_score",
    "read_tuh_edf",
    "resample_to_256hz",
    "segment_eeg",
]
