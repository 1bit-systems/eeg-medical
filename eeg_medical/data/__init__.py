"""EEG Medical — data package init."""

from eeg_medical.data.tuh import (
    preprocess_recording,
    read_tuh_edf,
    resample_to_256hz,
    segment_eeg,
    quality_score,
)
from eeg_medical.data.clinical_dataset import (
    ClinicalEEGDataset,
    prepare_clinical_data,
)

__all__ = [
    "preprocess_recording",
    "read_tuh_edf",
    "resample_to_256hz",
    "segment_eeg",
    "quality_score",
    "ClinicalEEGDataset",
    "prepare_clinical_data",
]
