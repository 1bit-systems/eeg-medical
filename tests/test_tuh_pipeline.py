"""Round-trip test: preprocess_recording output must be readable by ClinicalEEGDataset.

Guards the n_channels/n_samples dim order: segment is (n_channels, n_samples),
and the metadata must say so, or the memmap gets read transposed.
"""

import json

import numpy as np

from eeg_medical.data.tuh import preprocess_recording
from eeg_medical.data.clinical_dataset import ClinicalEEGDataset


def _fake_edf_read(edf_path):
    rng = np.random.default_rng(0)
    return (
        rng.normal(0, 20, (2, 512)).astype(np.float32),
        {
            "ch_names": ["F3", "F4"],
            "sfreq": 256,
            "n_channels": 2,
            "n_samples": 512,
            "subject": "0001",
            "session": "01",
            "task": "test",
        },
        2.0,
    )


def test_preprocess_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("eeg_medical.data.tuh.read_tuh_edf", _fake_edf_read)
    monkeypatch.setattr(
        "eeg_medical.data.tuh.apply_montage",
        lambda data, ch_names: np.zeros((2, 3), dtype=np.float32),
    )

    out = tmp_path / "out"
    metas = preprocess_recording(
        tmp_path / "sub-0001_ses-01_task-test.edf", out, segment_duration=1.0
    )
    assert len(metas) == 2  # 2 s of data / 1 s segments

    meta = json.loads(metas[0].read_text())
    assert meta["n_channels"] == 2
    assert meta["n_samples"] == 256

    # ClinicalEEGDataset must read the memmap with the true (n_ch, n_samples) layout
    ds = ClinicalEEGDataset(out, segment_duration=1.0, shuffle=False)
    seg = next(iter(ds))
    assert seg["eeg"].shape == (2, 256)
    assert seg["chan_pos"].shape == (2, 3)
