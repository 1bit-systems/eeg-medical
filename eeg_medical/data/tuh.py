"""EEG Medical — TUH EEG data pipeline for ZUNA1.1 fine-tuning."""

from pathlib import Path
from typing import Iterator, Optional
import json
import numpy as np
import mne


def read_tuh_edf(edf_path: str | Path) -> tuple[np.ndarray, dict, float]:
    """Read a TUH EDF file, return (data, metadata, duration_seconds).

    data shape: (n_channels, n_samples)
    metadata: channel names, montage info, clinical labels
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = raw.info["ch_names"]

    # Extract clinical metadata from TUH filename convention
    # Format: sub-XXX_ses-XXX_task-XXX_run-XXX.edf
    stem = Path(edf_path).stem
    parts = dict(p.split("-") for p in stem.split("_") if "-" in p)

    metadata = {
        "ch_names": ch_names,
        "sfreq": sfreq,
        "n_channels": len(ch_names),
        "n_samples": data.shape[1],
        "subject": parts.get("sub"),
        "session": parts.get("ses"),
        "task": parts.get("task"),
    }

    return data, metadata, data.shape[1] / sfreq


def resample_to_256hz(data: np.ndarray, orig_sfreq: float) -> np.ndarray:
    """Resample EEG data to 256 Hz (ZUNA1.1 native sample rate)."""
    if orig_sfreq == 256:
        return data
    info = mne.create_info(data.shape[0], orig_sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.resample(256, verbose=False)
    return raw.get_data()


def apply_montage(data: np.ndarray, ch_names: list[str]) -> Optional[np.ndarray]:
    """Attempt to fit standard 10-20 montage. Returns 3D scalp coords (n_ch, 3)."""
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(ch_names, 256, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(montage, match_case=False, on_missing="warn")

        pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
        return pos
    except Exception:
        return None


def segment_eeg(
    data: np.ndarray,
    segment_duration: float = 30.0,
    sfreq: int = 256,
    overlap: float = 0.0,
) -> Iterator[np.ndarray]:
    """Slide a window over EEG data, yielding segments of fixed duration.

    segment_duration: seconds per segment (default 30s, matches ZUNA1.1 training)
    """
    segment_samples = int(segment_duration * sfreq)
    stride = int(segment_samples * (1 - overlap))
    n_samples = data.shape[1]

    start = 0
    while start + segment_samples <= n_samples:
        yield data[:, start : start + segment_samples]
        start += stride


def quality_score(data: np.ndarray, sfreq: int = 256) -> float:
    """Per-channel quality score (0=bad, 1=clean).
    Penalizes: flat-lining, extreme amplitudes, line noise.
    """
    n_ch, n_samp = data.shape
    scores = np.ones(n_ch)

    for i in range(n_ch):
        ch = data[i]
        # Flat-line check
        if np.std(ch) < 1e-6:
            scores[i] = 0.0
            continue
        # Extreme amplitude check (>500 µV is likely artifact)
        if np.max(np.abs(ch)) > 500:
            scores[i] *= 0.3
        # Variance of variance (nonstationarity)
        window_var = np.var(ch.reshape(-1, sfreq), axis=1)
        if np.std(window_var) / (np.mean(window_var) + 1e-8) > 3:
            scores[i] *= 0.5

    return float(np.mean(scores))


def preprocess_recording(
    edf_path: Path,
    output_dir: Path,
    segment_duration: float = 30.0,
    min_quality: float = 0.2,
) -> list[Path]:
    """Full preprocessing pipeline for one TUH recording.

    Returns list of output file paths.
    """
    data, meta, duration = read_tuh_edf(edf_path)
    data = resample_to_256hz(data, meta["sfreq"])
    pos = apply_montage(data, meta["ch_names"])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = edf_path.stem
    outputs = []

    for i, segment in enumerate(segment_eeg(data, segment_duration)):
        q = quality_score(segment)
        if q < min_quality:
            continue

        # Save as float32 numpy memmap
        mmap_path = output_dir / f"{stem}_seg{i:04d}.mmap"
        mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=segment.shape)
        mmap[:] = segment.astype(np.float32)
        mmap.flush()

        # Save metadata JSON
        meta_path = output_dir / f"{stem}_seg{i:04d}.json"
        meta_dict = {
            **meta,
            "segment_idx": i,
            "duration": segment_duration,
            "n_channels": segment.shape[1],
            "n_samples": segment.shape[0],
            "quality_score": q,
            "scalp_positions_3d": pos.tolist() if pos is not None else None,
        }
        meta_path.write_text(json.dumps(meta_dict))

        outputs.append(meta_path)

    return outputs
