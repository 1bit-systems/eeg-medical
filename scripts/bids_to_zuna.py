#!/usr/bin/env python3
"""Convert OpenNeuro BIDS-format EEG to ZUNA v3 training format.

Walks a BIDS directory, reads .set/.edf files, extracts channel positions,
resamples to 256 Hz, segments into fixed-duration windows with quality scoring,
and writes JSON metadata + float32 memmap pairs.

Output format is compatible with ZUNA's EEGDataset_v3 and our ClinicalEEGDataset.

Usage:
  python scripts/bids_to_zuna.py --input data/hbn --output data/hbn_zuna
  python scripts/bids_to_zuna.py --input data/mmidb --output data/mmidb_zuna --segment 10s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import mne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIDS → ZUNA v3 EEG converter")
    p.add_argument("--input", type=str, required=True,
                   help="Root of BIDS dataset directory")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory for ZUNA-format files")
    p.add_argument("--segment", type=str, default="30s",
                   choices=["5s", "10s", "30s"],
                   help="Segment duration")
    p.add_argument("--min-quality", type=float, default=0.15,
                   help="Minimum per-segment quality score (0-1)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max recordings to process (0=all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-processed recordings")
    return p.parse_args()


SEGMENT_DURATION_MAP = {"5s": 5.0, "10s": 10.0, "30s": 30.0}
TARGET_SFREQ = 256


def find_bids_recordings(root: Path) -> list[dict]:
    """Discover EEG recordings in BIDS structure.

    Returns list of dicts: {subject, task, session, path, format}
    """
    recordings = []

    for pattern in ["*.edf", "*.set"]:
        for f in sorted(root.rglob(pattern)):
            if "/sourcedata/" in str(f):
                continue  # skip raw source data, use the primary derivatives

            parts = f.stem.split("_")
            meta = {"path": f, "format": f.suffix[1:]}

            for p in parts:
                if p.startswith("sub-"):
                    meta["subject"] = p[4:]
                elif p.startswith("task-"):
                    meta["task"] = p[5:]
                elif p.startswith("ses-"):
                    meta["session"] = p[4:]
                elif p.startswith("run-"):
                    meta["run"] = p[4:]

            meta.setdefault("subject", "unknown")
            meta.setdefault("task", "unknown")
            meta.setdefault("session", "01")

            recordings.append(meta)

    return recordings


def read_eeg(recording: dict) -> Optional[tuple[np.ndarray, float, list[str]]]:
    """Read EEG data from BIDS file. Returns (data_array, sfreq, channel_names)."""
    path = recording["path"]
    fmt = recording["format"]

    try:
        if fmt == "edf":
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        elif fmt == "set":
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        else:
            print(f"  Unknown format: {fmt}", file=sys.stderr)
            return None

        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        ch_names = raw.info["ch_names"]
        raw.close()
        return data, sfreq, ch_names

    except Exception as e:
        print(f"  Failed to read {path.name}: {e}", file=sys.stderr)
        return None


def extract_channel_positions(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float = TARGET_SFREQ,
) -> np.ndarray:
    """Extract 3D scalp coordinates for each channel using 10-20 montage.

    Returns: (n_channels, 3) float32 array, or uniform sphere positions.
    """
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(ch_names, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data[:len(ch_names)], info, verbose=False)
        raw.set_montage(montage, match_case=False, on_missing="warn")

        pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float32)

        # Check for unplaced channels (NaN or all-zero)
        valid = ~np.all(pos == 0, axis=1) & ~np.any(np.isnan(pos), axis=1)
        if valid.sum() < 4:
            raise ValueError("Too few valid channel positions")
        return pos

    except Exception:
        # Fallback: Fibonacci sphere
        n_ch = len(ch_names)
        idx = np.arange(n_ch, dtype=np.float64) + 0.5
        phi = np.arccos(1 - 2 * idx / n_ch)
        theta = np.pi * (1 + np.sqrt(5)) * idx
        return np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=1).astype(np.float32)


def quality_score(data: np.ndarray, sfreq: float = TARGET_SFREQ) -> float:
    """Per-segment quality score (0=bad, 1=clean)."""
    n_ch, n_samp = data.shape
    scores = np.ones(n_ch, dtype=np.float32)

    for i in range(n_ch):
        ch = data[i]
        if np.std(ch) < 1e-6:
            scores[i] = 0.0
            continue
        if np.max(np.abs(ch)) > 500:
            scores[i] *= 0.3
        window_var = np.var(ch.reshape(-1, max(1, int(sfreq))), axis=1)
        if len(window_var) > 1 and np.std(window_var) / (np.mean(window_var) + 1e-8) > 3:
            scores[i] *= 0.5

    return float(np.mean(scores))


def segment_data(
    data: np.ndarray,
    sfreq: float,
    segment_dur: float,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Slide window over data, yielding segments of fixed duration."""
    seg_samples = int(segment_dur * TARGET_SFREQ)
    stride = int(seg_samples * (1 - overlap))
    segments = []

    start = 0
    while start + seg_samples <= data.shape[1]:
        segments.append(data[:, start:start + seg_samples])
        start += stride

    return segments


def process_recording(
    recording: dict,
    output_dir: Path,
    segment_dur: float,
    min_quality: float,
) -> int:
    """Convert one BIDS recording to ZUNA format. Returns number of output segments."""
    path = recording["path"]
    subject = recording.get("subject", "unknown")
    task = recording.get("task", "unknown")
    session = recording.get("session", "01")

    result = read_eeg(recording)
    if result is None:
        return 0

    data, sfreq, ch_names = result
    n_ch = data.shape[0]

    # Resample to 256 Hz
    if sfreq != TARGET_SFREQ:
        info = mne.create_info(n_ch, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.resample(TARGET_SFREQ, verbose=False)
        data = raw.get_data()
        raw.close()

    # Extract positions
    positions = extract_channel_positions(data, ch_names)

    # Segment
    segments = segment_data(data, TARGET_SFREQ, segment_dur)

    # Output prefix
    base = f"sub-{subject}_task-{task}_ses-{session}"
    if "run" in recording:
        base += f"_run-{recording['run']}"

    n_saved = 0
    for i, seg in enumerate(segments):
        q = quality_score(seg)
        if q < min_quality:
            continue

        stem = f"{base}_seg{i:04d}"

        # Save memmap
        mmap_path = output_dir / f"{stem}.mmap"
        mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=seg.shape)
        mmap[:] = seg.astype(np.float32)
        mmap.flush()

        # Save JSON metadata
        meta = {
            "subject": subject,
            "task": task,
            "session": session,
            "run": recording.get("run", "01"),
            "source_file": str(path.name),
            "segment_idx": i,
            "duration_sec": segment_dur,
            "n_channels": seg.shape[0],
            "n_samples": seg.shape[1],
            "sfreq": TARGET_SFREQ,
            "ch_names": ch_names,
            "scalp_positions_3d": positions.tolist(),
            "quality_score": q,
        }
        meta_path = output_dir / f"{stem}.json"
        meta_path.write_text(json.dumps(meta))

        n_saved += 1

    return n_saved


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_dir = Path(args.output)
    segment_dur = SEGMENT_DURATION_MAP[args.segment]
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = find_bids_recordings(input_root)
    if not recordings:
        print(f"No EEG recordings found in {input_root}", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        recordings = recordings[:args.limit]

    print(f"Found {len(recordings)} recordings in {input_root}")
    print(f"Output: {output_dir}")
    print(f"Segment duration: {args.segment} ({segment_dur}s @ {TARGET_SFREQ}Hz)")

    total_segments = 0
    processed = 0
    skipped = 0
    t0 = time.time()

    for i, rec in enumerate(recordings):
        # Check if already processed
        if args.resume:
            base = f"sub-{rec.get('subject', 'unknown')}_task-{rec.get('task', 'unknown')}"
            existing = list(output_dir.glob(f"{base}_seg*.json"))
            if existing:
                skipped += 1
                continue

        try:
            n = process_recording(rec, output_dir, segment_dur, args.min_quality)
            total_segments += n
            processed += 1
        except Exception as e:
            print(f"  ERROR {rec['path'].name}: {e}", file=sys.stderr)
            continue

        if (i + 1) % 20 == 0 or i == len(recordings) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1) * 60
            print(f"  [{i+1}/{len(recordings)}] {total_segments} segments "
                  f"({rate:.0f} rec/min) — {processed} done, {skipped} skipped",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total_segments} segments from {processed} recordings "
          f"in {elapsed/60:.0f} min")


if __name__ == "__main__":
    main()
