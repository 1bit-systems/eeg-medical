"""Clinical EEG dataset adapter for ZUNA1.1 training.

Wraps preprocessed TUH data (JSON + memmap pairs) into the format
expected by ZUNA's EEGDataset_v3 / EEGProcessor pipeline.

Format per segment:
  JSON: metadata (ch_names, sfreq, n_channels, scalp_positions_3d, quality_score)
  .mmap: float32 data of shape (n_channels, n_samples)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ClinicalEEGDataset(IterableDataset):
    """Streaming dataset over preprocessed clinical EEG segments.

    Compatible with ZUNA's training pipeline: yields dicts with
    keys matching what EEGProcessor.process() and EncoderDecoder.forward() expect.
    """

    def __init__(
        self,
        data_dir: str | Path,
        segment_duration: float = 30.0,
        sfreq: int = 256,
        min_quality: float = 0.2,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.segment_duration = segment_duration
        self.sfreq = sfreq
        self.max_seqlen = int(segment_duration * sfreq / 32)  # tokens at 32-sample resolution
        self.min_quality = min_quality
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        # Discover segments
        self.meta_files = sorted(self.data_dir.glob("*.json"))
        if not self.meta_files:
            raise FileNotFoundError(f"No JSON metadata files found in {data_dir}")

        # Filter by quality
        self.meta_files = self._filter_by_quality(self.meta_files)

        # Shard across workers
        self.meta_files = self.meta_files[rank::world_size]

    def _filter_by_quality(self, files: list[Path]) -> list[Path]:
        """Filter segments by minimum quality score."""
        kept = []
        for f in files:
            try:
                meta = json.loads(f.read_text())
                if meta.get("quality_score", 1.0) >= self.min_quality:
                    kept.append(f)
            except (json.JSONDecodeError, KeyError):
                continue
        return kept

    def _load_segment(self, meta_path: Path) -> dict | None:
        """Load one segment: read JSON metadata + memmap data.

        Returns dict with keys matching ZUNA's expected format, or None if invalid.
        """
        try:
            meta = json.loads(meta_path.read_text())
            mmap_path = meta_path.with_suffix(".mmap")
            if not mmap_path.exists():
                # Try .npy or .dat
                for ext in [".npy", ".dat"]:
                    alt = meta_path.with_suffix(ext)
                    if alt.exists():
                        mmap_path = alt
                        break
                else:
                    return None

            # Load data
            if mmap_path.suffix == ".mmap":
                data = np.memmap(
                    mmap_path, dtype=np.float32, mode="r",
                    shape=(meta["n_channels"], meta["n_samples"]),
                )
            elif mmap_path.suffix == ".npy":
                data = np.load(mmap_path)
            else:
                data = np.memmap(
                    mmap_path, dtype=np.float32, mode="r",
                    shape=(meta["n_channels"], meta["n_samples"]),
                )

            # Get scalp positions (3D coords per channel)
            positions = meta.get("scalp_positions_3d")
            if positions is None:
                # Generate uniform positions on a sphere as fallback
                n_ch = meta["n_channels"]
                positions = self._uniform_sphere_positions(n_ch)

            positions = np.array(positions, dtype=np.float32)

            return {
                "eeg": torch.from_numpy(data.copy().astype(np.float32)),
                "chan_pos": torch.from_numpy(positions),
                "n_channels": meta["n_channels"],
                "n_samples": meta["n_samples"],
                "quality_score": meta.get("quality_score", 1.0),
                "subject": meta.get("subject", "unknown"),
                "has_seizure": meta.get("has_seizure", False),
            }
        except Exception:
            return None

    @staticmethod
    def _uniform_sphere_positions(n_ch: int) -> np.ndarray:
        """Generate evenly spaced positions on a unit sphere (fallback)."""
        indices = np.arange(0, n_ch, dtype=np.float64) + 0.5
        phi = np.arccos(1 - 2 * indices / n_ch)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def __iter__(self) -> Iterator[dict]:
        """Stream segments indefinitely (required by ZUNA training loop)."""
        indices = list(range(len(self.meta_files)))

        while True:  # infinite stream for training
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                segment = self._load_segment(self.meta_files[idx])
                if segment is not None:
                    yield segment

    def __len__(self) -> int:
        return len(self.meta_files)


def prepare_clinical_data(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """One-shot conversion: preprocessed TUH segments → ZUNA v3 format.

    Creates the metadata/*.json + .dat file structure that EEGDataset_v3 expects.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    print(f"Converting {len(json_files)} segments to ZUNA v3 format...")

    for jf in json_files:
        meta = json.loads(jf.read_text())

        # Build ZUNA-compatible metadata
        zuna_meta = {
            "n_channels": meta["n_channels"],
            "n_samples": meta["n_samples"],
            "duration_sec": meta.get("duration", meta["n_samples"] / 256.0),
            "xyz": meta.get("scalp_positions_3d", None),
            "dat_file": str(jf.with_suffix(".dat").absolute()),
            "quality_file": str(jf.with_suffix(".q.dat").absolute()),
            "is_epoched": False,
        }

        # Symlink or copy the .mmap → .dat
        mmap_path = jf.with_suffix(".mmap")
        dat_path = jf.with_suffix(".dat")
        if mmap_path.exists() and not dat_path.exists():
            dat_path.symlink_to(mmap_path.resolve())

        # Write ZUNA metadata
        out_meta = meta_dir / f"{jf.stem}.json"
        out_meta.write_text(json.dumps(zuna_meta))

    print(f"Done. {len(json_files)} segments ready in {output_dir}")
