# EEG Medical — Open EEG Foundation Models for Healthcare

**Fine-tune Zyphra's ZUNA1.1 EEG foundation model on clinical EEG data for medical applications. Free and open-source (AGPL-3.0).**

> This repository is the canonical, independently-maintained home of the EEG Medical project. It is not part of the 1bit.systems family of projects. A public status/information site describing this project is mirrored at [eeg-medical.1bit.systems](https://eeg-medical.1bit.systems), but development, code, and data handling all live here.

## Why

Clinical EEG is underserved by AI. Most models are proprietary, locked behind paywalls, or trained only on healthy subjects. ZUNA1.1 is a 380M-parameter diffusion autoencoder that reconstructs, denoises, and upsamples EEG from any montage — but it was trained on public research datasets, not clinical-grade data.

This project bridges that gap:

- **Fine-tune ZUNA1.1 on TUH EEG Corpus** — the largest public clinical EEG dataset (30,000+ recordings, seizure and non-seizure)
- **Medical evaluation benchmarks** — seizure detection, artifact removal, montage reconstruction for clinical montages (10-20, neonatal, ECoG)
- **Pre-trained checkpoints** — freely available on HuggingFace for anyone to use

## What ZUNA1.1 does

| Capability | Clinical relevance |
|------------|-------------------|
| **Denoise** channels | Clean up muscle/eye-blink artifacts without losing signal |
| **Reconstruct** missing channels | Fix dead electrodes, extend recordings from sparse montages |
| **Upsample** montages | Predict 128-channel EEG from a standard 19-channel clinical recording |
| **Channel-agnostic** (3D scalp coordinates) | Works on any electrode layout — no retraining per montage |

## Quick start

```bash
# Install
git clone https://github.com/1bit-systems/eeg-medical.git && cd eeg-medical
pip install -e ".[gpu]"

# Download TUH EEG (requires Temple University access grant)
python scripts/download_tuh.py --output data/tuh_raw

# Preprocess TUH data into ZUNA-compatible format
python scripts/preprocess_tuh.py --input data/tuh_raw --output data/tuh_processed

# Fine-tune ZUNA1.1 on clinical EEG
python scripts/train_zuna.py --config configs/tuh_clinical_finetune.yaml

# Benchmarks live in eeg_medical/evaluation/ (see tests/ for usage)
```

## Project structure

```
eeg_medical/
├── data/ # TUH data loader, preprocessing, quality scoring
├── training/ # Fine-tuning loop, distributed training, config parsing
├── evaluation/ # Clinical benchmarks: seizure detection, artifact removal
├── models/ # ZUNA1.1 wrapper, LoRA adapters, task heads
└── visualization/ # EEG plots, reconstruction overlays
configs/ # YAML config files per experiment
scripts/ # Entry-point scripts
notebooks/ # Exploratory analysis
tests/ # Unit tests
```

## Medical benchmarks (planned)

- **Seizure detection** — classify ictal vs interictal vs normal EEG segments
- **Artifact robustness** — reconstruct channels corrupted by clinical artifacts (movement, electrode pop, muscle)
- **Montage generalization** — evaluate on clinical montages (10-20, neonatal, intraoperative ECoG)
- **Pathology-aware denoising** — preserve clinically relevant features while removing noise
- **Cross-dataset transfer** — fine-tune on TUH, evaluate on other clinical datasets

## Data access

TUH EEG Corpus requires an access request from [Temple University](https://www.isip.piconepress.com/projects/tuh_eeg/). Once granted:

1. Place EDF files under `data/tuh_raw/`
2. Run preprocessing: `python scripts/preprocess_tuh.py`
3. Output lands in `data/tuh_processed/` as ZUNA-compatible `.json` + `.mmap` pairs

This project never redistributes the TUH corpus, in raw or processed form, to any third party, consistent with Temple University's data use agreement. Only this project's own trained model weights and artifacts are intended for open distribution.

## Citation

Built on ZUNA1.1 by Zyphra ([GitHub](https://github.com/Zyphra/zuna), [HuggingFace](https://huggingface.co/Zyphra/ZUNA1.1)).

## License

Code in this repository is licensed under AGPL-3.0 — free for research and commercial use, with the condition that any network/service deployments must release their modifications.

The TUH EEG Corpus itself is **not** relicensed by this project and is **not** CC-BY-4.0 — it remains governed entirely by Temple University's own data use agreement, which prohibits redistribution to third parties. This project uses that data only internally for training and evaluation, and never re-publishes the raw or processed clinical recordings.

Any of this project's own outputs that are released openly — fine-tuned model weights, checkpoints, and aggregate (non-identifying) benchmark results — are made freely available, separate and distinct from the TUH corpus itself. No closed-source profiteering from this research.
