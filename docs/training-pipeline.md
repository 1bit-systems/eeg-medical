# Training Pipeline

This document describes how raw TUH EEG recordings become fine-tuning inputs and checkpoints.

## Overview

1. **Access** — A data access request is submitted to Temple University for the TUH EEG Corpus. Once granted, recordings are synced into this project's own infrastructure — never into this repository.
2. **Raw storage** — EDF files land under `data/tuh_raw/`.
3. **Preprocessing** — `scripts/preprocess_tuh.py` converts raw EDF recordings into ZUNA-compatible `.json` + `.mmap` pairs, applying quality scoring along the way. Output lands under `data/tuh_processed/`.
4. **Fine-tuning** — `scripts/finetune.py`, configured via `configs/tuh_clinical_finetune.yaml`, fine-tunes Zyphra's ZUNA1.1 (a 380M-parameter diffusion autoencoder) on the processed clinical data.
5. **Evaluation** — `scripts/evaluate.py` runs the fine-tuned checkpoint against clinical benchmarks such as seizure detection. These benchmarks are currently **planned, not yet completed** — see the main [README](../README.md) for current status.
6. **Checkpoints** — resulting checkpoints are saved locally, then published per the [Model Distribution](./model-distribution.md) policy.

## Aggregate progress reporting

A small reference script, `update_stats.sh`, computes aggregate pipeline counts (recordings synced, data volume synced, percentage of corpus, last sync time) and writes them to `stats.json`, which powers the public sync-stats panel on [eeg-medical.1bit.systems](https://eeg-medical.1bit.systems). This script only ever emits aggregate counts — never file listings, filenames, or any per-subject detail.

## Data handling boundaries

Every step in this pipeline stays within this project's own infrastructure until the final, explicit publication step for model weights. See [Data Policy](./data-policy.md) for the full data handling and licensing rules.
