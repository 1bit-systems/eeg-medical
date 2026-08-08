# EEG Medical Documentation

This is the documentation index for the EEG Medical project — fine-tuning Zyphra's ZUNA1.1 EEG foundation model on the Temple University Hospital (TUH) EEG Corpus for clinical applications.

This repository, [bong-water-water-bong/eeg-medical](https://github.com/bong-water-water-bong/eeg-medical), is the canonical, independently-maintained home of the project. It is not part of the 1bit.systems product family; [eeg-medical.1bit.systems](https://eeg-medical.1bit.systems) is only a public status mirror.

## Contents

- [Data Policy](./data-policy.md) — what data this project uses, how it is handled, and what is never redistributed
- [Training Pipeline](./training-pipeline.md) — how raw TUH recordings become fine-tuning inputs and checkpoints
- [Model Distribution](./model-distribution.md) — where fine-tuned model weights and artifacts are published once training completes

## Project status

Fine-tuning and evaluation are in progress. Medical benchmarks (seizure detection, artifact robustness, montage generalization, pathology-aware denoising, cross-dataset transfer) are **planned, not yet completed**. This is a research pipeline, not a diagnostic tool — outputs have not been validated for clinical decision-making and should not be used to diagnose or treat patients.

## License

Code is AGPL-3.0. The TUH EEG Corpus remains governed entirely by Temple University's own data use agreement and is never redistributed by this project. See [Data Policy](./data-policy.md) for details.
