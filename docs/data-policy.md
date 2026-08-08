# Data Policy

This document describes exactly what data this project uses, how it is handled, and what is — and is never — redistributed.

## Source data

This project fine-tunes Zyphra's ZUNA1.1 EEG foundation model using the **Temple University Hospital (TUH) EEG Corpus**, the largest public clinical EEG dataset (30,000+ recordings, seizure and non-seizure). Access to the TUH EEG Corpus requires a data access request directly from [Temple University](https://www.isip.piconepress.com/projects/tuh_eeg/) and is governed entirely by Temple's own data use agreement.

## What that agreement requires

Under Temple's data use agreement, this project:

- Will not release the TUH data, in raw or processed form, to any third party
- Will not attempt to re-identify any subject in the corpus
- Will not use the data for any malicious purpose
- Will delete the data when the project's use of it is finished

## What this means in practice

- Raw EDF files and processed `.json` / `.mmap` pairs live only inside this project's own infrastructure (`data/tuh_raw/`, `data/tuh_processed/`). They are never committed to this repository, never uploaded to Hugging Face or Freenet, and never published on the public status site.
- The public site at [eeg-medical.1bit.systems](https://eeg-medical.1bit.systems) only ever shows **aggregate, non-identifying pipeline metrics** — for example, a count of recordings synced or a percentage of the corpus processed. No file listings, no per-subject data, no patient information.
- Only this project's own outputs — fine-tuned model weights, checkpoints, and aggregate (non-identifying) benchmark results — are ever distributed outside this project's infrastructure. See [Model Distribution](./model-distribution.md).

## Not a diagnostic tool

This is a research pipeline, not a diagnostic tool. Outputs have not been validated for clinical decision-making and should not be used to diagnose or treat patients.
