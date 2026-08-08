# Model Distribution

This document describes where fine-tuned model weights and artifacts from this project are published once training completes.

## What gets distributed

Only this project's own outputs are ever distributed:

- Fine-tuned model checkpoints (weights derived from fine-tuning Zyphra's ZUNA1.1 on the TUH EEG Corpus)
- Model cards describing training configuration and evaluation results
- Aggregate, non-identifying benchmark results

## Where

- **[Hugging Face](https://huggingface.co)** — the primary home for pre-trained checkpoints, model cards, and versioned releases, for anyone to download and use directly.
- **[Freenet](https://freenet.org)** — a peer-to-peer mirror with no central host or company in the middle, so the weights remain available even if any single site goes down.

## What never gets distributed

The TUH EEG Corpus itself — raw or processed, in any form — never leaves this project's own infrastructure. It is never uploaded to Hugging Face, Freenet, GitHub, or anywhere else. This is a hard requirement of Temple University's data use agreement, not just a preference. See [Data Policy](./data-policy.md) for the full policy.

## Status

Fine-tuning is still in progress. No checkpoints have been published yet. This page will be updated with links once the first checkpoint is released.
