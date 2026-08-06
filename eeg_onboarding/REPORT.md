# EEG onboarding via MOABB — working notes

Environment: venv at `/home/bcloud/eeg/venv` (moabb 1.5.0, torch 2.13 cpu, mne 1.12),
data at `/home/bcloud/mne_data`. All on nvme, off tmpfs/RAM.

## What works (verified end-to-end)

| # | Script | Task | Result |
|---|--------|------|--------|
| 1 | `1_scaleup.py` | 9-subject Leave-One-Subject-Out, CSP+LDA | **77.2%** (chance 50%); per-subject 0.51–0.99 |
| 2 | `2_search.py` | Catalog all MOABB datasets | 146 datasets (imagery 54, p300 68, ssvep 16, cvep 8) |
| 3 | `3_seqmodel.py` | Raw EEG → channel-meaned LSTM (torch) | **~50% (chance)** — time-only model fails |
| 4 | `4_eegnet.py` | Raw EEG → EEGNet-1D (channel-aware), within-subject | **78.6%** — beats LSTM by using channel dimension |
| 5 | `5_xsubj.py` | Cross-subject raw EEGNet (train 8, test 1) | **66.4%** mean (chance 50%) |
| 6 | `6_probe_bandpass.py` | Within-subject, BP is the lever | subj1: full-band 89.7% → **8-30 Hz 96.6%** |
| 7 | `7_xsubj_bandpass.py` | Cross-subject EEGNet + 8-30 Hz BP, AdamW/cosine | **71.1%** (was 66.4%) |
| 8 | `8_ceiling.py` | Within-subject, 5-fold CV, 60ep, BP (per-subject ceiling) | **mean 76.8%**; strongest ~96.5%, weakest ~58-62% |

## The key finding (for the zyphra / raw-signal-model angle)

On identical raw trials + identical split, the ONLY difference between #3 and #4
is channel-awareness:
- flatten channels — mean / LSTM: **chance (~50%)**
- treat channels as a real dimension (depthwise spatial conv): **79%**

=> Raw EEG tokenization MUST preserve the channel/spatial axis. A per-sample
sequence model over flattened signals discards the information.
Cross-subject generalization is the hard gap: 79 in-domain vs 66 cross-subject.
Band-passing to the mu/beta rhythm (8-30 Hz) is the single biggest free lever:
off-the-shelf, standard MI preprocessing, and it lifts WHAT YOU ALREADY HAVE:
  within-subject: 89.7 -> 96.6 (+7)   cross-subject: 66.4 -> 71.1 (+4.7)
That's 'undiscovered code' in practice: the naive raw model removed band info;
the signal the motor cortex actually emits lives in 8-30 Hz.

## The ceiling probe (8_ceiling.py) — where the remaining gradient really is
9 subjects, within-subject 5-fold CV, band-pass, 60ep. THE ceiling is subject-
dependent and NOT model-limited:
  strong (3,8,9): 92-96.5%  <- within reach of ~100%, needs arch/data/transfer work
  mid   (1,4,6):  68-86%
  weak  (2,5,7):  58-62%    <- signal-limited floor; the recorded MI is intrinsically weak
Mean 76.8%. Takeaway: on BNCI2014_001, chasing 100% is only possible on the strong
subjects; the weak-subject floor is set by the data, not the decoder.

## Data held locally (cost)

- BNCI2014_001 (9 subjects): ~800 MB in `~/mne_data`

## To run
```bash
cd /home/bcloud/eeg && venv/bin/python 1_scaleup.py
                    venv/bin/python 4_eegnet.py     # within-subject raw
                    venv/bin/python 5_xsubj.py      # cross-subject raw (slow, ~10 min)
```
Kill /tmp-backed anything; keep venv + data on the nvme.

## Next candidates (justify before I pull GBs)
- **PhysionetMI** (109 subj, ~6–10 GB): breadth for pretraining — same MI theme, big.
- **Different paradigm** (P300 `BI2014a`; SSVEP `Liu2022EldBETA` 100): tests model-transfer across BCI types — highest marginal value per GB.
- HBN/OpenNeuro: true scale for foundation-model pretraining.

## NOW DONE — OpenNeuro HBN pulled + verified
Pull X RestingState recordings (open, keyless S3) via `pull_resting.py`.
- 5 subjects x RestingState EEG, 129ch @500 Hz, ~404s each = **466 MB** (thin slice)
- **ds005505** (Release 1) = 136 subjects; later releases bigger (R7=381, R11=430)
- Full corpus is ~103 GB (1342 .set files) — pull only the slices you need
- S3 bucket public: `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1/`
- Verified valid signal (clean 1/f PSD), loads in MNE via EOGLAB reader.
- Your OpenNeuro API key: needed only for restricted datasets; for public HBN the
  keyless S3 path is enough. The `openneuro` CLI (Bun/Deno) is the route when you
  need authenticated data — but its `login` needs your key typed on your own
  terminal (I won't ask you to paste it in chat).

## NOW DONE — 60-subject HBN RestingState corpus (public, keyless)
60 RestingState recordings = 60 EEG .set + 300 sidecars = **6.2 GB** in `/home/bcloud/eeg/hbn_r1`
(129ch @500 Hz, ~400s each). Pull more with `pull_resting.py <n>`. 136 subjects available in R1.

## Auth key status (track b)
- openneuro CLI 5.4.0 (Deno) authenticated via `~/.config/openneuro/config.json` (perm 600).
- Works for CLI download + git-annex/datalad get on any OpenNeuro dataset (restricted blobs).
- Key finding: **HBN-EEG is OPEN** (neuro/EEG needs no DUA) — auth not required for it.
- The genuinely-restricted grand corpus is **TUEG** (Temple, 60k clinical EEGs), but it
  uses completely SEPARATE auth: isip.piconepress account + signed form + rsync creds,
  NOT the OpenNeuro JWT. It's the real foundation-model corpus to pursue next.

## TUEG access — recorded (waiting on form approval)
Form already submitted; access via SSH+rsync (as of Jan 2026 NEW method).
On approval NEDC sends ssh-key + rsync instructions.

Local readiness:
- ssh + rsync present, ed25519 key EXISTS: ~/.ssh/id_ed25519 (+ .pub)
  pub fingerprint: SHA256:hQJA3VUAF5pGjx8UBCC5Uh3AU4iY7Lwq1g9IuQsMIgQ
  (send ONLY the .pub when asked; never share the private key)
- Mandatory test command they require (before full download):
    rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" \
      nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST .
  (must be ONE line; if it fails use -auxvvvL for a diagnostic log)
- Main corpus rsync path: data/tuh_eeg/tuh_eeg/v2.0.2  (TUEG, 26,846 recordings)
- Window/rsync server: www.isip.piconepress.com, user nedc-tuh-eeg
- Always use -L (follow links). EDF format -> we have edfread.m + edfbrowser + pyedflib path.

## TUEG tooling notes (tueg-tools reviewed)
- Saved tueg_tools_ref.py + README. Good for TUEG folder/session schema: vX.Y.Z/edf/ -> tcp_ar -> subj -> ses -> *.edf; TUAB adds eval|train + normal|abnormal.
- Session ID parses as ses_no_year_month_day; anonymous `token` in filename.
- Its download() uses OLD HTTP basic-auth (username/password) — predates Jan-2026 ssh+rsync
  move. Use rsync for the actual pull; reuse only the walker/naming logic.
- It does NOT decode EDFs (only navigates). Decode with pyedflib/edfread when files land.
