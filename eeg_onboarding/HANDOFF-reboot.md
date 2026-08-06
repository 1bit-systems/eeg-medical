# EEG / ZUNA session — handoff before reboot (2026-08-05)

Everything lives under `/home/bcloud/eeg/` (venv + scripts + HBN data) and the 1bit
repo is on branch `feat/eeg-zuna-research`. Nothing is running (HBN autoencoder was
intentionally stopped — it was an off-plan side quest).

## Persistent state (survives reboot — all on nvme, off tmpfs)
- venv: `/home/bcloud/eeg/venv` (moabb 1.5.0, torch 2.13 cpu, mne 1.12). Activate:
  `/home/bcloud/eeg/venv/bin/python`
- HBN RestingState corpus: `/home/bcloud/eeg/hbn_r1` = 60 subjects / 6.2 GB
  (129ch @500 Hz, ~400s each, 60 `.set` + 300 sidecars). Pull more: `venv/bin/python pull_resting.py <n>`
- MOABB data: `~/mne_data` (~744 MB; BNCI2014_001 9-subj motor imagery)
- Script index: `1_scaleup.py` (9-subj CSP+LDA, 77.2%), `2_search.py` (146 MOABB datasets),
  `3_seqmodel.py` (raw LSTM, chance), `4_eegnet.py` (raw EEGNet, 78.6%), `5_xsubj.py`
  (cross-subj, 66.4%), `6_probe_bandpass.py` (8-30Hz BP: 89.7->96.6 within, 66.4->71.1 cross),
  `7_xsubj_bandpass.py`, `8_ceiling.py` (per-subject ceiling: strong ~96%, weak ~60%),
  `9_pretrain_hbn.py` (autoencoder, /dev currently stopped)
- OpenNeuro CLI (Deno): `/home/bcloud/.deno/bin/openneuro` v5.4.0; auth config
  `~/.config/openneuro/config.json` (perm 600). Deno at `/home/bcloud/.deno/bin/deno`.
- Bun at `/home/bcloud/.bun/bin/bun` (on PATH only via absolute path).
- TUEG editable form saved: `/home/bcloud/eeg/tuh_eeg_form_OFFICIAL_EDITABLE.pdf`
  (user already submitted; just waiting approval — do NOT resend flat one in ~/Downloads)
- tueg_tools reference: `/home/bcloud/eeg/tueg_tools_ref.py` + `README`

## Remaining issues
1. **Stopped mid-run:** HBN autoencoder (`9_pretrain_hbn.py`) — data ready (12,502 windows),
   epoch1 val_mse=0.459. It's OFF-PLAN (the committed ZUNA port matters, not this).
   Only resume if we explicitly choose the "corpus/pretrain" lane and accept ~40 CPU-min cost.
   Note: torch CPU is thread-thrashy on this box — I pinned `torch.set_num_threads(4)`.
2. **TUEG access — WAITING on approval, then use ssh+rsync** (NOT the old HTTP/tueg-tools downloader).
   - test cmd: `rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519" nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST .`
   - main: `rsync -auvxL ... :data/tuh_eeg/tuh_eeg/v2.0.2/ /home/bcloud/tuheg/` (large — pick corpora)
   - ed25519 pub fingerprint `SHA256:hQJA3VUAF5pGjx8UBCC5Uh3AU4iY7Lwq1g9IuQsMIgQ` (send .pub when asked)
3. **Decide the lane** (docs/research/eeg-zuna-impl-plan.md is the committed work; CPU core DONE):
   a. corpus/preprocessing hardening (HBN/MOABB/OpenNeuro → plan's pilot corpus), or
   b. ZUNA-in-1bit deferred phases (`1bit zuna` subcommand/server, GPU 4D-RoPE kernel), or
   c. wait on TUEG.
4. **JWT hygiene:** the key pasted into chat earlier is compromised — revoke at openneuro.org/keygen
   if not already; the on-disk one (`~/Documents/open neuro api key.txt`) is the good copy.
5. **Downloads disk:** ~120 GB of Xilinx/Vitis/FPGA tars in ~/Downloads — unrelated to EEG,
   can be pruned if space needed (not touched this session).

## Disk
- `/home/bcloud/eeg` 7.6 GB, `~/mne_data` 744 MB, ~525 GB free. All on nvme. Nothing on tmpfs.
