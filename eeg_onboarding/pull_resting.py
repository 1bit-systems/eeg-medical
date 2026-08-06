"""Sequential, thin pull of HBN R1 RestingState EEG for a small subject list.
Only .set + its .json/.tsv sidecars. No thread race, no broad nets. No sidecars not
adjacent to the target .set path.
Usage: python pull_resting.py <n_subs>   (default 5)
"""
import os, sys, requests
import list_hbn as L

BUCKET = "https://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/HBN/BIDS_EEG/cmi_bids_R1/"
DEST = "/home/bcloud/eeg/hbn_r1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    keys = L.list_prefix(PREFIX)
    # collect all RestingState .set keys with their exact sidecar siblings
    set_keys = sorted(k for k in keys if k.endswith("_task-RestingState_eeg.set"))
    # subject dirs, stable order
    subj_dirs = sorted({k.split("/")[5] for k in set_keys})
    chosen = subj_dirs[:N]
    print(f"pulling RestingState for {len(chosen)} subjects of {len(subj_dirs)} available")

    wire = []
    for sub in chosen:
        wire += [k for k in keys
                 if k.startswith(f"{PREFIX}{sub}/eeg/")
                 and k.split("/")[-1].startswith(sub + "_task-RestingState")
                 and (k.endswith(".set") or k.endswith(".json") or k.endswith(".tsv"))]
    print(f"files to fetch: {len(wire)} (~{len(chosen)*0.1:.1f} GB)")

    got = 0
    for w in wire:
        rel = w[len(PREFIX):]
        dp = os.path.join(DEST, rel)
        if os.path.exists(dp) and os.path.getsize(dp) > 100:
            continue
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        r = requests.get(BUCKET + "/" + w, timeout=120)
        with open(dp, "wb") as f:
            f.write(r.content)
        got += 1
        sz = len(r.content) / 1e6
        print(f"  [{got:2d}/{len(wire)}] {rel.split('/')[-1]} {sz:6.1f} MB", flush=True)
    print(f"done: {got} files -> {DEST}")
    du = os.popen(f"du -sh {DEST}").read().strip()
    print(du)


if __name__ == "__main__":
    main()
