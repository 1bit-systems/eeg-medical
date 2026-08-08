#!/usr/bin/env python3
"""Download TUH EEG Corpus via rsync (requires Temple University access grant).

TUH is hosted at www.isip.piconepress.com. Apply for access at:
https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

On approval, NEDC sends an ssh key + rsync instructions. The mandatory test
before the full pull is the TEST corpus:

    python scripts/download_tuh.py --output data/tuh_raw

Then pull the full corpus (26,846 recordings) with:

    python scripts/download_tuh.py --remote data/tuh_eeg/tuh_eeg/v2.0.2 --output data/tuh_raw
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download TUH EEG Corpus via rsync")
    p.add_argument("--output", type=str, default="data/tuh_raw")
    p.add_argument(
        "--remote",
        type=str,
        default="data/tuh_eeg/TEST",
        help="Remote path on www.isip.piconepress.com (TEST corpus by default)",
    )
    p.add_argument(
        "--ssh-key",
        type=str,
        default=os.environ.get("TUH_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
    )
    p.add_argument("--user", type=str, default="nedc-tuh-eeg")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("rsync") is None:
        print("rsync not found on PATH.", file=sys.stderr)
        sys.exit(1)

    ssh_key = Path(args.ssh_key).expanduser()
    if not ssh_key.exists():
        print(
            f"SSH key not found: {ssh_key}\n"
            "TUH access is granted via ssh key, not username/password. Apply at\n"
            "https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
            file=sys.stderr,
        )
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync", "-auvxL",
        "-e", f"ssh -i {ssh_key}",
        f"{args.user}@www.isip.piconepress.com:{args.remote}",
        str(output),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\nAfter downloading, run preprocessing:")
    print(f"  python scripts/preprocess_tuh.py --input {output} --output data/tuh_processed")


if __name__ == "__main__":
    main()
