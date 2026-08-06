#!/usr/bin/env python3
"""Download TUH EEG Corpus (requires Temple University access grant).

After receiving access:
1. Set TUH_USERNAME and TUH_PASSWORD environment variables
2. Run: python scripts/download_tuh.py --output data/tuh_raw

The TUH EEG Corpus is hosted at:
https://www.isip.piconepress.com/projects/tuh_eeg/

Apply for access at: https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download TUH EEG Corpus")
    p.add_argument("--output", type=str, default="data/tuh_raw")
    p.add_argument("--dataset", type=str, default="tuh_eeg_seizure",
                   choices=["tuh_eeg_seizure", "tuh_eeg_abnormal", "tuh_eeg_artifact"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    username = os.environ.get("TUH_USERNAME")
    password = os.environ.get("TUH_PASSWORD")

    if not username or not password:
        print("Set TUH_USERNAME and TUH_PASSWORD environment variables.", file=sys.stderr)
        print("\nApply for access at:", file=sys.stderr)
        print("  https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset} to {output}")
    print("This is a stub — implement with your preferred download method.")
    print(f"Username: {username}")
    print(f"Dataset: {args.dataset}")

    # TUH provides rsync or direct HTTP download.
    # Example rsync (requires Temple VPN or approved IP):
    # subprocess.run([
    #     "rsync", "-avP",
    #     f"{username}@www.isip.piconepress.com:data/tuh_eeg_seizure/",
    #     str(output),
    # ])

    print("\nAfter downloading, run preprocessing:")
    print(f"  python scripts/preprocess_tuh.py --input {output} --output data/tuh_processed")


if __name__ == "__main__":
    main()
