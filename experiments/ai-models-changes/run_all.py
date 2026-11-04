"""Extract features if needed, then evaluate the planned models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/nadavnissim/Downloads/photocleaner-export")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    py = sys.executable
    extract_cmd = [py, str(ROOT / "extract_features.py"), "--data", args.data]
    if args.force_extract:
        extract_cmd.append("--force")
    subprocess.check_call(extract_cmd)
    subprocess.check_call(
        [py, str(ROOT / "evaluate_models.py"), "--seeds", str(args.seeds)]
    )


if __name__ == "__main__":
    main()
