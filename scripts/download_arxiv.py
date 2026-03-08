"""
Download the arXiv metadata dataset from Kaggle.

Requirements:
    pip install kaggle
    Place your Kaggle API token at ~/.kaggle/kaggle.json
    (Download from https://www.kaggle.com/settings -> API -> Create New Token)
"""

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading arXiv metadata to {DATA_DIR} ...")
    subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "Cornell-University/arxiv",
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        check=True,
    )
    print("Done. File saved to:", DATA_DIR / "arxiv-metadata-oai-snapshot.json")


if __name__ == "__main__":
    main()
