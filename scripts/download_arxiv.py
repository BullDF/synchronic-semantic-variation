# download arxiv metadata from kaggle
# need kaggle api token at ~/.kaggle/kaggle.json first

import subprocess
import sys
from pathlib import Path

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

subprocess.run([
    sys.executable, "-m", "kaggle",
    "datasets", "download",
    "-d", "Cornell-University/arxiv",
    "-p", str(out_dir),
    "--unzip",
], check=True)
