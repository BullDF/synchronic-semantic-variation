# download arxiv metadata from kaggle
# need kaggle api token at ~/.kaggle/kaggle.json first

import subprocess
from pathlib import Path

out_dir = Path('data/raw')
out_dir.mkdir(parents=True, exist_ok=True)

subprocess.run([
    'python3', '-m', 'kaggle',
    'datasets', 'download',
    '-d', 'Cornell-University/arxiv',
    '-p', str(out_dir),
    '--unzip',
], check=True)
