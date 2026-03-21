import csv
import re
import os
import argparse
from gensim.models import Word2Vec
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_csv_dir = os.path.normpath(os.path.join(_script_dir, '..', 'reddit_filtered'))
_default_out_dir = os.path.normpath(os.path.join(_script_dir, '..', 'embeddings', 'word2vec'))

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', default=_default_csv_dir, help='Directory of CSVs (id, time, content)')
parser.add_argument('--sample', type=float, default=1e-5)
args = parser.parse_args()

out_dir = os.path.normpath(os.path.join(_script_dir, '..', 'embeddings', 'word2vec'))
os.makedirs(out_dir, exist_ok=True)

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())

def iter_texts_from_csv_dir(csv_dir_path):
    for name in sorted(os.listdir(csv_dir_path)):
        if not name.lower().endswith('.csv'):
            continue
        path = os.path.join(csv_dir_path, name)
        with open(path, encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            try:
                first = next(reader)
            except StopIteration:
                continue
            if len(first) >= 3 and first[0].strip().lower() == 'id':
                pass
            else:
                if len(first) >= 3 and first[2].strip():
                    yield first[2].strip()
            for row in reader:
                if len(row) >= 3 and row[2].strip():
                    yield row[2].strip()

class SentenceIterator:
    """Re-iterable: each pass re-reads CSVs from disk so Word2Vec can do multiple epochs."""
    def __init__(self, csv_dir_path):
        self.csv_dir_path = csv_dir_path

    def __iter__(self):
        for text in tqdm(iter_texts_from_csv_dir(self.csv_dir_path), desc="Word2Vec", unit=" doc"):
            tokens = tokenize(text)
            if tokens:
                yield tokens

print(f'Training Word2Vec on Reddit CSVs from {args.input_dir}...')
model = Word2Vec(
    SentenceIterator(args.input_dir),
    vector_size=300,
    window=4,
    min_count=5,
    sample=args.sample,
    negative=5,
    ns_exponent=0.75,
    workers=4,
    epochs=5,
    seed=2611,
)

out_path = f'{out_dir}/reddit.model'
model.save(out_path)
print(f'Saved to {out_path} (vocab size: {len(model.wv)})')
