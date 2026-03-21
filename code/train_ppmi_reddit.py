import csv
import re
import os
import argparse
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_csv_dir = os.path.normpath(os.path.join(_script_dir, '..', 'reddit_filtered'))

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', default=_default_csv_dir, help='Directory of CSVs (id, time, content)')
parser.add_argument('--window', type=int, default=4)
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--cds', type=float, default=0.75)   # context distribution smoothing exponent
parser.add_argument('--alpha', type=float, default=1.6094)  # negative prior log(5), shifted PPMI
args = parser.parse_args()

out_dir = os.path.normpath(os.path.join(_script_dir, '..', 'embeddings', 'ppmi'))
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

# First pass: build vocabulary
print('Building vocabulary...')
word_counts = Counter()
for text in tqdm(iter_texts_from_csv_dir(args.input_dir), desc="Vocabulary", unit=" doc"):
    word_counts.update(tokenize(text))

vocab = sorted(w for w, c in word_counts.items() if c >= args.min_count)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f'Vocab size: {V}')

# Second pass: count co-occurrences
print('Counting co-occurrences...')
cooc = Counter()
for text in tqdm(iter_texts_from_csv_dir(args.input_dir), desc="Co-occurrences", unit=" doc"):
    tokens = [t for t in tokenize(text) if t in w2i]
    for i, w in enumerate(tokens):
        wi = w2i[w]
        for j in range(max(0, i - args.window), min(len(tokens), i + args.window + 1)):
            if j != i:
                cooc[(wi, w2i[tokens[j]])] += 1

print(f'Unique (w, c) pairs: {len(cooc)}')

# Build sparse counts matrix
rows = np.array([k[0] for k in cooc], dtype=np.int32)
cols = np.array([k[1] for k in cooc], dtype=np.int32)
data = np.array(list(cooc.values()), dtype=np.float32)
del cooc

counts = csr_matrix((data, (rows, cols)), shape=(V, V))

# Compute PPMI with context distribution smoothing
print('Computing PPMI...')
total = counts.sum()
row_sums = np.array(counts.sum(axis=1)).flatten()
col_sums = np.array(counts.sum(axis=0)).flatten()

col_sums_smooth = col_sums ** args.cds
col_sums_smooth /= col_sums_smooth.sum()

coo = counts.tocoo()
p_wc = coo.data / total
p_w = row_sums[coo.row] / total
p_c = col_sums_smooth[coo.col]

ppmi_vals = np.log(p_wc / (p_w * p_c)) - args.alpha
ppmi_vals = np.maximum(ppmi_vals, 0)

ppmi = csr_matrix((ppmi_vals, (coo.row, coo.col)), shape=(V, V))
ppmi.eliminate_zeros()

# Save
out_prefix = f'{out_dir}/reddit'
save_npz(f'{out_prefix}.npz', ppmi)
with open(f'{out_prefix}.vocab', 'w') as f:
    for w in vocab:
        f.write(w + '\n')

print(f'Saved {out_prefix}.npz (shape {ppmi.shape}, nnz {ppmi.nnz}) and {out_prefix}.vocab')
