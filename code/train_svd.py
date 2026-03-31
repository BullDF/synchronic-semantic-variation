import json
import re
import os
import argparse
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

data_dir = '../data'

corpora = {
    'arxiv': (f'{data_dir}/processed/arxiv_cs_2010_2011.jsonl', 'abstract'),
    'yelp':  (f'{data_dir}/processed/yelp_reviews_2010_2011.jsonl', 'text'),
    'ciao':  (f'{data_dir}/processed/ciao_reviews_2010_2020.jsonl', 'text'),
}

parser = argparse.ArgumentParser()
parser.add_argument('corpus', choices=corpora.keys())
parser.add_argument('--window', type=int, default=4)
parser.add_argument('--min_count', type=int, default=5)
parser.add_argument('--cds', type=float, default=0.75)
parser.add_argument('--dim', type=int, default=300)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.0)  # negative prior; 0 for SVD per paper
args = parser.parse_args()

print(f'Corpus: {args.corpus}')
path, field = corpora[args.corpus]
out_dir = '../embeddings/svd'
os.makedirs(out_dir, exist_ok=True)

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())

print('Building vocabulary...')
word_counts = Counter()
with open(path) as f:
    for line in f:
        r = json.loads(line)
        word_counts.update(tokenize(r[field]))

vocab = sorted(w for w, c in word_counts.items() if c >= args.min_count)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f'Vocab size: {V}')

print('Counting co-occurrences...')
cooc = Counter()
with open(path) as f:
    for n, line in enumerate(f):
        r = json.loads(line)
        tokens = [t for t in tokenize(r[field]) if t in w2i]
        for i, w in enumerate(tokens):
            wi = w2i[w]
            for j in range(max(0, i - args.window), min(len(tokens), i + args.window + 1)):
                if j != i:
                    cooc[(wi, w2i[tokens[j]])] += 1
        if n % 50000 == 0:
            print(f'  {n} docs processed')

rows = np.array([k[0] for k in cooc], dtype=np.int32)
cols = np.array([k[1] for k in cooc], dtype=np.int32)
data = np.array(list(cooc.values()), dtype=np.float32)
del cooc

counts = csr_matrix((data, (rows, cols)), shape=(V, V))

# Compute unshifted PPMI (alpha=0 for SVD per Hamilton et al.)
print('Computing PPMI (alpha=0)...')
total = counts.sum()
row_sums = np.array(counts.sum(axis=1)).flatten()
col_sums = np.array(counts.sum(axis=0)).flatten()

col_sums_smooth = col_sums ** args.cds
col_sums_smooth /= col_sums_smooth.sum()

coo = counts.tocoo()
p_wc = coo.data / total
p_w = row_sums[coo.row] / total
p_c = col_sums_smooth[coo.col]

ppmi_vals = np.maximum(np.log(p_wc / (p_w * p_c)) - args.alpha, 0)
ppmi = csr_matrix((ppmi_vals, (coo.row, coo.col)), shape=(V, V))
ppmi.eliminate_zeros()

# Truncated SVD: M = U S V^T, embeddings = U * S^gamma
print(f'Running SVD (dim={args.dim}, gamma={args.gamma})...')
U, s, _ = svds(ppmi.asfptype(), k=args.dim)
assert U is not None
# svds returns singular values in ascending order, reverse to descending
idx = np.argsort(s)[::-1]
U = U[:, idx]
s = s[idx]

embeddings = U * (s ** args.gamma)

out_prefix = f'{out_dir}/{args.corpus}'
np.save(f'{out_prefix}.npy', embeddings)
with open(f'{out_prefix}.vocab', 'w') as f:
    for w in vocab:
        f.write(w + '\n')

print(f'Saved {out_prefix}.npy (shape {embeddings.shape}) and {out_prefix}.vocab')
