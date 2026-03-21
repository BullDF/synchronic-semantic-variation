import os
import argparse
import numpy as np
from scipy.sparse import load_npz

emb_dir = '../embeddings'
out_dir = '../results'
os.makedirs(out_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--top_k', type=int, default=2000)
args = parser.parse_args()

def load_ppmi(corpus):
    m = load_npz(f'{emb_dir}/ppmi/{corpus}.npz').tocsr()
    with open(f'{emb_dir}/ppmi/{corpus}.vocab') as f:
        vocab = [line.strip() for line in f]
    return vocab, m

print(f'Loading {args.source}...')
src_vocab, src_m = load_ppmi(args.source)
print(f'Loading {args.target}...')
tgt_vocab, tgt_m = load_ppmi(args.target)

src_w2i = {w: i for i, w in enumerate(src_vocab)}
tgt_w2i = {w: i for i, w in enumerate(tgt_vocab)}

common = sorted(set(src_vocab) & set(tgt_vocab))
shared_ctx = sorted(set(src_vocab) & set(tgt_vocab))
print(f'Common vocab: {len(common)}, shared context: {len(shared_ctx)}')

src_ctx_idx = np.array([src_w2i[w] for w in shared_ctx])
tgt_ctx_idx = np.array([tgt_w2i[w] for w in shared_ctx])

# Rank shared context words by average column sum across both corpora
src_col_sums = np.array(src_m[:, src_ctx_idx].sum(axis=0)).flatten()
tgt_col_sums = np.array(tgt_m[:, tgt_ctx_idx].sum(axis=0)).flatten()
avg_col_sums = (src_col_sums + tgt_col_sums) / 2

top_k = min(args.top_k, len(shared_ctx))
top_idx = np.argsort(avg_col_sums)[::-1][:top_k]
print(f'Using top {top_k} context words by column sum')

src_ctx_top = src_ctx_idx[top_idx]
tgt_ctx_top = tgt_ctx_idx[top_idx]

cos_dist = []
for w in common:
    u = np.asarray(src_m[src_w2i[w], :][:, src_ctx_top].todense()).flatten().astype(float)
    v = np.asarray(tgt_m[tgt_w2i[w], :][:, tgt_ctx_top].todense()).flatten().astype(float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        cos_dist.append(1.0)
    else:
        cos_dist.append(1 - np.dot(u, v) / (nu * nv))

out = f'{out_dir}/ppmi_top{top_k}_{args.source}_{args.target}.tsv'
with open(out, 'w') as f:
    for w, d in sorted(zip(common, cos_dist), key=lambda x: -x[1]):
        f.write(f'{w}\t{d:.6f}\n')
print(f'Saved {out} ({len(common)} words)')
