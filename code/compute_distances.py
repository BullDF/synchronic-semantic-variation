import os
import numpy as np
from scipy.sparse import load_npz

emb_dir = '../embeddings'
out_dir = '../results'
os.makedirs(out_dir, exist_ok=True)

pairs = [('arxiv', 'yelp'), ('arxiv', 'ciao'), ('arxiv', 'reddit'), ('yelp', 'ciao'), ('yelp', 'reddit'), ('ciao', 'reddit')]

def load_npy_vocab(path):
    vecs = np.load(f'{path}.npy')
    with open(f'{path}.vocab') as f:
        vocab = [line.strip() for line in f]
    return vocab, vecs

def load_ppmi_vocab(corpus):
    m = load_npz(f'{emb_dir}/ppmi/{corpus}.npz').tocsr()
    with open(f'{emb_dir}/ppmi/{corpus}.vocab') as f:
        vocab = [line.strip() for line in f]
    return vocab, m

for src, tgt in pairs:
    print(f'\n=== {src} vs {tgt} ===')

    for method in ['word2vec', 'svd']:
        src_vocab, src_vecs = load_npy_vocab(f'{emb_dir}/{method}_aligned/{src}_to_{tgt}')
        tgt_vocab, tgt_vecs = load_npy_vocab(f'{emb_dir}/{method}_aligned/{tgt}_normalized')

        src_w2i = {w: i for i, w in enumerate(src_vocab)}
        tgt_w2i = {w: i for i, w in enumerate(tgt_vocab)}
        common = sorted(set(src_vocab) & set(tgt_vocab))

        src_idx = [src_w2i[w] for w in common]
        tgt_idx = [tgt_w2i[w] for w in common]

        # cosine distance = 1 - dot product (vectors are already L2-normalized)
        cos_sim = np.sum(src_vecs[src_idx] * tgt_vecs[tgt_idx], axis=1)
        cos_dist = 1 - cos_sim

        out = f'{out_dir}/{method}_{src}_{tgt}.tsv'
        with open(out, 'w') as f:
            for w, d in sorted(zip(common, cos_dist), key=lambda x: -x[1]):
                f.write(f'{w}\t{d:.6f}\n')
        print(f'{method}: {len(common)} common words, saved {out}')

    # PPMI — keep sparse, compute cosine distance row by row to avoid dense matrix
    src_vocab, src_m = load_ppmi_vocab(src)
    tgt_vocab, tgt_m = load_ppmi_vocab(tgt)

    src_w2i = {w: i for i, w in enumerate(src_vocab)}
    tgt_w2i = {w: i for i, w in enumerate(tgt_vocab)}
    common = sorted(set(src_vocab) & set(tgt_vocab))

    all_src_ctx = set(range(src_m.shape[1]))
    all_tgt_ctx = set(range(tgt_m.shape[1]))
    shared_ctx = sorted(set(src_vocab) & set(tgt_vocab))
    src_ctx_idx = [src_w2i[w] for w in shared_ctx]
    tgt_ctx_idx = [tgt_w2i[w] for w in shared_ctx]

    cos_dist = []
    for w in common:
        u = np.asarray(src_m[src_w2i[w], :][:, src_ctx_idx].todense()).flatten()
        v = np.asarray(tgt_m[tgt_w2i[w], :][:, tgt_ctx_idx].todense()).flatten()
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            cos_dist.append(1.0)
        else:
            cos_dist.append(1 - np.dot(u, v) / (nu * nv))

    out = f'{out_dir}/ppmi_{src}_{tgt}.tsv'
    with open(out, 'w') as f:
        for w, d in sorted(zip(common, cos_dist), key=lambda x: -x[1]):
            f.write(f'{w}\t{d:.6f}\n')
    print(f'ppmi: {len(common)} common words, saved {out}')
