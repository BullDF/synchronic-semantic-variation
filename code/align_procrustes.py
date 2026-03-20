import os
import argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec

emb_dir = '../embeddings'

corpora = ['arxiv', 'yelp', 'ciao', 'reddit']

parser = argparse.ArgumentParser()
parser.add_argument('method', choices=['word2vec', 'svd'])
parser.add_argument('source', choices=corpora)
parser.add_argument('target', choices=corpora)
args = parser.parse_args()

assert args.source != args.target, 'Source and target must be different'

out_dir = f'{emb_dir}/{args.method}_aligned'
os.makedirs(out_dir, exist_ok=True)

def load_embeddings(method, corpus):
    if method == 'word2vec':
        m = Word2Vec.load(f'{emb_dir}/word2vec/{corpus}.model')
        vocab = list(m.wv.index_to_key)
        vecs = np.array([m.wv[w] for w in vocab])
    else:
        vecs = np.load(f'{emb_dir}/svd/{corpus}.npy')
        with open(f'{emb_dir}/svd/{corpus}.vocab') as f:
            vocab = [line.strip() for line in f]
    return vocab, vecs

print(f'Loading {args.source}...')
src_vocab, src_vecs = load_embeddings(args.method, args.source)
print(f'Loading {args.target}...')
tgt_vocab, tgt_vecs = load_embeddings(args.method, args.target)

src_w2i = {w: i for i, w in enumerate(src_vocab)}
tgt_w2i = {w: i for i, w in enumerate(tgt_vocab)}

common = sorted(set(src_vocab) & set(tgt_vocab))
print(f'Common vocabulary: {len(common)} words')

src_idx = [src_w2i[w] for w in common]
tgt_idx = [tgt_w2i[w] for w in common]

A = src_vecs[src_idx]
B = tgt_vecs[tgt_idx]

# L2-normalize rows before Procrustes (standard practice)
A = A / np.linalg.norm(A, axis=1, keepdims=True)
B = B / np.linalg.norm(B, axis=1, keepdims=True)

print('Running Procrustes...')
W, _ = orthogonal_procrustes(A, B)

# Apply rotation to full source embedding matrix
src_vecs_norm = src_vecs / np.linalg.norm(src_vecs, axis=1, keepdims=True)
src_aligned = src_vecs_norm @ W

prefix = f'{out_dir}/{args.source}_to_{args.target}'
np.save(f'{prefix}.npy', src_aligned)
with open(f'{prefix}.vocab', 'w') as f:
    for w in src_vocab:
        f.write(w + '\n')

print(f'Saved {prefix}.npy (shape {src_aligned.shape}) and {prefix}.vocab')
print(f'Target vocab saved as reference: {len(tgt_vocab)} words')

# Also save normalized target for convenience
tgt_vecs_norm = tgt_vecs / np.linalg.norm(tgt_vecs, axis=1, keepdims=True)
tgt_prefix = f'{out_dir}/{args.target}_normalized'
np.save(f'{tgt_prefix}.npy', tgt_vecs_norm)
with open(f'{tgt_prefix}.vocab', 'w') as f:
    for w in tgt_vocab:
        f.write(w + '\n')
print(f'Saved normalized target: {tgt_prefix}.npy')
