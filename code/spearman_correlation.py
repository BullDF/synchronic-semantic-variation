import os
import numpy as np
from scipy.stats import spearmanr

results_dir = '../results'

pairs = [('arxiv', 'yelp'), ('arxiv', 'ciao'), ('yelp', 'ciao')]
methods = ['word2vec', 'ppmi', 'svd']

def load_rankings(method, src, tgt):
    path = f'{results_dir}/{method}_{src}_{tgt}.tsv'
    words, dists = [], []
    with open(path) as f:
        for line in f:
            w, d = line.strip().split('\t')
            words.append(w)
            dists.append(float(d))
    return {w: d for w, d in zip(words, dists)}

print('Spearman correlations between methods (per corpus pair)')
print('=' * 60)

for src, tgt in pairs:
    print(f'\n{src} vs {tgt}')
    rankings = {m: load_rankings(m, src, tgt) for m in methods}

    # Common vocab across all three methods
    common = sorted(set(rankings['word2vec']) & set(rankings['ppmi']) & set(rankings['svd']))
    print(f'Common vocabulary: {len(common)} words')

    dists = {m: np.array([rankings[m][w] for w in common]) for m in methods}

    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            r, p = spearmanr(dists[m1], dists[m2])
            print(f'  {m1} vs {m2}: r={r:.4f}, p={p:.2e}')
