import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

results_dir = '../results'

def load_rankings(method, src, tgt):
    path = f'{results_dir}/{method}_{src}_{tgt}.tsv'
    words, dists = [], []
    with open(path) as f:
        for line in f:
            w, d = line.strip().split('\t')
            words.append(w)
            dists.append(float(d))
    return {w: d for w, d in zip(words, dists)}

ppmi = load_rankings('ppmi', 'yelp', 'ciao')
svd = load_rankings('svd', 'yelp', 'ciao')

common = sorted(set(ppmi) & set(svd))
print(f'Common vocabulary: {len(common)} words')

ppmi_dists = np.array([ppmi[w] for w in common])
svd_dists = np.array([svd[w] for w in common])

r, p = spearmanr(ppmi_dists, svd_dists)
print(f'Spearman r={r:.4f}, p={p:.2e}')

# Ranks (1 = most shifted)
ppmi_ranks = len(common) + 1 - ppmi_dists.argsort().argsort()
svd_ranks = len(common) + 1 - svd_dists.argsort().argsort()

fig, ax = plt.subplots(figsize=(5, 5))

# Subsample for readability — too many points otherwise
rng = np.random.default_rng(2611)
idx = rng.choice(len(common), size=min(3000, len(common)), replace=False)

ax.scatter(ppmi_ranks[idx], svd_ranks[idx], s=2, alpha=0.3, color='steelblue', linewidths=0)

ax.set_xlabel('PPMI rank (Yelp vs Ciao)')
ax.set_ylabel('SVD rank (Yelp vs Ciao)')
ax.set_title(f'PPMI vs SVD semantic shift ranking\nYelp vs Ciao — Spearman r = {r:.2f}')

plt.tight_layout()
plt.savefig('../results/spearman_ppmi_svd_yelp_ciao.pdf', bbox_inches='tight')
plt.savefig('../results/spearman_ppmi_svd_yelp_ciao.png', dpi=150, bbox_inches='tight')
print('Saved to ../results/spearman_ppmi_svd_yelp_ciao.png')
