# synchronic-semantic-variation

CSC2611 project (University of Toronto, Winter 2026).

Adapts diachronic semantic change detection methods to a synchronic setting: instead of tracking how word meanings shift over time, we measure how word meanings differ across social groups (arXiv, Yelp, Ciao, Reddit) within the same time window (2010–2011).

We train word embeddings (word2vec, PPMI, SVD) separately per corpus, align them with Procrustes, and rank words by their cross-group semantic variation.

## Pipeline

1. Filter corpora to 2010–2011 window
2. Train embeddings per corpus — `code/train_word2vec.py`, `train_ppmi.py`, `train_svd.py`
3. Align with Procrustes (word2vec, SVD); PPMI is naturally aligned
4. Compute cosine distances per word across aligned embeddings
5. Rank by semantic variation; evaluate with Spearman correlation across methods
