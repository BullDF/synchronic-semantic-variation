# Synchronic Semantic Variation

CSC2611 project (University of Toronto, Winter 2026).

Adapts diachronic semantic change detection methods to a synchronic setting: instead of tracking how word meanings shift over time, we measure how word meanings differ across social groups (arXiv, Yelp, Ciao, Reddit) within the same time window (2010–2011).

We train word embeddings (word2vec, PPMI, SVD) separately per corpus, align them with Procrustes, and rank words by their cross-group semantic variation.

## Pipeline

1. Filter corpora to 2010–2011 window
2. Train embeddings per corpus — `code/train_word2vec.py`, `train_ppmi.py`, `train_svd.py`
3. Align with Procrustes (word2vec, SVD); PPMI shares a fixed context vocabulary, no alignment needed
4. Compute cosine distances per word across aligned embeddings
5. Rank by semantic variation; evaluate with Spearman correlation across methods
6. Qualitative analysis via ConShift anchor words

## Setup

Dependencies: `gensim`, `numpy`, `scipy`, `kaggle`

```bash
pip install gensim numpy scipy kaggle
```

All scripts are run from inside `code/`. Paths in scripts are relative to `code/`.

## Data

`data/` and `embeddings/` and `results/` are gitignored — each person sets them up locally.

Expected structure:

```
data/
  raw/         # original downloaded files
  processed/   # filtered .jsonl files
embeddings/
  word2vec/
  ppmi/
  svd/
  word2vec_aligned/
  svd_aligned/
results/
```

| Corpus    | Domain            | Documents (2010–2011) |
|-----------|-------------------|-----------------------|
| arXiv CS  | Academic (CS)     | 16,711                |
| Yelp      | Restaurant reviews| 369,400               |
| Ciao      | Product reviews   | 48,018                |
| Reddit    | Online communities| ~16M                  |

Text field: `abstract` for arXiv, `text` for Yelp/Ciao. Reddit handled separately (see `code/filter_reddit.py`).

## Training Embeddings

All three methods use: window=4, dim=300, min_count=5. Run from `code/`:

```bash
bash train_word2vec_all.sh   # → embeddings/word2vec/{corpus}.model
bash train_ppmi_all.sh       # → embeddings/ppmi/{corpus}.npz + .vocab
bash train_svd_all.sh        # → embeddings/svd/{corpus}.npy + .vocab
```

Or run individual corpora:

```bash
python3 train_word2vec.py arxiv --sample 1e-3   # arxiv needs higher subsampling (small corpus)
python3 train_word2vec.py yelp
python3 train_word2vec.py ciao
python3 train_ppmi.py arxiv   # (same pattern for ppmi/svd)
python3 train_svd.py arxiv
```

## Alignment

Procrustes alignment for word2vec and SVD (PPMI not needed):

```bash
bash align_procrustes_all.sh
```

This aligns all 6 pairs across the 4 corpora (arxiv, yelp, ciao, reddit) for both methods. Output in `embeddings/word2vec_aligned/` and `embeddings/svd_aligned/`:
- `{source}_to_{target}.npy` + `.vocab` — aligned source vectors (L2-normalized)
- `{target}_normalized.npy` + `.vocab` — normalized target vectors

## Distances & Evaluation

Compute cosine distances between aligned embeddings for each word:

```bash
python3 compute_distances.py   # → results/{method}_{src}_{tgt}.tsv (sorted by distance descending)
```

Compute Spearman correlation between method rankings:

```bash
python3 spearman_correlation.py
```

PPMI vs SVD: r=0.44–0.61 across all pairs (consistent, since SVD is derived from PPMI).  
word2vec vs count-based: varies by domain distance — r≈0.29–0.50 for similar domains (arXiv–Reddit), near zero or negative for distant ones.

**Note on PPMI:** PPMI cosine distances collapse to ~1.0 for distant domain pairs (e.g. arXiv–Yelp). This is a known limitation: non-zero PPMI dimensions are nearly disjoint across corpora, making cosine similarity ~0 uniformly. PPMI is not usable for cross-domain ranking; use SVD and word2vec only for qualitative analysis.

## Qualitative Analysis (ConShift)

We use an iterative ConShift-style Procrustes alignment to find stable anchor words and words with high semantic variation:

```bash
python3 anchors_word2vec_conshift.py
```

See `notebooks/anchors_word2vec_results.ipynb` for visualization of top-N shifted/stable words per corpus pair. Best pairs for interpretation: **arXiv–Yelp** and **arXiv–Reddit**.
