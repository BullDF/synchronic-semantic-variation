# Plan: Train Word2Vec, PPMI, and SVD on Reddit Data

You have **24 CSV files** (one per month), each with columns: `id`, `time` (Unix timestamp), `content`.

---

## Step 1: Raw data location

Your 24 CSV files are in **`reddit_filtered/`** (project root). The conversion script uses that folder by default. Month is inferred from the `time` column in each row.

---

## Step 2: Convert CSV → JSONL (one file per month)

Run the provided script to produce **24 JSONL files** in the format the trainers expect:

- **Input**: directory of CSVs with columns `id`, `time`, `content`.
- **Output**: `data/reddit_YYYY_MM.jsonl` (one per month), each line:  
  `{"id": "...", "time": 1262304000, "text": "First example row for testing.", "year": 2010, "month": 1}`

Each document is one row; the text field used by the trainers is `text`.

```bash
cd code
python filter_reddit.py
```

(Input defaults to `../reddit_filtered`; output to `../data/`. Override with `python filter_reddit.py /path/to/csvs --out-dir /path/to/out`.)

**Optional – two corpora by year:**  
If you prefer **2 corpora** (e.g. 2010 vs 2011) instead of 24, you can merge after conversion:

- `reddit_2010.jsonl` = concat of `reddit_2010_01.jsonl` … `reddit_2010_12.jsonl`
- `reddit_2011.jsonl` = concat of `reddit_2011_01.jsonl` … `reddit_2011_12.jsonl`

Then point the trainers at these two paths (see Step 3).

---

## Step 3: Register Reddit in the three trainers

The training scripts use a **corpora** dict: corpus name → `(path, text_field)`.

**Option A – 24 monthly corpora (e.g. for diachronic or per-month variation)**  
Add 24 entries, e.g. `reddit_2010_01`, …, `reddit_2011_12`, each pointing to the corresponding JSONL and field `'text'`.  
Easier to do this **programmatically** (e.g. glob `data/reddit_*.jsonl` and build the dict) so the code doesn’t list 24 names by hand.

**Option B – 2 yearly corpora (simplest, matches “two groups” setup)**  
Add two entries:

- `reddit_2010` → `(data/reddit_2010.jsonl, 'text')`
- `reddit_2011` → `(data/reddit_2011.jsonl, 'text')`

Then you train once per corpus (2 runs per script) and alignment is one pair: `reddit_2010` ↔ `reddit_2011`.

**Files to edit:**

- `code/train_word2vec.py` — add Reddit corpus/corpora to `corpora` and extend `parser.add_argument('corpus', choices=...)`.
- `code/train_ppmi.py` — same.
- `code/train_svd.py` — same.

If you use **Option B**, you can add the two entries by hand. If you use **Option A**, add a loop or glob so the script discovers all `reddit_YYYY_MM.jsonl` and builds the list of corpus names and paths.

---

## Step 4: Train the three models

**Word2Vec** (one run per corpus):

```bash
python train_word2vec.py reddit_2010   # and reddit_2011 for Option B
# For 24 months: loop over reddit_2010_01 … reddit_2011_12
```

**PPMI** (same):

```bash
python train_ppmi.py reddit_2010
python train_ppmi.py reddit_2011
```

**SVD** (same):

```bash
python train_svd.py reddit_2010
python train_svd.py reddit_2011
```

Outputs:

- Word2Vec: `embeddings/word2vec/reddit_2010.model`, `reddit_2011.model`
- PPMI: `embeddings/ppmi/reddit_2010.npz` + `.vocab`, same for `reddit_2011`
- SVD: `embeddings/svd/reddit_2010.npy` + `.vocab`, same for `reddit_2011`

---

## Step 5: Align embeddings (Word2Vec and SVD only)

Current `align_procrustes.py` is **pairwise**: (source, target). For Reddit with 2 corpora:

- Add `reddit_2010` and `reddit_2011` to the `corpora` list in `align_procrustes.py`.
- Run alignment for the pair you care about, e.g. align one to the other:

```bash
python align_procrustes.py word2vec reddit_2010 reddit_2011
python align_procrustes.py word2vec reddit_2011 reddit_2010
python align_procrustes.py svd reddit_2010 reddit_2011
python align_procrustes.py svd reddit_2011 reddit_2010
```

(Exact output filenames depend on how `align_procrustes.py` names files; typically `*_to_*` and `*_normalized`.)

PPMI is used **unaligned** (shared-vocab distance in the existing pipeline).

---

## Step 6: Compute distances and rank

Extend `compute_distances.py` so one of the **pairs** is Reddit vs Reddit (or Reddit vs another corpus), e.g.:

- Add `('reddit_2010', 'reddit_2011')` to the `pairs` list (and ensure the aligned files for that pair exist).

Then run:

```bash
python compute_distances.py
```

You get distance TSVs for word2vec, svd, and ppmi for that pair (and any other pairs you added).

---

## Summary

| Step | Action |
|------|--------|
| 1 | Put 24 CSVs in `data/raw/reddit/` with clear names (e.g. by month). |
| 2 | Run `filter_reddit.py` → 24 JSONL in `data/` (or merge to 2 by year). |
| 3 | Add Reddit corpus/corpora to `train_word2vec.py`, `train_ppmi.py`, `train_svd.py`. |
| 4 | Train: `train_word2vec.py`, `train_ppmi.py`, `train_svd.py` for each Reddit corpus. |
| 5 | Align Word2Vec and SVD with `align_procrustes.py` for the Reddit pair(s). |
| 6 | Add Reddit pair to `compute_distances.py` and run to get distance rankings. |

If you use **2 corpora (2010 vs 2011)**, you only need to add two corpus names and two paths everywhere; the rest of the pipeline stays the same.
