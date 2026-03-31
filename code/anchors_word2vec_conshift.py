import argparse
import os

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes

try:
    from gensim.parsing.preprocessing import STOPWORDS as gensim_stop
except Exception:
    gensim_stop = frozenset()

script_dir = os.path.dirname(os.path.abspath(__file__))
emb_dir_default = os.path.normpath(os.path.join(script_dir, '..', 'embeddings'))

fallback_stop = frozenset(
    'a an the and or but if in on at to for of as by with from into through during '
    'before after above below between under again further then once here there when '
    'where why how all each both few more most other some such no nor not only own '
    'same so than too very can will just don should now'.split()
)
stopwords = gensim_stop | fallback_stop

corpora_default = ['arxiv', 'yelp', 'ciao']


def load_word2vec_matrix(corpus, emb_dir):
    path = os.path.join(emb_dir, 'word2vec', f'{corpus}.model')
    m = Word2Vec.load(path)
    vocab = list(m.wv.index_to_key)
    vecs = np.asarray([m.wv[w] for w in vocab], dtype=np.float64)
    return m, vocab, vecs


def freq_rank(wv):
    return {w: i for i, w in enumerate(wv.index_to_key)}


def word_count_wv(wv, w):
    if hasattr(wv, 'get_vecattr'):
        try:
            c = wv.get_vecattr(w, 'count')
            if c is not None:
                return int(c)
        except (KeyError, ValueError, TypeError):
            pass
    vocab = getattr(wv, 'vocab', None)
    if vocab is not None and w in vocab:
        return int(vocab[w].count)
    return 0


def build_shared_words(corpus_a, corpus_b, emb_dir, strip_stopwords, max_rank_a, max_rank_b, min_word_length):
    ma, va, Ea = load_word2vec_matrix(corpus_a, emb_dir)
    mb, vb, Eb = load_word2vec_matrix(corpus_b, emb_dir)
    ra = freq_rank(ma.wv)
    rb = freq_rank(mb.wv)

    ia = {w: i for i, w in enumerate(va)}
    ib = {w: i for i, w in enumerate(vb)}
    common = set(va) & set(vb)

    def keep(w):
        if min_word_length > 0 and len(w) < min_word_length:
            return False
        if strip_stopwords and w in stopwords:
            return False
        if max_rank_a is not None and ra.get(w, 10**9) >= max_rank_a:
            return False
        if max_rank_b is not None and rb.get(w, 10**9) >= max_rank_b:
            return False
        return True

    words = sorted(w for w in common if keep(w))
    if len(words) < 10:
        raise SystemExit(
            f'Too few shared words ({len(words)}). Relax filters (--no-stopwords, '
            '--min-rank-per-corpus, --min-word-length 0).'
        )

    idx_a = [ia[w] for w in words]
    idx_b = [ib[w] for w in words]
    ca = [word_count_wv(ma.wv, w) for w in words]
    cb = [word_count_wv(mb.wv, w) for w in words]
    return words, Ea[idx_a], Eb[idx_b], ca, cb


def procrustes_from_anchor_indices(Ea_norm, Eb_norm, anchor_idx):
    A = Ea_norm[anchor_idx]
    B = Eb_norm[anchor_idx]
    W, _ = orthogonal_procrustes(A, B)
    return W


def shift_scores(Ea_aligned, Eb_norm):
    return 1.0 - np.sum(Ea_aligned * Eb_norm, axis=1)


def select_anchors(scores, quantile, min_anchors, max_anchors):
    n = len(scores)
    k = int(np.ceil(n * quantile))
    k = max(min_anchors, min(k, max_anchors, n))
    order = np.argsort(scores)
    return order[:k]


def iterative_anchors(Ea, Eb, quantile, max_iter, min_anchors, max_anchors, tol):
    Ea_norm = Ea / np.linalg.norm(Ea, axis=1, keepdims=True)
    Eb_norm = Eb / np.linalg.norm(Eb, axis=1, keepdims=True)

    n = Ea_norm.shape[0]
    min_anchors = min(min_anchors, n)
    max_anchors = min(max_anchors, n)

    anchor_idx = np.arange(n, dtype=np.int64)
    prev_anchor = None

    for it in range(max_iter):
        W = procrustes_from_anchor_indices(Ea_norm, Eb_norm, anchor_idx)
        Ea_aligned = Ea_norm @ W
        Ea_aligned = Ea_aligned / np.linalg.norm(Ea_aligned, axis=1, keepdims=True)
        scores = shift_scores(Ea_aligned, Eb_norm)
        new_anchor_idx = select_anchors(scores, quantile, min_anchors, max_anchors)

        if len(new_anchor_idx) == len(anchor_idx) and np.array_equal(
            np.sort(new_anchor_idx), np.sort(anchor_idx)
        ):
            return new_anchor_idx, scores, it + 1
        if tol > 0 and prev_anchor is not None:
            cur = set(new_anchor_idx.tolist())
            if len(prev_anchor.symmetric_difference(cur)) <= tol:
                return new_anchor_idx, scores, it + 1
        prev_anchor = set(new_anchor_idx.tolist())
        anchor_idx = new_anchor_idx

    W = procrustes_from_anchor_indices(Ea_norm, Eb_norm, anchor_idx)
    Ea_aligned = Ea_norm @ W
    Ea_aligned = Ea_aligned / np.linalg.norm(Ea_aligned, axis=1, keepdims=True)
    scores = shift_scores(Ea_aligned, Eb_norm)
    return anchor_idx, scores, max_iter


def run_pair(source, target, emb_dir, out_dir, strip_stopwords, min_rank_per_corpus, quantile, max_iter, min_anchors, max_anchors, tol, min_word_length):
    mr = min_rank_per_corpus
    words, Ea, Eb, cnt_src, cnt_tgt = build_shared_words(
        source, target, emb_dir, strip_stopwords, mr, mr, min_word_length
    )
    anchor_idx, scores, iters = iterative_anchors(
        Ea, Eb, quantile, max_iter, min_anchors, max_anchors, tol
    )
    anchor_words = {words[i] for i in anchor_idx.tolist()}

    os.makedirs(out_dir, exist_ok=True)
    base = f'{source}_to_{target}'
    shifts_path = os.path.join(out_dir, f'{base}_shifts.tsv')
    anchors_path = os.path.join(out_dir, f'{base}_anchors.txt')
    meta_path = os.path.join(out_dir, f'{base}_meta.txt')

    col_src = f'count_{source}'
    col_tgt = f'count_{target}'
    with open(shifts_path, 'w', encoding='utf-8') as f:
        f.write(f'word\tshift\t{col_src}\t{col_tgt}\n')
        for i, w in enumerate(words):
            f.write(f'{w}\t{scores[i]:.6f}\t{int(cnt_src[i])}\t{int(cnt_tgt[i])}\n')

    with open(anchors_path, 'w', encoding='utf-8') as f:
        for w in sorted(anchor_words):
            f.write(w + '\n')

    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f'source\t{source}\n')
        f.write(f'target\t{target}\n')
        f.write(f'shared_vocab\t{len(words)}\n')
        f.write(f'anchor_count\t{len(anchor_words)}\n')
        f.write(f'quantile\t{quantile}\n')
        f.write(f'iterations\t{iters}\n')
        f.write(f'stopwords_removed\t{strip_stopwords}\n')
        f.write(f'min_rank_per_corpus\t{min_rank_per_corpus}\n')
        f.write(f'min_word_length\t{min_word_length}\n')

    print(
        f'{source}->{target}: |V|={len(words)}, anchors={len(anchor_words)}, '
        f'iters={iters}, shifts->{shifts_path}'
    )
    return anchor_words


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pair', 'all-pairs', 'triple-intersection'], default='all-pairs')
    p.add_argument('--source', choices=corpora_default)
    p.add_argument('--target', choices=corpora_default)
    p.add_argument('--corpora', nargs='+', default=corpora_default)
    p.add_argument('--emb-dir', default=emb_dir_default)
    p.add_argument('--out-dir', default=os.path.normpath(os.path.join(script_dir, '..', 'results', 'anchors_word2vec')))
    p.add_argument('--anchor-quantile', type=float, default=0.15)
    p.add_argument('--max-iter', type=int, default=30)
    p.add_argument('--min-anchors', type=int, default=200)
    p.add_argument('--max-anchors', type=int, default=50000)
    p.add_argument('--convergence-tol', type=int, default=0)
    p.add_argument('--no-stopwords', action='store_true')
    p.add_argument('--min-rank-per-corpus', type=int, default=None, metavar='K')
    p.add_argument('--min-word-length', type=int, default=4, metavar='N')
    args = p.parse_args()

    corpora = args.corpora
    if len(set(corpora)) != len(corpora):
        raise SystemExit('--corpora must be unique')
    strip_sw = not args.no_stopwords

    if args.mode == 'pair':
        if args.source is None or args.target is None:
            raise SystemExit('mode=pair requires --source and --target')
        if args.source == args.target:
            raise SystemExit('source and target must differ')
        run_pair(
            args.source, args.target, args.emb_dir, args.out_dir, strip_sw,
            args.min_rank_per_corpus, args.anchor_quantile, args.max_iter,
            args.min_anchors, args.max_anchors, args.convergence_tol, args.min_word_length,
        )
        return

    pairs = []
    for i, a in enumerate(corpora):
        for b in corpora[i + 1:]:
            pairs.append((a, b))

    anchor_sets = []
    for a, b in pairs:
        s = run_pair(
            a, b, args.emb_dir, args.out_dir, strip_sw,
            args.min_rank_per_corpus, args.anchor_quantile, args.max_iter,
            args.min_anchors, args.max_anchors, args.convergence_tol, args.min_word_length,
        )
        anchor_sets.append(s)

    if args.mode == 'triple-intersection' and len(anchor_sets) >= 2:
        inter = set.intersection(*anchor_sets)
        path = os.path.join(args.out_dir, 'anchors_triple_intersection.txt')
        os.makedirs(args.out_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for w in sorted(inter):
                f.write(w + '\n')
        print(f'triple intersection: |anchors|={len(inter)} -> {path}')


if __name__ == '__main__':
    main()
