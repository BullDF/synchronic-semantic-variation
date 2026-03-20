#!/bin/bash
set -e

for method in word2vec svd; do
    echo "=== $method ==="
    python3 align_procrustes.py $method arxiv yelp
    python3 align_procrustes.py $method arxiv ciao
    python3 align_procrustes.py $method arxiv reddit
    python3 align_procrustes.py $method yelp ciao
    python3 align_procrustes.py $method yelp reddit
    python3 align_procrustes.py $method ciao reddit
done
