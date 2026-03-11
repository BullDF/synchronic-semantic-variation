#!/bin/bash
python3 train_word2vec.py arxiv --sample 1e-3
python3 train_word2vec.py yelp
python3 train_word2vec.py ciao
