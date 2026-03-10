import json
import re
import os
import argparse
from gensim.models import Word2Vec

data_dir = '../data'

corpora = {
    'arxiv': (f'{data_dir}/processed/arxiv_cs_2010_2011.jsonl', 'abstract'),
    'yelp':  (f'{data_dir}/processed/yelp_reviews_2010_2011.jsonl', 'text'),
    'ciao':  (f'{data_dir}/processed/ciao_reviews_2010_2020.jsonl', 'text'),
}

parser = argparse.ArgumentParser()
parser.add_argument('corpus', choices=corpora.keys())
args = parser.parse_args()

path, field = corpora[args.corpus]
out_dir = '../embeddings/word2vec'
os.makedirs(out_dir, exist_ok=True)

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())

print(f'Loading {args.corpus}...')
sentences = []
with open(path) as f:
    for line in f:
        r = json.loads(line)
        tokens = tokenize(r[field])
        if tokens:
            sentences.append(tokens)
print(f'Loaded {len(sentences)} documents')

model = Word2Vec(
    sentences,
    vector_size=300,
    window=4,
    min_count=5,
    sample=1e-5,
    negative=5,
    ns_exponent=0.75,
    workers=4,
    epochs=5,
    seed=2611,
)

out_path = f'{out_dir}/{args.corpus}.model'
model.save(out_path)
print(f'Saved to {out_path} (vocab size: {len(model.wv)})')
