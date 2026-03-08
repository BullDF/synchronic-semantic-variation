import json
import os

data_dir = '../data'
src = f'{data_dir}/raw/yelp_dataset/yelp_academic_dataset_review.json'
out = f'{data_dir}/processed/yelp_reviews_2010_2020.jsonl'
os.makedirs(f'{data_dir}/processed', exist_ok=True)

kept = 0
with open(src) as f, open(out, 'w') as g:
    for i, line in enumerate(f):
        p = json.loads(line)

        year = int(p['date'][:4])
        if not (2010 <= year <= 2020):
            continue

        g.write(json.dumps({
            'id': p['review_id'],
            'year': year,
            'text': p['text'].strip(),
        }) + '\n')
        kept += 1

        if i % 100000 == 0:
            print(f'{i} processed, {kept} kept')

print(f'Done, kept {kept}')
