import json
import os

data_dir = '../data'
src = f'{data_dir}/raw/Ciao/dataset/rating.txt'
out = f'{data_dir}/processed/ciao_reviews_all.jsonl'
os.makedirs(f'{data_dir}/processed', exist_ok=True)

kept = 0
with open(src, encoding='utf-8', errors='ignore') as f, open(out, 'w') as g:
    for i, line in enumerate(f):
        parts = line.strip().split('::::')
        if len(parts) < 7:
            continue

        text = parts[6].strip()
        if not text:
            continue

        date = parts[5].strip()
        try:
            year = int(date.split('.')[-1])
        except:
            year = None

        g.write(json.dumps({
            'user_id': parts[0].strip(),
            'category': parts[2].strip(),
            'year': year,
            'text': text,
        }) + '\n')
        kept += 1

        if i % 100000 == 0:
            print(f'{i} processed, {kept} kept')

print(f'Done, kept {kept}')
