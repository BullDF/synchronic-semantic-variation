import json
import os

data_dir = '../data'
src = f'{data_dir}/raw/arxiv-metadata-oai-snapshot.json'
out = f'{data_dir}/processed/arxiv_cs_2010_2020.jsonl'
os.makedirs(f'{data_dir}/processed', exist_ok=True)

kept = 0
with open(src) as f, open(out, 'w') as g:
    for i, line in enumerate(f):
        p = json.loads(line)

        if not any(c.startswith('cs.') for c in p.get('categories', '').split()):
            continue

        # date looks like "Mon, 1 Jan 2020 00:00:00 GMT"
        year = int(p['versions'][0]['created'].split()[-3])
        if not (2010 <= year <= 2020):
            continue

        g.write(json.dumps({
            'id': p['id'],
            'categories': p['categories'],
            'year': year,
            'abstract': p['abstract'].strip(),
        }) + '\n')
        kept += 1

        if i % 100000 == 0:
            print(f'{i} processed, {kept} kept')

print(f'Done, kept {kept}')
