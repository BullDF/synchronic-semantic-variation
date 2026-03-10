import json

data_dir = '../data'
src = f'{data_dir}/processed/arxiv_cs_2010_2020.jsonl'
out = f'{data_dir}/processed/arxiv_cs_2010_2011.jsonl'

kept = 0
with open(src) as f, open(out, 'w') as g:
    for line in f:
        p = json.loads(line)
        if p['year'] not in (2010, 2011):
            continue
        g.write(line)
        kept += 1

print(f'Done, kept {kept}')
