import os
import csv
import json
import time
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
default_input = os.path.normpath(os.path.join(script_dir, '..', 'reddit_filtered'))
default_out = os.path.normpath(os.path.join(script_dir, '..', 'data'))

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', nargs='?', default=default_input)
parser.add_argument('--out-dir', default=default_out)
parser.add_argument('--no-header', action='store_true', help='CSV has no header row')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

open_files = {}
counts = {}

def get_handle(year, month):
    key = (year, month)
    if key not in open_files:
        out_name = f'reddit_{year}_{month:02d}.jsonl'
        out_path = os.path.join(args.out_dir, out_name)
        open_files[key] = open(out_path, 'w', encoding='utf-8')
        counts[key] = 0
    return open_files[key], key

for name in sorted(os.listdir(args.input_dir)):
    if not name.lower().endswith('.csv'):
        continue
    path = os.path.join(args.input_dir, name)
    with open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            continue
        skip_first = False
        if not args.no_header and len(first) >= 3 and first[0].strip().lower() == 'id':
            skip_first = True
        if not skip_first:
            row = first
            if len(row) >= 3:
                id_, time_str, content = row[0].strip(), row[1].strip(), row[2].strip()
                if content:
                    try:
                        ts = int(time_str)
                        t = time.gmtime(ts)
                        year, month = t.tm_year, t.tm_mon
                        fh, key = get_handle(year, month)
                        fh.write(json.dumps({'id': id_, 'time': ts, 'text': content, 'year': year, 'month': month}) + '\n')
                        counts[key] += 1
                    except (ValueError, OSError):
                        pass
        for row in reader:
            if len(row) < 3:
                continue
            id_, time_str, content = row[0].strip(), row[1].strip(), row[2].strip()
            if not content:
                continue
            try:
                ts = int(time_str)
                t = time.gmtime(ts)
                year, month = t.tm_year, t.tm_mon
                fh, key = get_handle(year, month)
                fh.write(json.dumps({'id': id_, 'time': ts, 'text': content, 'year': year, 'month': month}) + '\n')
                counts[key] += 1
            except (ValueError, OSError):
                pass

for fh in open_files.values():
    fh.close()

for (year, month), n in sorted(counts.items()):
    print(f'Wrote {args.out_dir}/reddit_{year}_{month:02d}.jsonl ({n} documents)')
