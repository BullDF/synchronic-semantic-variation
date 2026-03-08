"""
Filter the arXiv metadata snapshot to CS abstracts from 2010-2020.

Input:  data/raw/arxiv-metadata-oai-snapshot.json  (one JSON object per line)
Output: data/processed/arxiv_cs_2010_2020.jsonl     (id, categories, year, abstract)

Usage:
    python scripts/filter_arxiv.py
"""

import json
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "arxiv-metadata-oai-snapshot.json"
OUT = Path(__file__).parent.parent / "data" / "processed" / "arxiv_cs_2010_2020.jsonl"

START_YEAR = 2010
END_YEAR = 2020


def parse_year(paper: dict) -> int | None:
    """Extract submission year from the first version's created date."""
    try:
        # Format: "Mon, 1 Jan 2020 00:00:00 GMT"
        created = paper["versions"][0]["created"]
        return int(created.split()[-2])  # second-to-last token is the year
    except (KeyError, IndexError, ValueError):
        return None


def is_cs(paper: dict) -> bool:
    return "cs." in paper.get("categories", "")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    with open(RAW) as fin, open(OUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            paper = json.loads(line)

            if not is_cs(paper):
                continue

            year = parse_year(paper)
            if year is None or not (START_YEAR <= year <= END_YEAR):
                continue

            record = {
                "id": paper["id"],
                "categories": paper["categories"],
                "year": year,
                "abstract": paper["abstract"].strip(),
            }
            fout.write(json.dumps(record) + "\n")
            kept += 1

            if total % 100_000 == 0:
                print(f"  Processed {total:,} papers, kept {kept:,} ...")

    print(f"Done. Kept {kept:,} / {total:,} papers -> {OUT}")


if __name__ == "__main__":
    main()
