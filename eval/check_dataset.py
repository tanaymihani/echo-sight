"""Validate the question set: every evidence quote must appear verbatim in the PDF.

Also reports how much of the evidence each chunking keeps findable (v1 silently dropped text),
and assigns a stratified dev/test split the first time it runs.

    python eval/check_dataset.py
"""

import json
import random
from collections import Counter, defaultdict

from common import DATA, covers, load_questions, match_key

from echosight.config import PDF_PATH
from echosight.document import parse_pdf
from echosight.index import load_chunks
from eval import v1_baseline


def assign_splits(items: list[dict]) -> bool:
    if all("split" in q for q in items):
        return False
    by_type = defaultdict(list)
    for q in items:
        by_type[q["type"]].append(q)
    rng = random.Random(13)
    for group in by_type.values():
        rng.shuffle(group)
        for i, q in enumerate(group):
            q["split"] = "dev" if i % 2 == 0 else "test"
    DATA.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in items))
    return True


def main():
    items = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    ids = [q["id"] for q in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    if assign_splits(items):
        print("Assigned dev/test splits")
    items = load_questions()

    document = match_key(" ".join(s.text for s in parse_pdf(PDF_PATH)))
    v1_text = " ".join(c["text"] for c in v1_baseline.chunks())
    missing = [(q["id"], quote) for q in items for g in q.get("evidence", []) for quote in g
               if match_key(quote) not in document]
    for qid, quote in missing:
        print(f"NOT IN PDF  {qid}: {quote!r}")

    answerable = [q for q in items if q["type"] != "out_of_scope"]
    pipelines = {
        "v1 chunks (500/80, regex sections)": [c["text"] for c in v1_baseline.chunks()],
        "v2 chunks (800, heading-aware)": [c.text for c in load_chunks(800)],
    }
    for name, texts in pipelines.items():
        groups = [g for q in answerable for g in q["evidence"]]
        found = sum(any(covers(t, g) for t in texts) for g in groups)
        print(f"{name}: {found}/{len(groups)} evidence groups findable in some chunk")
    lost = [q["id"] for q in answerable if not all(covers(v1_text, g) for g in q["evidence"])]
    print(f"Questions whose evidence v1 dropped from its index entirely: {len(lost)} {lost}")

    print(Counter((q["split"], q["type"]) for q in items))
    print(f"{len(answerable)} answerable, {len(items) - len(answerable)} out of scope; "
          f"{len(missing)} quotes not found")


if __name__ == "__main__":
    main()
