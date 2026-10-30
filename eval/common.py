"""Shared eval helpers: the labelled question set, evidence matching, and ranking metrics.

A retrieved passage counts as relevant when it contains one of the question's evidence quotes
(verbatim from the PDF). This makes every metric independent of how a pipeline chunks the
document, so the v1 baseline and every v2 configuration are scored the same way.
"""

import json
import math
import re
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).parent / "data" / "questions.jsonl"
RESULTS = Path(__file__).parent / "results"
KS = (1, 3, 5, 10)


def load_questions(split: str | None = None, answerable: bool | None = None) -> list[dict]:
    items = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if split and split != "all":
        items = [q for q in items if q["split"] == split]
    if answerable is not None:
        items = [q for q in items if (q["type"] != "out_of_scope") == answerable]
    return items


def match_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_quote(text: str, quote: str) -> bool:
    """Full containment, or, for quotes of 40+ characters cut in two by a chunk boundary, at least half
    of the quote where the cut is: the chunk ends with the quote's beginning or starts with its end.
    (Matching a half-quote anywhere would credit unrelated passages that share a generic opening.)"""
    text, quote = match_key(text), match_key(quote)
    if quote in text:
        return True
    if len(quote) < 40:
        return False
    need = max(25, math.ceil(len(quote) / 2))
    return any(text.endswith(quote[:n]) or text.startswith(quote[-n:]) for n in range(need, len(quote)))


def covers(text: str, group: list[str]) -> bool:
    return any(contains_quote(text, quote) for quote in group)


def is_relevant(text: str, item: dict) -> bool:
    return any(covers(text, group) for group in item["evidence"])


def ranking_metrics(texts: list[str], item: dict) -> dict:
    """Hit@k, reciprocal rank and evidence recall for one ranked list of passage texts."""
    relevant = [is_relevant(t, item) for t in texts]
    rank = relevant.index(True) + 1 if True in relevant else None
    out = {f"hit@{k}": float(rank is not None and rank <= k) for k in KS}
    out["mrr"] = 1.0 / rank if rank and rank <= 10 else 0.0
    for k in (4, 8):
        found = sum(any(covers(t, g) for t in texts[:k]) for g in item["evidence"])
        out[f"recall@{k}"] = found / len(item["evidence"])
    out["rank"] = rank
    return out


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(values, p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * p / 100
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def median(values) -> float:
    return statistics.median(values) if values else float("nan")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """columns: (key, header). Floats in [0, 1] print as percentages unless the key is 'mrr'."""
    def fmt(key, value):
        if isinstance(value, float):
            if key in ("mrr",) or key.endswith("_s") or key.endswith("_ms"):
                return f"{value:.3f}" if key == "mrr" else f"{value:.2f}"
            return f"{value:.1%}"
        return str(value)
    lines = ["| " + " | ".join(h for _, h in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(k, row.get(k, "")) for k, _ in columns) + " |" for row in rows]
    return "\n".join(lines)
