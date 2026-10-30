"""How long would a screen-reader user listen before reaching each answer without Echo-Sight?

Two strategies over the tagged PDF (its H1-H4 tags match our 78 parsed headings):
  linear   - listen from the top until the evidence quote
  headings - jump heading by heading (hearing each title plus "heading level N"), stop at the
             right section, then listen within it until the evidence. This assumes the listener
             recognises the right heading immediately, so it is a generous baseline.

    python eval/listening_baseline.py --split test
"""

import argparse
import json

from common import RESULTS, load_questions, match_key, median, percentile

from echosight.config import PDF_PATH
from echosight.document import parse_pdf

RATES = (180, 360)          # default speech rate; fast rate used by experienced screen-reader users
ANNOUNCEMENT_WORDS = 4      # "heading level three"


def words(text: str) -> int:
    return len(text.split())


def locate(sections, quote: str):
    """(section index, paragraph index, words before the quote inside that paragraph) or None."""
    key = match_key(quote)
    for s, section in enumerate(sections):
        for p, (_, text) in enumerate(section.paragraphs):
            position = match_key(text).find(key)
            if position >= 0:
                return s, p, words(match_key(text)[:position])
    return None


def listening_words(sections, location) -> tuple[int, int]:
    s, p, offset = location
    before_section = sum(words(sec.title) + sum(words(t) for _, t in sec.paragraphs) for sec in sections[:s])
    in_section = sum(words(t) for _, t in sections[s].paragraphs[:p]) + offset
    linear = before_section + words(sections[s].title) + in_section
    headings = sum(words(sec.title) + ANNOUNCEMENT_WORDS for sec in sections[: s + 1]) + in_section
    return linear, headings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    sections = parse_pdf(PDF_PATH)
    total_words = sum(words(s.title) + sum(words(t) for _, t in s.paragraphs) for s in sections)

    linear, headings = [], []
    for item in load_questions(args.split, answerable=True):
        found = [loc for quote in item["evidence"][0] if (loc := locate(sections, quote))]
        lin, head = min(listening_words(sections, loc) for loc in found)
        linear.append(lin)
        headings.append(head)

    summary = {"split": args.split, "questions": len(linear), "document_words": total_words}
    lines = [f"Document: {total_words:,} words, 78 headings. {len(linear)} answerable {args.split} questions.", "",
             "| Strategy | Words before answer (median) | " +
             " | ".join(f"Time at {r} wpm (median / p90)" for r in RATES) + " |",
             "|---|---|" + "---|" * len(RATES)]
    for name, values in (("Linear listening", linear), ("Heading navigation", headings)):
        cells = []
        for rate in RATES:
            med, p90 = median(values) / rate * 60, percentile(values, 90) / rate * 60
            summary[f"{name}_{rate}"] = {"median_s": med, "p90_s": p90}
            cells.append(f"{med / 60:.1f} min / {p90 / 60:.1f} min" if med >= 60 else f"{med:.0f} s / {p90:.0f} s")
        summary[f"{name}_median_words"] = median(values)
        lines.append(f"| {name} | {median(values):,.0f} | " + " | ".join(cells) + " |")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"listening_{args.split}.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / f"listening_{args.split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
