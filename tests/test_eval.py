import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from common import contains_quote, load_questions, match_key, ranking_metrics  # noqa: E402

from echosight.config import PDF_PATH  # noqa: E402
from echosight.document import parse_pdf  # noqa: E402


def test_quote_matching_is_typography_insensitive():
    assert contains_quote("the student’s record and the ﬁle", "student's record and the file")


def test_quote_cut_by_a_chunk_boundary_gets_credit():
    quote = "If a guest violates university policy, the host may be held accountable"
    assert contains_quote("... for specific regulations. If a guest violates university policy, the host", quote)
    assert contains_quote("violates university policy, the host may be held accountable for actions.", quote)
    assert not contains_quote("A guest is someone you invite.", quote)


def test_shared_generic_opening_elsewhere_gets_no_credit():
    quote = "the university reserves the right to notify a student's parents"
    unrelated = "In such cases the university reserves the right to delay the awarding of a degree."
    assert not contains_quote(unrelated, quote)


def test_ranking_metrics():
    item = {"evidence": [["five business days"], ["Appeals Board"]]}
    texts = ["irrelevant", "appeal within five business days", "heard by the Appeals Board", "other"]
    m = ranking_metrics(texts, item)
    assert m["rank"] == 2 and m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["mrr"] == 0.5 and m["recall@4"] == 1.0


def test_every_evidence_quote_is_verbatim_in_the_pdf():
    document = match_key(" ".join(s.text for s in parse_pdf(PDF_PATH)))
    questions = load_questions()
    assert len(questions) == 186 and {q["split"] for q in questions} == {"dev", "test"}
    for q in questions:
        for group in q.get("evidence", []):
            for quote in group:
                assert match_key(quote) in document, (q["id"], quote)
