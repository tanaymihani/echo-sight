from echosight.voice import SentenceStream


def stream(deltas):
    out = []
    s = SentenceStream(out.append)
    for delta in deltas:
        s.feed(delta)
    s.flush()
    return out


def test_sentences_are_emitted_as_tokens_arrive():
    out = []
    s = SentenceStream(out.append)
    for delta in ["Yes, you ", "can appeal [1]. ", "You have five"]:
        s.feed(delta)
    assert out == ["Yes, you can appeal."]
    s.feed(" business days [2].")
    s.flush()
    assert out == ["Yes, you can appeal.", "You have five business days."]


def test_citation_after_the_period_is_dropped():
    assert stream(["Appeals are written. ", "[", "3", "] They are reviewed."]) == [
        "Appeals are written.", "They are reviewed."]


def test_abbreviations_do_not_split_sentences():
    assert stream(["Use alternate means (e.g., mediation) if all agree."]) == [
        "Use alternate means (e.g., mediation) if all agree."]
