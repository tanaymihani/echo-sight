from types import SimpleNamespace

from echosight.chunking import Chunk
from echosight.document import spoken_title
from echosight.llm import Completion
from echosight.prompts import REFUSAL
from echosight.rag import EchoSight, build_messages, cited_numbers, is_refusal, strip_citations
from echosight.retrieval import BM25Index, Hit, rrf, tokenize


def chunk(i, section="DOXING", text="Publishing personal identifying information."):
    return Chunk(f"c{i}", text, section, ("Standards of Conduct", section), 9)


def test_rrf_rewards_agreement():
    fused = [doc for doc, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[:2] == [1, 3]
    assert set(fused) == {1, 2, 3, 4}


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("Filing the appeals") == tokenize("file an appeal")
    assert "the" not in tokenize("the hearing")


def test_bm25_matches_keywords():
    chunks = (chunk(0, "SMOKING", "Smoking of any tobacco products is prohibited."),
              chunk(1, "GAMBLING", "A game of chance for money."))
    assert BM25Index(chunks).search("Can I vape or smoke tobacco?", 1)[0][0] == 0


def test_citation_parsing():
    text = "Yes, within five business days [2]. Appeals are written [1, 3].[4]"
    assert cited_numbers(text) == [2, 1, 3, 4]
    assert strip_citations(text) == "Yes, within five business days. Appeals are written."


def test_refusal_detection_tolerates_curly_apostrophes():
    assert is_refusal("I couldn’t find that in the Code of Student Conduct. Try the Housing guide.")
    assert not is_refusal("Yes, you can appeal within five business days.")


def test_spoken_title():
    assert spoken_title("FAILURE TO COMPLY") == "Failure to Comply"
    assert spoken_title("THE ROLE OF THE OFFICE") == "The Role of the Office"
    assert spoken_title("Interim Suspension") == "Interim Suspension"


def test_prompt_numbers_passages_with_location():
    user = build_messages("What is doxing?", [Hit(chunk(0), 0.9)])[1]["content"]
    assert "[1] Standards of Conduct > DOXING (page 9)" in user
    assert user.endswith("Question: What is doxing?")


class FakeRetriever:
    def __init__(self, score):
        self.cfg = SimpleNamespace(min_score=0.1)
        self.reranker = object()
        self.last_timing = {"retrieval_s": 0.01}
        self.score = score

    def search(self, query):
        return [Hit(chunk(0), self.score), Hit(chunk(1, "HARASSMENT"), self.score / 2)]


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def chat(self, messages, on_text=None, **_):
        self.calls += 1
        if on_text:
            on_text(self.text)
        return Completion(self.text, 0.1, 0.5, 100, 20)


def test_low_confidence_retrieval_refuses_without_calling_llm():
    llm = FakeLLM("should not be used")
    answer = EchoSight(FakeRetriever(score=0.01), llm).ask("What is the Wi-Fi password?")
    assert answer.refused and answer.gated and answer.text == REFUSAL
    assert llm.calls == 0


def test_answer_maps_citations_to_sources():
    llm = FakeLLM("Doxing is publishing someone's personal information [1].")
    answer = EchoSight(FakeRetriever(score=0.9), llm).ask("What is doxing?")
    assert not answer.refused and [h.chunk.id for h in answer.cited] == ["c0"]
    assert answer.spoken == "Doxing is publishing someone's personal information. Source: Doxing, page 9."


class Done:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def result(self):
        if self.error:
            raise self.error
        return self.value


def gated_retriever(score, routed):
    retriever = FakeRetriever(score)
    retriever.cfg = SimpleNamespace(min_score=None, toc_gate_score=0.05)
    retriever.last_routing = routed
    return retriever


def test_toc_gate_needs_both_empty_route_and_low_score():
    llm = FakeLLM("Doxing is defined [1].")
    assert EchoSight(gated_retriever(0.001, Done([])), llm).ask("Wi-Fi password?").gated
    assert not EchoSight(gated_retriever(0.001, Done([4])), llm).ask("Doxing?").gated
    assert not EchoSight(gated_retriever(0.9, Done([])), llm).ask("Doxing?").gated


def test_router_failure_never_refuses():
    llm = FakeLLM("Doxing is defined [1].")
    answer = EchoSight(gated_retriever(0.001, Done(error=TimeoutError())), llm).ask("Doxing?")
    assert not answer.gated and llm.calls == 1


def test_source_line_names_each_section_once():
    llm = FakeLLM("It is doxing [1]. Doxing is defined there [1, 2].")
    retriever = FakeRetriever(score=0.9)
    retriever.search = lambda query: [Hit(chunk(0), 0.9), Hit(chunk(1), 0.8), Hit(chunk(2, "HARASSMENT"), 0.7)]
    answer = EchoSight(retriever, llm).ask("Is that doxing?")
    assert answer.source_line == "Source: Doxing, page 9."
