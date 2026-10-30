import json
from types import SimpleNamespace

import httpx
from openai import BadRequestError

from echosight.config import RetrievalConfig
from echosight.document import Section
from echosight.retrieval import Retriever
from echosight.toc import ToCRouter, number_sections

SECTIONS = [
    Section("Code of Student Conduct", ("Code of Student Conduct",), 1, [(1, "Adopted 1971.")]),
    Section("General Expectations", ("General Expectations",), 1, [(1, "Be civil.")]),
    Section("STUDENT EXPECTATIONS", ("General Expectations", "STUDENT EXPECTATIONS"), 2, [(2, "Rights.")]),
    Section("Standards of Conduct", ("Standards of Conduct",), 7, [(7, "Listed alphabetically.")]),
    Section("DOXING", ("Standards of Conduct", "DOXING"), 9, [(9, "Publishing personal information.")]),
    Section("Alcohol", ("Standards of Conduct", "DOXING", "Alcohol"), 9, [(9, "Sub.")]),
]


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.cfg = SimpleNamespace(model="fake")
        self.name = "fake/model"
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def _reasoning_params(self):
        return {}

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])


def test_outline_numbers():
    assert number_sections(SECTIONS) == ["0", "1", "1.1", "2", "2.1", "2.1.1"]


def test_router_constrains_output_to_toc_numbers():
    llm = FakeLLM([{"sections": ["2.1", "1"]}])
    router = ToCRouter(SECTIONS, llm, k=3)
    assert router.route("What is doxing?") == [4, 1]
    request = llm.requests[0]
    assert request["response_format"]["json_schema"]["schema"]["properties"]["sections"]["items"]["enum"] == \
        ["0", "1", "1.1", "2", "2.1", "2.1.1"]
    assert "2.1 Doxing" in request["messages"][0]["content"]
    router.route("What is doxing?")          # cached: no second request
    assert len(llm.requests) == 1


def test_rejected_generation_counts_as_invalid():
    response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    error = BadRequestError("json_validate_failed", response=response, body=None)
    router = ToCRouter(SECTIONS, FakeLLM([error]), k=3)
    assert router.route("Wi-Fi password?") == [] and router.invalid == 1


def test_routed_parent_covers_its_subsections():
    retriever = Retriever.__new__(Retriever)
    retriever.router = SimpleNamespace(sections=SECTIONS)
    retriever.chunks = [SimpleNamespace(path=s.path) for s in SECTIONS]
    # Routing "Standards of Conduct" brings in DOXING and its subsection, ordered by the fused ranking.
    assert retriever._toc_ranking([3], order=[5, 4]) == [5, 4, 3]


def test_stair_is_off_by_default():
    assert RetrievalConfig().toc_router is None
