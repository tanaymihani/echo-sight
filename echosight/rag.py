"""Retrieval-augmented answering: retrieve, gate on retrieval confidence, generate with citations."""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from echosight.llm import LLM
from echosight.prompts import CONDENSE_PROMPT, PASSAGE_TEMPLATE, REFUSAL, SYSTEM_PROMPT, USER_TEMPLATE
from echosight.retrieval import Hit, Retriever

_CITATION = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def cited_numbers(text: str) -> list[int]:
    numbers = []
    for group in _CITATION.findall(text):
        numbers += [int(n) for n in re.split(r"\s*,\s*", group)]
    return list(dict.fromkeys(numbers))


def is_refusal(text: str) -> bool:
    return REFUSAL.lower().rstrip(".") in text.replace("’", "'").lower()


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    passages = "\n\n".join(
        PASSAGE_TEMPLATE.format(n=i, breadcrumb=h.chunk.breadcrumb, page=h.chunk.page, text=h.chunk.text)
        for i, h in enumerate(hits, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(passages=passages, question=question)}]


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[Hit] = field(default_factory=list)
    refused: bool = False
    gated: bool = False           # refused by the retrieval-confidence gate, without calling the LLM
    timings: dict = field(default_factory=dict)
    tokens: int = 0

    @property
    def source_line(self) -> str:
        """Up to two distinct cited sections, e.g. 'Source: Medical Amnesty, page 8.'"""
        first: dict[str, str] = {}
        for hit in self.cited:
            first.setdefault(hit.chunk.section, hit.citation)
        sources = "; ".join(list(first.values())[:2])
        return f"Source: {sources}." if sources else ""

    @property
    def spoken(self) -> str:
        return f"{strip_citations(self.text)} {self.source_line}".strip()


class EchoSight:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        turns = "\n".join(f"User: {q}\nAssistant: {strip_citations(a)}" for q, a in history[-3:])
        prompt = CONDENSE_PROMPT.format(history=turns, question=question)
        return self.llm.chat([{"role": "user", "content": prompt}], max_tokens=400, temperature=0).text

    def _out_of_scope(self, hits: list[Hit], timings: dict) -> bool:
        """Refuse before generation when retrieval is not confident (see RetrievalConfig)."""
        cfg, top = self.retriever.cfg, (hits[0].score if hits else 0.0)
        if not self.retriever.reranker:
            return False
        if cfg.min_score is not None and top < cfg.min_score:
            return True
        routing, gate = getattr(self.retriever, "last_routing", None), getattr(cfg, "toc_gate_score", None)
        if routing is not None and gate is not None and top < gate:
            wait = time.perf_counter()
            try:
                routed = routing.result()      # only low-confidence queries wait for the ToC router
            except Exception:
                return False                   # a router outage must never cause a refusal
            finally:
                timings["toc_wait_s"] = time.perf_counter() - wait
            return not routed
        return False

    def ask(self, question: str, on_text: Callable[[str], None] | None = None,
            history: list[tuple[str, str]] | None = None) -> Answer:
        start = time.perf_counter()
        query = self.condense(question, history) if history else question
        hits = self.retriever.search(query)
        timings = {"retrieval_s": self.retriever.last_timing["retrieval_s"]}
        if self._out_of_scope(hits, timings):
            if on_text:
                on_text(REFUSAL)
            timings["total_s"] = time.perf_counter() - start
            return Answer(question, REFUSAL, hits, refused=True, gated=True, timings=timings)

        completion = self.llm.chat(build_messages(query, hits), on_text=on_text)
        cited = [hits[n - 1] for n in cited_numbers(completion.text) if 0 < n <= len(hits)]
        timings.update(ttft_s=completion.ttft_s, llm_s=completion.total_s,
                       total_s=time.perf_counter() - start)
        return Answer(question, completion.text, hits, cited, refused=is_refusal(completion.text),
                      timings=timings, tokens=completion.prompt_tokens + completion.completion_tokens)
