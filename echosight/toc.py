"""STAIR-style table-of-contents routing (Kumar et al., 2026, "STAIR: STructure Aware Information Retriever").

STAIR gives an LLM the document's table of contents together with the query and has it generate the
section that answers it, with decoding restricted to valid ToC entries. The paper fine-tunes Mistral-7B;
this is the zero-shot variant: an instruction-following LLM picks up to k section numbers, and a strict
JSON schema whose enum lists every ToC number rejects anything that is not a real section.
"""

import json
import threading
import time
from pathlib import Path

from openai import BadRequestError

from echosight.document import Section, spoken_title
from echosight.llm import LLM

PROMPT = """### Document: Northeastern University Code of Student Conduct
Pick the sections from the table of contents below that can answer the user's question. Reply with up to \
{k} section numbers, best first. Only use numbers from the table of contents.

### Table of Contents
{toc}"""


def number_sections(sections: list[Section]) -> list[str]:
    """Outline numbers like "4", "4.5", "4.5.3" from each section's depth; the untitled preamble is "0"."""
    numbers, counters = [], []
    for section in sections:
        depth = len(section.path)
        if section is sections[0] and depth == 1 and section.title == sections[0].path[0] and not numbers:
            numbers.append("0")
            continue
        counters = counters[:depth] + [0] * (depth - len(counters))
        counters[depth - 1] += 1
        numbers.append(".".join(str(c) for c in counters))
    return numbers


class ToCRouter:
    def __init__(self, sections: list[Section], llm: LLM, k: int = 3, cache_path: Path | None = None):
        self.sections, self.llm, self.k = sections, llm, k
        self.ids = number_sections(sections)
        self.index = {n: i for i, n in enumerate(self.ids)}
        lines = []
        for number, section in zip(self.ids, sections):
            title = spoken_title(section.title) + (" (introduction)" if number == "0" else "")
            lines.append(f"{number} {title}")
        self.toc = "\n".join(lines)
        self.system = PROMPT.format(k=k, toc=self.toc)
        self.schema = {
            "type": "object",
            "properties": {"sections": {"type": "array", "items": {"type": "string", "enum": self.ids}}},
            "required": ["sections"],
            "additionalProperties": False,
        }
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
        self.invalid = 0           # outputs rejected because they named a section that doesn't exist
        self._lock = threading.Lock()

    def route(self, query: str) -> list[int]:
        """Positions in `sections` of the routed sections, best first (empty if routing failed)."""
        key = f"{self.llm.name}|{self.k}|{query}"
        if key not in self.cache:
            start, invalid = time.perf_counter(), self.invalid
            numbers = self._ask(query)
            self.cache[key] = {"sections": numbers, "latency_s": time.perf_counter() - start,
                               "invalid": self.invalid > invalid}
            if self.cache_path:
                with self._lock:
                    self.cache_path.write_text(json.dumps(self.cache, indent=1))
        self.last_latency = self.cache[key]["latency_s"]
        return [self.index[n] for n in self.cache[key]["sections"] if n in self.index][: self.k]

    def _ask(self, query: str) -> list[str]:
        request = dict(
            model=self.llm.cfg.model, temperature=0, max_tokens=600,
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": f"### Query: {query}"}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "toc_sections", "strict": True, "schema": self.schema}},
            extra_body=self.llm._reasoning_params(),
        )
        try:
            reply = self.llm.client.chat.completions.create(**request)
        except BadRequestError as error:
            if "json_validate_failed" not in str(error) and "schema" not in str(error):
                raise
            self.invalid += 1
            return []
        try:
            numbers = json.loads(reply.choices[0].message.content or "{}").get("sections", [])
        except json.JSONDecodeError:
            self.invalid += 1
            return []
        if any(n not in self.index for n in numbers):
            self.invalid += 1
        return list(dict.fromkeys(n for n in numbers if n in self.index))
