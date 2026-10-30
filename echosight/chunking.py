"""Split sections into retrieval chunks that never cross a heading boundary."""

import re
from dataclasses import dataclass

from echosight.document import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    section: str
    path: tuple[str, ...]
    page: int

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)

    def indexed_text(self, headers: bool = True) -> str:
        """What the retrievers see: the heading path gives short chunks the context they lack."""
        return f"{self.breadcrumb}\n{self.text}" if headers else self.text


def _pack(pieces: list[str], max_chars: int, sep: str) -> list[str]:
    out, current = [], ""
    for piece in pieces:
        if current and len(current) + len(sep) + len(piece) > max_chars:
            out.append(current)
            current = piece
        else:
            current = f"{current}{sep}{piece}" if current else piece
    if current:
        out.append(current)
    return out


def chunk_sections(sections: list[Section], max_chars: int = 800) -> list[Chunk]:
    chunks = []
    for section in sections:
        units = []   # (page, text) pieces no longer than max_chars where possible
        for page, para in section.paragraphs:
            if len(para) <= max_chars:
                units.append((page, para))
            else:
                units += [(page, s) for s in _pack(_SENTENCE_END.split(para), max_chars, " ")]
        start = 0
        while start < len(units):
            end, size = start + 1, len(units[start][1])
            while end < len(units) and size + 1 + len(units[end][1]) <= max_chars:
                size += 1 + len(units[end][1])
                end += 1
            text = "\n".join(t for _, t in units[start:end])
            chunks.append(Chunk(f"c{len(chunks):04d}", text, section.title, section.path, units[start][0]))
            start = end
    return chunks
