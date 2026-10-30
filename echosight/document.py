"""Parse the source PDF into a tree of sections using the font styles of its headings."""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

_LIST_ITEM = re.compile(r"^(\d+ ?\.|[a-z]\.|•)\s")
_TERMINAL = (".", ":", ";", "?", "!", ")", '"', "”")
HEADER_FRACTION = 0.06   # running header / page number band at the top of each page


@dataclass
class Section:
    title: str
    path: tuple[str, ...]                       # heading titles from the top level down to this one
    page: int
    paragraphs: list[tuple[int, str]] = field(default_factory=list)   # (page, text)

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.paragraphs)


def normalize(text: str) -> str:
    """Undo PDF typography (ligatures, non-breaking spaces) and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(\d+) \.", r"\1.", text)


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}


def spoken_title(title: str) -> str:
    """'FAILURE TO COMPLY' -> 'Failure to Comply', so speech engines and screen readers don't spell it out."""
    if not title.isupper():
        return title
    words = title.lower().split()
    return " ".join(w if i and w in _SMALL_WORDS else w[:1].upper() + w[1:] for i, w in enumerate(words))


def heading_level(span: dict) -> int | None:
    """0 = document title, 1 = part, 2 = section (all caps), 3 = subsection (bold italic)."""
    size, font = round(span["size"]), span["font"]
    if size >= 14:
        return 0
    if "Bold" not in font:
        return None
    if size >= 10:
        return 1
    if size == 9:
        return 3 if "Ital" in font else 2
    return None


def _join(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if out and not out.endswith(("-", "/")):
            out += " "
        out += line
    return out


def _continues(previous: str, text: str) -> bool:
    """True when a paragraph was cut in two by a page break."""
    if _LIST_ITEM.match(text):
        return False
    return text[:1].islower() or not previous.endswith(_TERMINAL)


def _runs(page) -> list[tuple[int | None, str]]:
    """Group the lines of a page into (heading level or None, text) runs."""
    top = page.rect.height * HEADER_FRACTION
    runs = []
    for block in page.get_text("dict")["blocks"]:
        current_level, lines = "start", []
        for line in block.get("lines", []):
            if not line["spans"] or line["bbox"][3] < top:
                continue
            level = heading_level(line["spans"][0])
            text = "".join(span["text"] for span in line["spans"])
            if level != current_level and lines:
                runs.append((current_level, _join(lines)))
                lines = []
            current_level = level
            lines.append(text)
        if lines:
            runs.append((current_level, _join(lines)))
    return runs


def parse_pdf(path: str | Path) -> list[Section]:
    sections: list[Section] = []
    open_headings: dict[int, str] = {}
    current = None
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, 1):
            first_on_page = True
            for level, raw in _runs(page):
                text = normalize(raw)
                if not text:
                    continue
                if level == 0:
                    open_headings = {}
                    current = Section(text, (text,), page_no)
                    sections.append(current)
                elif level is not None:
                    open_headings = {lvl: t for lvl, t in open_headings.items() if lvl < level}
                    open_headings[level] = text
                    path = tuple(open_headings[lvl] for lvl in sorted(open_headings))
                    current = Section(text, path, page_no)
                    sections.append(current)
                else:
                    if current is None:
                        current = Section("Introduction", ("Introduction",), page_no)
                        sections.append(current)
                    paras = current.paragraphs
                    if first_on_page and paras and _continues(paras[-1][1], text):
                        paras[-1] = (paras[-1][0], paras[-1][1] + " " + text)
                    else:
                        paras.append((page_no, text))
                first_on_page = False
    return [s for s in sections if s.paragraphs]
