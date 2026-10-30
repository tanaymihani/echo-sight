"""Reproduction of the original Echo-Sight v1 pipeline (git commit b5a266f), used as the baseline.

Same steps as v1: pypdf text, an all-caps heading regex for sections, 500/80 character chunks,
all-MiniLM-L6-v2 embeddings, top-8 similarity search, section-grouped context and the v1 prompt.
"""

import re
from functools import lru_cache

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from echosight.config import PDF_PATH

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are Echo-Sight, a voice-first document assistant designed for Blind "
    "and Low-Vision users. Your job is to answer questions using ONLY the "
    "retrieved document context provided below. Follow these rules strictly:\n"
    "\n"
    "1. Answer in 3 to 5 concise, complete sentences that are easy to follow "
    "when read aloud.\n"
    "2. Do NOT use markdown, bullet points, numbered lists, or any visual "
    "formatting. Write in plain, flowing prose.\n"
    "3. Mention the relevant section name when it helps the user locate "
    "information later.\n"
    "4. If the provided context does not contain enough information to answer "
    "the question, say so honestly. Never invent or hallucinate content.\n"
    "5. Keep your tone helpful, calm, and professional."
)

USER_PROMPT_TEMPLATE = (
    "Here is the retrieved context from the document, grouped by section:\n"
    "\n"
    "{context}\n"
    "\n"
    "Based only on the context above, please answer the following question:\n"
    "{question}"
)


@lru_cache(maxsize=1)
def chunks() -> list[dict]:
    reader = PdfReader(PDF_PATH)
    text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)
    matches = list(re.finditer(r"\n([A-Z][A-Z\s]{3,})\n", text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    out = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        for chunk in splitter.split_text(text[match.start():end].strip()):
            out.append({"text": chunk, "section": title})
    return out


class V1Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = chunks()
        self.vectors = self.model.encode([c["text"] for c in self.chunks], normalize_embeddings=True)

    def search(self, query: str, k: int = 8) -> list[dict]:
        scores = self.vectors @ self.model.encode(query, normalize_embeddings=True)
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]


def build_context(docs: list[dict]) -> str:
    by_section: dict[str, list[str]] = {}
    for doc in docs:
        by_section.setdefault(doc["section"], []).append(doc["text"])
    parts = []
    for section, texts in by_section.items():
        parts.append(f"--- {section} ---")
        parts += [t.strip() for t in texts]
        parts.append("")
    return "\n".join(parts)


def messages(question: str, docs: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=build_context(docs), question=question)},
    ]
