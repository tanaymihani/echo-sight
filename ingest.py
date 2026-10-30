"""Parse the PDF, chunk it by heading, embed the chunks and persist them to Chroma.

    python ingest.py

Optional: the app rebuilds the index automatically whenever the PDF or chunking changes.
"""

from collections import Counter

from echosight.config import PDF_PATH, RetrievalConfig
from echosight.document import parse_pdf
from echosight.index import VectorIndex


def main():
    cfg = RetrievalConfig()
    sections = parse_pdf(PDF_PATH)
    index = VectorIndex(cfg, rebuild=True)
    levels = Counter(len(s.path) for s in sections)
    print(f"Parsed {len(sections)} sections ({levels[1]} top-level, {levels[2]} sections, "
          f"{levels[3]} subsections) from {PDF_PATH.name}")
    print(f"Indexed {len(index.chunks)} chunks with {cfg.embed_model} "
          f"into collection {index.collection.name!r}")


if __name__ == "__main__":
    main()
