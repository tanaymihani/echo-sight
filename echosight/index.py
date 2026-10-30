"""Persistent vector index: chunk embeddings in a Chroma collection, rebuilt when the chunks change."""

import hashlib
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from echosight.chunking import Chunk, chunk_sections
from echosight.config import DEVICE, INDEX_DIR, PDF_PATH, RetrievalConfig
from echosight.document import parse_pdf

_BGE_QUERY = "Represent this sentence for searching relevant passages: "
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY,
    "Snowflake/snowflake-arctic-embed-m-v1.5": _BGE_QUERY,
    "intfloat/e5-base-v2": "query: ",
}
DOC_PREFIX = {"intfloat/e5-base-v2": "passage: "}


@lru_cache(maxsize=1)
def load_sections() -> tuple:
    return tuple(parse_pdf(PDF_PATH))


@lru_cache(maxsize=None)
def load_chunks(chunk_size: int) -> tuple[Chunk, ...]:
    return tuple(chunk_sections(list(load_sections()), chunk_size))


@lru_cache(maxsize=4)
def embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=DEVICE)


def _collection_name(cfg: RetrievalConfig) -> str:
    model = cfg.embed_model.replace("/", "--")
    return f"{model}-{cfg.chunk_size}-{'h' if cfg.section_headers else 'n'}"[-63:].strip("-.")


class VectorIndex:
    def __init__(self, cfg: RetrievalConfig, rebuild: bool = False):
        self.cfg = cfg
        self.chunks = load_chunks(cfg.chunk_size)
        self.model = embedder(cfg.embed_model)
        self._position = {c.id: i for i, c in enumerate(self.chunks)}
        client = chromadb.PersistentClient(str(INDEX_DIR / "chroma"), Settings(anonymized_telemetry=False))
        texts = [DOC_PREFIX.get(cfg.embed_model, "") + c.indexed_text(cfg.section_headers) for c in self.chunks]
        fingerprint = hashlib.sha1("\x00".join([cfg.embed_model, *texts]).encode()).hexdigest()
        name = _collection_name(cfg)
        collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if rebuild or (collection.metadata or {}).get("fingerprint") != fingerprint:
            client.delete_collection(name)
            collection = client.create_collection(
                name, metadata={"hnsw:space": "cosine", "fingerprint": fingerprint})
            vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
            collection.add(
                ids=[c.id for c in self.chunks],
                embeddings=vectors.tolist(),
                documents=[c.text for c in self.chunks],
                metadatas=[{"section": c.section, "path": c.breadcrumb, "page": c.page} for c in self.chunks],
            )
        self.collection = collection

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """(chunk position, cosine similarity), best first."""
        vector = self.model.encode(QUERY_PREFIX.get(self.cfg.embed_model, "") + query, normalize_embeddings=True)
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(n, len(self.chunks)))
        return [(self._position[i], 1 - d) for i, d in zip(result["ids"][0], result["distances"][0])]
