import time
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "../chroma_db"


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return db


def linear_search(db, keyword):
    docs = db.similarity_search("", k=500)

    for doc in docs:
        if keyword.lower() in doc.page_content.lower():
            return doc.page_content

    return None


if __name__ == "__main__":
    start_time = time.time()

    db = load_vector_db()
    result = linear_search(db, "sanctions")

    end_time = time.time()
    latency = end_time - start_time

    print("Baseline linear search result found:", result is not None)
    print(f"[Baseline Latency: {latency:.2f} seconds]")
