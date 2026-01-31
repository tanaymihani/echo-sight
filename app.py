from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pyttsx3
import time


load_dotenv()

CHROMA_DIR = "chroma_db"


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return db


def ask_question(db, question):
    q_lower = question.lower()

    # 1. Load many chunks once
    docs = db.similarity_search(question, k=8)

    # 2. Group by section
    section_chunks = {}
    for doc in docs:
        section = doc.metadata.get("section", "Unknown")
        section_chunks.setdefault(section, []).append(doc.page_content)

    # 3. If user explicitly names a section → drill down
    for section, chunks in section_chunks.items():
        if section.lower() in q_lower:
            content = "\n".join(chunks[:3])
            return (
                f"Here is a brief summary of the section '{section}'.\n\n"
                f"{content[:600]}\n\n"
                "Would you like more details or another section?"
            )

    # 4. Otherwise → answer based on retrieved content
    seen = set()
    deduped = []

    for chunks in section_chunks.values():
        for chunk in chunks[:2]:
            snippet = chunk.strip()
            if snippet not in seen:
                seen.add(snippet)
                deduped.append(snippet)

    combined_context = "\n".join(deduped)
    response = (
        "Here is a concise answer based on the relevant parts of the document:\n\n"
        f"{combined_context[:700]}\n\n"
        "Would you like me to narrow this down to a specific section?"
    )

    return response


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)  # comfortable speech speed
    engine.say(text)
    engine.runAndWait()



if __name__ == "__main__":
    start_time = time.time()

    db = load_vector_db()
    question = "What happens if I violate university policies?"
    answer = ask_question(db, question)

    end_time = time.time()
    latency = end_time - start_time

    print("\n--- Echo-Sight Response ---\n")
    print(answer)
    print(f"\n[Latency: {latency:.2f} seconds]")
    speak(answer)



