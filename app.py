from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
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


def build_context(docs):
    """Group retrieved chunks by section and format into a labelled context block."""
    section_chunks = {}
    for doc in docs:
        section = doc.metadata.get("section", "Unknown")
        section_chunks.setdefault(section, []).append(doc.page_content)

    context_parts = []
    for section, chunks in section_chunks.items():
        context_parts.append(f"--- {section} ---")
        for chunk in chunks:
            context_parts.append(chunk.strip())
        context_parts.append("")  # blank line between sections

    return "\n".join(context_parts)


def ask_question(db, question):
    # 1. Retrieve relevant chunks
    docs = db.similarity_search(question, k=8)

    # 2. Build section-grouped context string
    context = build_context(docs)

    # 3. Synthesize an answer via LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        ))
    ]

    response = llm.invoke(messages)
    return response.content


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
