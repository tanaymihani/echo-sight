from pypdf import PdfReader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()


PDF_PATH = "/Users/tanaymihani/PycharmProjects/PythonProject8_echo-sight/data/code-student-conduct.pdf"
CHROMA_DIR = "chroma_db"


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


def split_into_sections(text):
    heading_pattern = r"\n([A-Z][A-Z\s]{3,})\n"
    matches = list(re.finditer(heading_pattern, text))

    sections = []

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        title = matches[i].group(1).strip()
        content = text[start:end].strip()

        sections.append((title, content))

    return sections


def build_vector_db(sections):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    texts = []
    metadatas = []

    for title, content in sections:
        chunks = splitter.split_text(content)

        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "section": title,
                "chunk_id": idx
            })

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return db


if __name__ == "__main__":
    raw_text = extract_text(PDF_PATH)
    sections = split_into_sections(raw_text)

    print(f"Sections found: {len(sections)}")

    db = build_vector_db(sections)
    print("ChromaDB built and persisted successfully.")
