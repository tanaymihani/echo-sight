# Echo-Sight

**Echo-Sight** is a voice-first, RAG-powered document assistant designed for Blind and Low-Vision (BLV) users. It enables non-linear, conversational navigation of complex hierarchical documents — such as university handbooks — by retrieving and summarizing only the most relevant sections in plain, audio-friendly prose.

---

## Problem

Traditional screen readers force users to listen to documents linearly from start to finish. For long, structured documents like a Code of Student Conduct, finding a specific policy or procedure becomes slow, exhausting, and cognitively demanding.

## Solution

Echo-Sight acts as an **Information Concierge**. Instead of reading an entire handbook, a user can ask natural-language questions like:

- *"What happens if I violate university policies?"*
- *"How do I file a report?"*
- *"What is the appeals process?"*

Echo-Sight retrieves the most relevant document sections, synthesizes a concise answer using an LLM, and speaks it aloud — all in under 6 seconds.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | OpenAI GPT-4o-mini (via LangChain) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | ChromaDB |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **PDF Parsing** | pypdf |
| **Text-to-Speech** | pyttsx3 |
| **Environment** | Python 3 · dotenv |

---

## Architecture

```
┌──────────────┐
│   PDF Input  │  (data/code-student-conduct.pdf)
└──────┬───────┘
       │  extract & section-split (ingest.py)
       ▼
┌──────────────┐
│  Chunking &  │  RecursiveCharacterTextSplitter
│  Embedding   │  (500 chars, 80 overlap)
└──────┬───────┘
       │  sentence-transformers/all-MiniLM-L6-v2
       ▼
┌──────────────┐
│   ChromaDB   │  section-level metadata per chunk
└──────┬───────┘
       │  similarity_search(query, k=8)
       ▼
┌──────────────┐
│  RAG Prompt  │  Section-grouped context → GPT-4o-mini
│  (prompts.py)│  Grounded, plain-prose response
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   pyttsx3    │  Spoken audio output
└──────────────┘
```

1. **Ingestion** (`ingest.py`) — Extracts text from a PDF, splits it into logical sections using heading detection, chunks each section, embeds with MiniLM, and persists to ChromaDB with section metadata.
2. **Retrieval & Generation** (`app.py`) — Accepts a user question, retrieves the top-8 most similar chunks, groups them by section, and sends the context to GPT-4o-mini with a carefully designed system prompt.
3. **Prompt Engineering** (`prompts.py`) — Defines the system prompt (role, tone, grounding constraints, audio-friendly formatting rules) and the user prompt template with `{context}` and `{question}` placeholders.
4. **Voice Output** (`app.py`) — The generated answer is spoken aloud via pyttsx3 at a comfortable 170 WPM rate.

---

## Project Structure

```
echo-sight/
├── app.py              # Main RAG pipeline: retrieve → generate → speak
├── ingest.py           # PDF parsing, chunking, embedding, ChromaDB persistence
├── prompts.py          # System & user prompt templates for the LLM
├── data/
│   └── code-student-conduct.pdf   # Source document
├── chroma_db/          # Persisted vector store (git-ignored)
├── eval/
│   ├── baseline_linear.py         # Linear-search baseline for latency comparison
│   ├── baseline_times.txt         # Recorded baseline timings
│   └── tasks.txt                  # Evaluation task descriptions
├── .env                # API keys (git-ignored)
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/tanaymihani/echo-sight.git
cd echo-sight

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langchain-chroma langchain-openai langchain-community \
            langchain-text-splitters sentence-transformers \
            pypdf pyttsx3 python-dotenv
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

### Usage

**Step 1 — Ingest the document** (only needed once):

```bash
python ingest.py
```

This parses the PDF, chunks it, computes embeddings, and saves the vector store to `chroma_db/`.

**Step 2 — Ask a question:**

```bash
python app.py
```

Edit the `question` variable in `app.py` to change the query. The answer will be printed to the terminal and spoken aloud.

---

## Evaluation

System-level latency was measured for information-retrieval tasks against a baseline that performs linear keyword search over all stored chunks.

| Method | Approach | Latency |
|---|---|---|
| **Echo-Sight** | Semantic retrieval + LLM synthesis | ~5–6 seconds |
| **Baseline** | Linear scan over all chunks | ~1 min 53 sec (Task 1) |

Echo-Sight achieves **order-of-magnitude faster** task completion by combining semantic search with focused LLM summarization, while also producing richer, more useful answers than raw keyword matching.

Evaluation tasks (from `eval/tasks.txt`):
1. Find the sanctions for student misconduct
2. Find how to file a report
3. Find the appeals process

---

## User Persona

Blind and Low-Vision (BLV) users who rely on audio-based interfaces and need efficient, on-demand access to structured information in large documents.

## Status

Functional prototype demonstrating RAG-powered hierarchical document navigation with voice-first interaction. The current implementation targets a single document (Code of Student Conduct) and supports CLI-based querying.

---

## License

This project is for academic and research purposes.
