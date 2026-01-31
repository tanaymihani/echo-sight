# Echo-Sight

**Echo-Sight** is a context-aware, voice-first information navigation system designed to help Blind and Low-Vision (BLV) users efficiently browse complex hierarchical documents such as university handbooks, policies, and technical manuals.

## Problem
Traditional screen readers present documents linearly, making it slow and cognitively demanding for users to skim or locate specific information in long, structured documents.

## Solution
Echo-Sight acts as an *Information Concierge* that enables non-linear navigation. Users can ask high-level questions such as:
- “What are the main sections?”
- “Tell me more about sanctions”

Instead of reading entire documents, Echo-Sight retrieves and summarizes only the most relevant sections in an audio-friendly format.

## System Architecture
- **Document Ingestion**: PDF documents are parsed and segmented into logical sections.
- **Chunking & Embeddings**: Sections are split into small chunks and embedded using sentence-transformer embeddings.
- **Vector Database**: ChromaDB stores document chunks with section-level metadata.
- **Retrieval Logic**: User queries are routed using semantic retrieval and hierarchical section matching.
- **Voice Output**: Responses are spoken aloud using text-to-speech for accessibility.

## Evaluation
System-level latency was measured for information retrieval tasks and compared against a baseline linear retrieval approach.

- **Echo-Sight latency**: ~5–6 seconds
- **Baseline linear retrieval**: ~6 seconds

Echo-Sight achieves comparable latency while significantly improving navigability and reducing cognitive load by avoiding linear traversal.

## User Persona
Blind and Low-Vision (BLV) users who rely on audio-based interfaces and need efficient access to structured information.

## Status
This project is a functional prototype demonstrating hierarchical document navigation and voice-first interaction.
