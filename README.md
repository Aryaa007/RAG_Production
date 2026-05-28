# DocMind — Production RAG System

Ask questions about any PDF and get cited answers powered by Llama 3.3 70B.

---

## What it does

Upload any PDF — textbook, research paper, legal document, notes — and ask questions about it. The system retrieves the most relevant sections and generates a grounded answer with source citations.

---

## Architecture

```
PDF Upload
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embeddings (BAAI/bge-small-en-v1.5) → ChromaDB
    ↓
Query
    ↓
BM25 Keyword Search + Semantic Vector Search
    ↓
Reciprocal Rank Fusion (RRF merge)
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
Groq LLM (Llama 3.3 70B) → Cited Answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Frontend | React + Vite |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers (BAAI/bge-small-en-v1.5) |
| Retrieval | BM25 + Semantic + RRF + Cross-Encoder Reranker |
| LLM | Groq (Llama 3.3 70B) |
| Evaluation | LLM-as-a-judge with golden dataset |

---

## Features

- **Hybrid Search** — combines BM25 keyword search with semantic vector search for better retrieval accuracy
- **Cross-Encoder Reranking** — rescores retrieved chunks by reading query and chunk together as a pair
- **Hallucination Prevention** — LLM is instructed to only answer from retrieved context, says "I don't know" otherwise
- **Any PDF** — upload any document at runtime, system ingests and searches it instantly
- **Automated Evaluation** — golden dataset of 50 Q&A pairs with LLM-as-a-judge scoring, 87% accuracy

---

## Project Structure

```
├── ingest.py           Phase 1: PDF reading, chunking, embeddings, ChromaDB storage
├── retriever.py        Phase 2: BM25 + semantic search + RRF merge + reranker
├── answer.py           Phase 2: Groq LLM answer generation with citations
├── api.py              FastAPI backend — upload + ask endpoints
├── evaluate.py         Phase 3: automated evaluation pipeline
├── golden_dataset.json 15 verified Q&A pairs for evaluation
└── frontend/           React dark mode chat UI
    └── src/
        └── App.jsx
```

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/aryaa007/rag-system.git
cd rag-system
```

**2. Set up Python environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Add your Groq API key**
```bash
# create .env file
GROQ_API_KEY=your_key_here
```
Get a free key at console.groq.com

**4. Ingest your documents**
```bash
# put PDFs in the Docs/ folder
python ingest.py
```

**5. Start the backend**
```bash
uvicorn api:app --reload
```

**6. Start the frontend**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Evaluation

```bash
python evaluate.py
```

---
