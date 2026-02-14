HybridRAG – Configurable Hybrid Retrieval-Augmented Generation System
🚀 Overview

HybridRAG is a Streamlit-based Retrieval-Augmented Generation (RAG) system that combines semantic vector search with lexical keyword reinforcement to improve document retrieval accuracy.

Instead of relying purely on embeddings, this system implements a weighted hybrid scoring mechanism that blends:

Vector similarity (FAISS L2 distance)

Keyword match scoring

Adjustable weighting parameter (α)

This enables tunable retrieval behavior between semantic similarity and exact-term matching.

🧠 System Architecture
User Query
   ↓
Embedding Generation
   ↓
Vector Search (FAISS)
   ↓
Keyword Matching
   ↓
Hybrid Re-ranking
   ↓
Top-k Context Injection
   ↓
LLM Response Generation

🔬 Hybrid Scoring Formula

Documents are re-ranked using a weighted hybrid score:

HybridScore = α · VectorSim + (1 − α) · KeywordScore

Where:

VectorSim = 1 / (1 + VectorDistance)

VectorDistance = FAISS L2 distance (lower is better)

KeywordScore = normalized keyword match count

α ∈ [0,1] controls weighting between semantic and lexical retrieval

Special Cases:

α = 1 → Pure vector retrieval

α = 0 → Pure keyword retrieval

0 < α < 1 → Hybrid blending

This design allows dynamic control over retrieval behavior.

🛠 Tech Stack

Python

Streamlit (UI Layer)

FAISS (Vector Indexing)

LangChain (Retrieval Pipeline)

Ollama / Local LLM

✨ Key Features

Multi-document ingestion

Hybrid semantic + lexical retrieval

Configurable α weighting

Re-ranking based on blended score

Source transparency with similarity diagnostics

Conversational memory support

Fully local deployment

📦 Installation
git clone https://github.com/GeoffreyOkwi/HybridRAG.git
cd HybridRAG
pip install -r requirements.txt
streamlit run app.py

📊 Example Use Case

Upload structured documents (e.g., financial reports, technical logs, summaries) and query:

"Summarize in less than ten sentences what these documents reveal."

The system retrieves relevant context using hybrid ranking and generates a grounded response with cited sources.

🔮 Future Improvements

Add retrieval evaluation metrics (Precision@k)

Add score visualization dashboard

Add persistent vector database

Add reranker model (cross-encoder)

Deploy public demo version

👤 Author

Geoffrey Okwi
Applied AI Systems | Retrieval Engineering | Hybrid Search Architect