# AI for Movie Lovers 🍿
Upload movie scripts or subtitles, ask questions with RAG, extract **funny quotes**, and get **similar movie** recommendations.

## Features
- **Interactive Q&A:** Natural-language questions over your uploaded movies.
- **Funny Quotes Extractor:** Finds witty one-liners and comedic moments (uses timestamps if present).
- **Recommendations:** Content-based similar-movie suggestions from your library.
- **Tech:** LangChain + Chroma + MiniLM embeddings + optional Cross-Encoder reranker + Groq Llama 3.1 8B.

## Tech Stack
- **Framework:** LangChain
- **UI:** Streamlit
- **LLM:** Groq (Llama 3.1 8B Instant)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** ChromaDB (in-memory)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional)

## Getting Started
### 1) Clone
```bash
git clone https://github.com/your-username/ai-for-movie-lovers.git
cd ai-for-movie-lovers
