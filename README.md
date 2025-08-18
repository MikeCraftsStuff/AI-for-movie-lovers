# AI for Movie Lovers 🍿
This is an interactive Q&A application that allows you to “chat” with movie scripts and subtitles. The app uses a Retrieval-Augmented Generation (RAG) pipeline to understand the content of scripts and subtitles, then answer your questions in natural language.


## Features
Interactive Q&A: Ask natural questions about a movie’s script or subtitle file.

Accurate Retrieval: Uses a Cross-Encoder reranker for highly relevant context selection.

Fast Generation: Powered by the Groq API (Llama 3) for near-instant answers.

Open-Source Embeddings: Uses Hugging Face’s all-MiniLM-L6-v2 for embeddings.

Clean UI: Built with Streamlit for a smooth user experience.

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
```

2. Create a Virtual Environment

# Create venv
python3 -m venv venv  

# Activate it
# On macOS/Linux:
source venv/bin/activate  

# On Windows:
venv\Scripts\activate


---

3. Install Dependencies

pip install -r requirements.txt


---

4. Set Up Environment Variables

You’ll need a Groq API key for the LLM.

1. Create a .env file in the root directory.


2. Add your API key:



GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


---

▶️ Usage

Run the app:

streamlit run app.py
1. upload movie scripts or subtitles into the sidebar.


2. Click "Process".


3. Ask questions like:

“funny quotes on the movie?”

“related movie recommendations?”

“Summarize the key points of this review.”


---
##Project Structure

.
├── helpers/
│   ├── __init__.py       # Package initializer
│   ├── chain.py          # Final RAG chain with LLM
│   ├── chunker.py        # Splits transcript into chunks
│   ├── retriever.py      # Retriever + reranker logic
│   ├── vectorstore.py    # ChromaDB vector store
│   └── youtubeloader.py  # Fetches & cleans transcripts
├── .env                  # API keys (ignored by git)
├── .gitignore            # Ignore venv, env, cache, etc.
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
└── README.md             # Documentation



🔎 How It Works

1. Ingestion: loader.py reads movie scripts (.txt, .pdf) and subtitles (.srt) into raw text.


2. Chunking: chunker splits them into smaller overlapping parts.


3. Indexing: vectorstore embeds chunks with all-MiniLM-L6-v2 and stores them in Chroma.


4. Retrieval + Reranking: retriever pulls top chunks, reranked by a Cross-Encoder.


5. Generation: Groq LLM (Llama 3) generates a grounded, accurate response.


🚀 Future Improvements

Support for multiple videos at once

Multi-language support for transcripts

Export Q&A history as text or PDF

Integration with IMDb or TMDb APIs for richer context



---

🤝 Contributing

Pull requests and feedback are welcome! 🎉
Fork the repo, create a branch, and submit a PR.


---

📜 License

This project is licensed under the MIT License.


---

