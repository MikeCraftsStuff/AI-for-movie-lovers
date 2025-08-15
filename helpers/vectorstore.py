from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Build vectors first so we can also compute per-movie centroids (for recommendations)

def build_vector_assets(chunks: List[Document]) -> Dict[str, Any]:
    texts = [d.page_content for d in chunks]
    metas = [d.metadata for d in chunks]

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.embed_documents(texts)  # List[List[float]]

    vectordb = Chroma.from_embeddings(texts, vectors, metadatas=metas)

    # Compute movie centroids
    by_movie = defaultdict(list)
    for v, md in zip(vectors, metas):
        movie = (md or {}).get("movie", "Unknown")
        by_movie[movie].append(np.array(v, dtype=np.float32))

    centroids = {}
    for movie, arrs in by_movie.items():
        mat = np.vstack(arrs)
        centroids[movie] = mat.mean(axis=0)

    return {
        "vectordb": vectordb,
        "centroids": centroids,  # np arrays
        "movies": set(by_movie.keys()),
        "doc_ids": list(range(len(texts))),
        "embedder": embedder,
    }
