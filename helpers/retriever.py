from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


class ScopedRetriever:
    def __init__(self, base_retriever, scope_movie: Optional[str] = None):
        self.base = base_retriever
        self.scope_movie = scope_movie

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base.get_relevant_documents(query)
        if not self.scope_movie:
            return docs
        return [d for d in docs if (d.metadata or {}).get("movie") == self.scope_movie]


class RerankingRetriever:
    def __init__(self, base_retriever, cross_encoder: Optional[CrossEncoder] = None, top_n: Optional[int] = None):
        self.base = base_retriever
        self.xe = cross_encoder
        self.top_n = top_n

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base.get_relevant_documents(query)
        if not docs or self.xe is None or not self.top_n:
            return docs
        pairs = [(query, d.page_content) for d in docs]
        scores = self.xe.predict(pairs)
        reranked = [d for d, s in sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)[: self.top_n]]
        return reranked


def build_retriever_with_reranker(vectordb: Chroma, k: int = 10, top_n: int = None):
    base = vectordb.as_retriever(search_kwargs={"k": k})
    if top_n is None:
        return base
    xe = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return RerankingRetriever(base, xe, top_n)
