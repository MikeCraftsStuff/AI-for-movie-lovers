
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    # Subtitles often arrive pre-chunked (per caption). For long pages/scripts, we chunk.
    long_docs = []
    short_docs = []
    for d in docs:
        if len(d.page_content) > chunk_size:
            long_docs.append(d)
        else:
            short_docs.append(d)

    if not long_docs:
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return short_docs + splitter.split_documents(long_docs)
