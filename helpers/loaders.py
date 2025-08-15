from typing import List
from io import BytesIO
from langchain.schema import Document
import os
import re
import webvtt
import pysrt
from PyPDF2 import PdfReader


def _normalize_movie_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[_\-]+", " ", base).strip()


def _from_vtt(file, filename: str) -> List[Document]:
    docs = []
    movie = _normalize_movie_name(filename)
    vtt = webvtt.read_buffer(BytesIO(file.read()))
    for cap in vtt:
        text = cap.text.strip()
        if not text:
            continue
        stamp = f"{cap.start}–{cap.end}"
        docs.append(Document(page_content=text, metadata={"movie": movie, "timestamp": stamp}))
    return docs


def _from_srt(file, filename: str) -> List[Document]:
    docs = []
    movie = _normalize_movie_name(filename)
    subs = pysrt.from_string(file.read().decode("utf-8", errors="ignore"))
    for it in subs:
        text = it.text.replace("\n", " ").strip()
        if not text:
            continue
        start = str(it.start.to_time())
        end = str(it.end.to_time())
        stamp = f"{start}–{end}"
        docs.append(Document(page_content=text, metadata={"movie": movie, "timestamp": stamp}))
    return docs


def _from_txt(file, filename: str) -> List[Document]:
    movie = _normalize_movie_name(filename)
    text = file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"movie": movie})]


def _from_pdf(file, filename: str) -> List[Document]:
    movie = _normalize_movie_name(filename)
    reader = PdfReader(BytesIO(file.read()))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    return [Document(page_content=text, metadata={"movie": movie})]


def load_files(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        file_copy = BytesIO(f.getvalue())
        file_copy.seek(0)
        if ext == ".vtt":
            docs.extend(_from_vtt(file_copy, name))
        elif ext == ".srt":
            file_copy.seek(0)
            docs.extend(_from_srt(file_copy, name))
        elif ext == ".txt":
            file_copy.seek(0)
            docs.extend(_from_txt(file_copy, name))
        elif ext == ".pdf":
            file_copy.seek(0)
            docs.extend(_from_pdf(file_copy, name))
        else:
            continue
    return docs
