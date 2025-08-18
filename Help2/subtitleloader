from langchain.schema import Document
import re

def load_srt(file_path: str):
    """Load and clean SRT subtitle files into LangChain Documents."""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove timestamps & numbers
    cleaned = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", content)
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()

    docs.append(Document(page_content=cleaned, metadata={"source": file_path}))
    return docs
