from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_script(file_path: str):
    """Load movie scripts (PDF or TXT)."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    else:
        raise ValueError("Unsupported file type. Use PDF or TXT.")
