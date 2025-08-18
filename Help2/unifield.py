def load_movie_documents(file_path: str):
    """Detects file type and loads movie-related documents."""
    if file_path.endswith(".srt"):
        return load_srt(file_path)
    elif file_path.endswith(".pdf") or file_path.endswith(".txt"):
        return load_script(file_path)
    else:
        raise ValueError("File type not supported. Use .srt, .pdf, or .txt")
