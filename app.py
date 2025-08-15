import os
import streamlit as st
from dotenv import load_dotenv
from typing import List

from helpers.loaders import load_files
from helpers.chunker import chunk_documents
from helpers.vectorstore import build_vector_assets
from helpers.retriever import build_retriever_with_reranker
from helpers.chain import build_qa_chain, build_funny_quotes_chain
from helpers.recommender import recommend_similar_movies

load_dotenv()

st.set_page_config(page_title="AI for Movie Lovers", page_icon="🍿", layout="wide")
st.title("AI for Movie Lovers 🍿")
st.write("Upload movie scripts or subtitles, ask questions, find funny quotes, and get similar movie recs.")

if "assets" not in st.session_state:
    st.session_state.assets = None  # vectorstore + metadata
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "quotes_chain" not in st.session_state:
    st.session_state.quotes_chain = None

with st.sidebar:
    st.header("Upload Library")
    files = st.file_uploader(
        "Upload VTT/SRT/TXT/PDF (multiple allowed)",
        type=["vtt", "srt", "txt", "pdf"],
        accept_multiple_files=True,
    )
    use_reranker = st.checkbox("Use Cross-Encoder Reranker", value=True)
    k = st.slider("Retriever top-k", 2, 20, 10)
    top_n = st.slider("Reranker take top-n", 1, 10, 5)
    process = st.button("Process Library", use_container_width=True)

    st.markdown("---")
    if st.session_state.assets is None:
        st.info("No library processed.")
    else:
        st.success(f"Indexed {len(st.session_state.assets['doc_ids'])} chunks across {len(st.session_state.assets['movies'])} movies.")

if process:
    if not files:
        st.error("Please upload at least one file.")
        st.stop()

    with st.spinner("Loading files and building index..."):
        # 1) Load -> Documents (with movie metadata + timestamps when available)
        docs = load_files(files)
        if not docs:
            st.error("No text could be extracted from the uploads.")
            st.stop()

        # 2) Chunk
        chunks = chunk_documents(docs)
        # 3) Build embeddings + Chroma + movie centroids
        assets = build_vector_assets(chunks)
        # 4) Retriever (+ reranker optional)
        retriever = build_retriever_with_reranker(
            assets["vectordb"], k=k, top_n=top_n if use_reranker else None
        )
        # 5) Build chains
        qa_chain = build_qa_chain(retriever)
        quotes_chain = build_funny_quotes_chain(retriever)

        st.session_state.assets = assets
        st.session_state.retriever = retriever
        st.session_state.qa_chain = qa_chain
        st.session_state.quotes_chain = quotes_chain

    st.success("Library processed! Use the tabs below.")

st.markdown("---")

if st.session_state.assets is None:
    st.stop()

movies = sorted(list(st.session_state.assets["movies"]))

tab1, tab2, tab3 = st.tabs(["Q&A", "Funny Quotes", "Recommendations"])

with tab1:
    st.subheader("Ask about your uploaded movies")
    scope = st.selectbox("Search scope", ["All Movies"] + movies)
    question = st.text_input("Your question", placeholder="What were the main themes in Megamind?")
    ask = st.button("Ask", key="ask_btn")

    if ask:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({
                "question": question,
                "scope_movie": None if scope == "All Movies" else scope,
            })
        st.write(result["answer"])
        with st.expander("Show retrieved context"):
            for i, d in enumerate(result.get("contexts", []), 1):
                meta = d.metadata or {}
                stamp = meta.get("timestamp", "")
                movie = meta.get("movie", "?")
                prefix = f"**{movie}** " + (f"`{stamp}`" if stamp else "")
                st.markdown(f"{prefix}\n\n{d.page_content}")

with tab2:
    st.subheader("Find the funny quotes 😂")
    target_movie = st.selectbox("Pick a movie (optional)", ["All Movies"] + movies, key="quote_movie")
    hint = st.text_input("Optional hint", placeholder="jokes, banter, one-liners, witty comebacks")
    find = st.button("Find Funny Quotes")

    if find:
        with st.spinner("Hunting punchlines..."):
            result = st.session_state.quotes_chain.invoke({
                "question": hint or "Find the funny/humorous quotes and one-liners.",
                "scope_movie": None if target_movie == "All Movies" else target_movie,
            })
        st.markdown(result["answer"])
        with st.expander("Show retrieved lines"):
            for i, d in enumerate(result.get("contexts", []), 1):
                meta = d.metadata or {}
                stamp = meta.get("timestamp", "")
                movie = meta.get("movie", "?")
                st.markdown(f"**{i}. {movie}** {f'`{stamp}`' if stamp else ''} — {d.page_content}")

with tab3:
    st.subheader("Recommend similar movies 🎯")
    base = st.selectbox("Choose a movie", movies)
    n = st.slider("How many?", 1, 10, 5)
    go = st.button("Recommend")

    if go:
        recs = recommend_similar_movies(base, st.session_state.assets, top_n=n)
        if not recs:
            st.info("Need at least two movies uploaded for recommendations.")
        else:
            for rank, (m, score) in enumerate(recs, 1):
                st.markdown(f"**{rank}. {m}** — cosine {score:.3f}")
