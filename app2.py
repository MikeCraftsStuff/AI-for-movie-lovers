# app.py
import streamlit as st
from helpers.youtubeloader import load_youtube_transcript  # your custom loader
from helpers.chunker import chunk_text
from helpers.vectorstore import create_vectorstore
from helpers.retriever import get_retriever
from helpers.chain import create_rag_chain

st.set_page_config(page_title="AI Movie Lovers", layout="wide")

st.title("🎬 AI for Movie Lovers")
st.write("Upload movie scripts or subtitles, or paste YouTube links to start chatting!")

# --- Sidebar for input ---
st.sidebar.header("Movie Source")
source_type = st.sidebar.selectbox("Choose source type", ["YouTube URL", "Upload File"])
movie_data = None

if source_type == "YouTube URL":
    url = st.sidebar.text_input("Paste YouTube video URL here")
    if url:
        st.sidebar.info("Processing transcript...")
        movie_data = load_youtube_transcript(url)  # returns cleaned text
elif source_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload movie script/subtitle", type=["txt","srt","pdf"])
    if uploaded_file:
        movie_data = uploaded_file.read().decode("utf-8")  # simple for txt/srt; pdf requires extra handling

# --- Process the data ---
if movie_data:
    st.success("Movie script loaded!")
    
    # Chunk the text
    chunks = chunk_text(movie_data)
    
    # Create vector store
    vectorstore = create_vectorstore(chunks)
    
    # Get retriever (with optional reranker)
    retriever = get_retriever(vectorstore)
    
    # Create RAG chain
    rag_chain = create_rag_chain(retriever)

    # --- Chat interface ---
    st.header("Ask Questions About This Movie")
    user_question = st.text_input("Type your question here")
    
    if user_question:
        with st.spinner("Generating answer..."):
            answer = rag_chain.run(user_question)
        st.markdown(f"**Answer:** {answer}")

        # Optional: similar movie recommendations (example placeholder)
        st.subheader("You might also like:")
        st.write("🎥 Megamind 2, 🎥 Despicable Me, 🎥 The Incredibles")  # can be dynamic later

else:
    st.info("Please provide a movie script or YouTube URL to start.")
