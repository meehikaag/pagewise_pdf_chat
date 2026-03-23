import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import build_vector_store, search_similar_chunks
from llm import get_answer

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PageWise – Ask Your PDF",
    page_icon="📄",
    layout="centered"
)

# ── Minimal clean CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}
.main { background: #0f0f0f; }
.stApp { background: #0f0f0f; color: #f0f0f0; }

.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.title-block h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #f0f0f0;
    letter-spacing: -2px;
    margin-bottom: 0.2rem;
}
.title-block p {
    color: #888;
    font-size: 1rem;
}
.answer-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #7c6af7;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    color: #e0e0e0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.source-box {
    background: #141414;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
    color: #666;
    font-size: 0.82rem;
    font-family: monospace;
}
.status-pill {
    display: inline-block;
    background: #1c1c2e;
    color: #7c6af7;
    border: 1px solid #7c6af7;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>📄 PageWise</h1>
    <p>Upload a PDF. Ask anything. Get answers.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings_store" not in st.session_state:
    st.session_state.embeddings_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Upload ────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Reading and indexing your PDF..."):
        # Save temp file (works on both Windows and Linux)
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.read())

        # Extract + chunk
        raw_text = extract_text_from_pdf(str(tmp_path))
        chunks = chunk_text(raw_text)
        store = build_vector_store(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings_store = store
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []

    st.markdown(f'<span class="status-pill">✓ Indexed {len(chunks)} chunks from {uploaded_file.name}</span>', unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────
if st.session_state.embeddings_store is not None:
    st.markdown("---")
    question = st.text_input(
        "Ask a question about your PDF",
        placeholder='e.g. "What are the key findings?" or "Summarise section 3"',
        key="question_input"
    )

    if st.button("Ask →", use_container_width=True) and question.strip():
        with st.spinner("Thinking..."):
            relevant_chunks = search_similar_chunks(
                question,
                st.session_state.embeddings_store,
                st.session_state.chunks,
                top_k=4
            )
            answer = get_answer(question, relevant_chunks)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": relevant_chunks
        })

    # Display history (newest first)
    for item in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {item['question']}**")
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

        with st.expander("View source chunks"):
            for i, chunk in enumerate(item["sources"], 1):
                st.markdown(f'<div class="source-box"><b>Chunk {i}:</b> {chunk[:300]}...</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

else:
    st.info("Upload a PDF above to get started.")