"""
Hybrid RAG with ReRanker — Streamlit App
=========================================
Combines dense retrieval (ChromaDB) + sparse retrieval (BM25) via
Reciprocal Rank Fusion, followed by Cross-Encoder reranking.
Uses Groq LLM for final answer generation.
"""

import os
import warnings
import time

warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_groq import ChatGroq

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG — Ask Anything",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Load environment
# ──────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# ──────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0a0a0f;
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 3rem 1rem 1.5rem;
    }
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem;
        letter-spacing: -1px;
    }
    .hero p {
        color: #6b7280;
        font-size: 0.95rem;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Chip row ── */
    .chip-row {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1.2rem 0 2rem;
    }
    .chip {
        background: rgba(99, 102, 241, 0.08);
        color: #818cf8;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 100px;
        border: 1px solid rgba(99, 102, 241, 0.18);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Answer card ── */
    .answer-card {
        background: linear-gradient(160deg, #111827 0%, #0f172a 50%, #0c0f1d 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .answer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
    }
    .answer-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }
    .answer-label .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6366f1;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(99,102,241,0.5); }
        50% { opacity: 0.7; box-shadow: 0 0 8px 4px rgba(99,102,241,0.2); }
    }
    .answer-text {
        color: #e2e8f0;
        font-size: 1.05rem;
        line-height: 1.85;
        font-weight: 400;
    }
    .answer-meta {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        color: #4b5563;
        font-weight: 500;
    }
    .meta-item .meta-icon {
        font-size: 0.85rem;
    }

    /* ── Suggested questions ── */
    .suggest-title {
        color: #4b5563;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        text-align: center;
        margin: 2rem 0 1rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #374151;
    }
    .empty-state .empty-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }
    .empty-state p {
        font-size: 0.9rem;
        color: #4b5563;
        max-width: 360px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Warning/Info boxes ── */
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #fbbf24;
        font-size: 0.9rem;
        margin: 1rem 0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d0d14;
    }

    /* ── Hide branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Input styling ── */
    .stTextInput > div > div > input {
        background: #111827 !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.2rem !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Cached resource loaders
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Loading PDF and building chunks…")
def load_and_chunk_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    docs = splitter.split_documents(pages)
    chunks = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    return chunks, metadatas


@st.cache_resource(show_spinner="🧠 Loading embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="🎯 Loading reranker model…")
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="📦 Initialising vector store…")
def init_chromadb(_chunks, _metadatas):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(
        path=os.path.join(os.path.dirname(__file__), "chroma_db")
    )
    collection = client.get_or_create_collection(
        name="hybrid_rag", embedding_function=embedding_fn
    )
    if collection.count() == 0:
        batch = 500
        for s in range(0, len(_chunks), batch):
            e = min(s + batch, len(_chunks))
            collection.add(
                documents=_chunks[s:e],
                ids=[str(i) for i in range(s, e)],
                metadatas=_metadatas[s:e],
            )
    return collection


@st.cache_resource(show_spinner="📊 Building BM25 index…")
def build_bm25(_chunks):
    return BM25Okapi([doc.lower().split() for doc in _chunks])


# ──────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────
def hybrid_retrieve(query, collection, bm25, chunks, reranker):
    # Dense retrieval
    result = collection.query(query_texts=[query], n_results=5)
    dense_docs = [
        doc
        for doc, d in zip(result["documents"][0], result["distances"][0])
        if d < 1.0
    ]

    # Sparse retrieval (BM25)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
    sparse_docs = [chunks[i] for i, _ in top_indices]

    # Reciprocal Rank Fusion
    rrf = {}
    for rank, doc in enumerate(dense_docs):
        rrf[doc] = rrf.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(sparse_docs):
        rrf[doc] = rrf.get(doc, 0) + 1 / (60 + rank)

    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in merged[:5]]

    if not top_docs:
        return []

    # Cross-Encoder reranking
    pairs = [[query, doc] for doc in top_docs]
    ce_scores = reranker.predict(pairs)
    ranked = sorted(zip(ce_scores, top_docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked]


def generate_answer(query, context_docs):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_KEY)
    prompt = f"""
Use the context to answer the question, even if the wording differs slightly.
Return ONLY the final answer.
Do not include labels like "Ans:".
If the context does not contain enough information, say "Not found in the provided documents."

Context:
{context_docs}

Question:
{query}
"""
    response = llm.invoke(prompt)
    return response.content


# ──────────────────────────────────────────────
# UI — Hero
# ──────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-icon">✨</div>
    <h1>Ask Your Document</h1>
    <p>Powered by Hybrid RAG — combining semantic search, keyword matching, and intelligent reranking for precise answers.</p>
</div>
<div class="chip-row">
    <span class="chip">Dense Search</span>
    <span class="chip">BM25</span>
    <span class="chip">Rank Fusion</span>
    <span class="chip">Cross-Encoder</span>
    <span class="chip">Groq LLM</span>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────
PDF_PATH = os.path.join(os.path.dirname(__file__), "GK_Questions.pdf")

if not os.path.exists(PDF_PATH):
    st.error("📁 `GK_Questions.pdf` not found. Place it in the app directory.")
    st.stop()

chunks, metadatas = load_and_chunk_pdf(PDF_PATH)
embed_model = load_embedding_model()
reranker_model = load_reranker()
collection = init_chromadb(chunks, metadatas)
bm25 = build_bm25(chunks)

# ──────────────────────────────────────────────
# Chat interface
# ──────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested questions
if not st.session_state.messages:
    st.markdown('<div class="suggest-title">💡 Try asking</div>', unsafe_allow_html=True)

    suggestions = [
        "What is the name given to the EU's parliament?",
        "Which is the largest continent in the world?",
        "Who invented the telephone?",
    ]
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        with col:
            if st.button(q, key=f"sug_{q}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
<div style="display:flex; justify-content:flex-end; margin:0.8rem 0;">
    <div style="background:linear-gradient(135deg, #4f46e5, #6366f1); color:white;
         padding:0.8rem 1.2rem; border-radius:16px 16px 4px 16px;
         max-width:75%; font-size:0.95rem; font-weight:500;">
        {msg["content"]}
    </div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div class="answer-card">
    <div class="answer-label"><div class="dot"></div> AI Answer</div>
    <div class="answer-text">{msg["content"]}</div>
    <div class="answer-meta">
        <span class="meta-item"><span class="meta-icon">📄</span> GK_Questions.pdf</span>
        <span class="meta-item"><span class="meta-icon">⚡</span> Hybrid RAG + ReRanker</span>
    </div>
</div>""",
            unsafe_allow_html=True,
        )


# Query input
query = st.chat_input("Ask anything about the document…")

# Handle suggested question click
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    if not GROQ_KEY:
        st.markdown(
            '<div class="warning-box">⚠️ Groq API key not found. '
            "Add <code>GROQ_API_KEY</code> to your <code>.env</code> file.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(
        f"""
<div style="display:flex; justify-content:flex-end; margin:0.8rem 0;">
    <div style="background:linear-gradient(135deg, #4f46e5, #6366f1); color:white;
         padding:0.8rem 1.2rem; border-radius:16px 16px 4px 16px;
         max-width:75%; font-size:0.95rem; font-weight:500;">
        {query}
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    # Process
    start_time = time.time()

    with st.spinner(""):
        # Retrieval
        reranked_docs = hybrid_retrieve(
            query, collection, bm25, chunks, reranker_model
        )

        if not reranked_docs:
            answer = "I couldn't find any relevant information in the document for that question."
        else:
            answer = generate_answer(query, reranked_docs)

    elapsed = round(time.time() - start_time, 1)

    # Show answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(
        f"""
<div class="answer-card">
    <div class="answer-label"><div class="dot"></div> AI Answer</div>
    <div class="answer-text">{answer}</div>
    <div class="answer-meta">
        <span class="meta-item"><span class="meta-icon">📄</span> GK_Questions.pdf</span>
        <span class="meta-item"><span class="meta-icon">⚡</span> {elapsed}s</span>
        <span class="meta-item"><span class="meta-icon">🔗</span> {len(reranked_docs)} sources</span>
    </div>
</div>""",
        unsafe_allow_html=True,
    )
