# ─────────────────────────────────────────────────────────
# app.py — Streamlit UI with PDF + Image Support
# ─────────────────────────────────────────────────────────

import os
import uuid
import streamlit as st
from PIL import Image

# Import RAG modules
from rag.config import get_secret
from rag.ingest import ingest_pdf
from rag.embedder import (
    store_chunks,
    chunk_already_exists,
    get_collection_stats,
    delete_session_collection,
)
from rag.retriever import retrieve_with_confidence
from rag.generator import generate_answer
from rag.vision import run_image_pipeline, IMAGE_RELEVANCE_THRESHOLD

# ── Page configuration ──────────────────────────────────
st.set_page_config(
    page_title="Document & Image Q&A Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom styling ──────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #4DA6FF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #aaa;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #1a2a3a;
        border-left: 4px solid #4DA6FF;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #E8E8E8 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .answer-box * { color: #E8E8E8 !important; }
    .source-box {
        background-color: #1e1e2e;
        border: 1px solid #333;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #ccc !important;
    }
    .source-box * { color: #ccc !important; }
    .confidence-high   { color: #0d9b76; font-weight: 600; }
    .confidence-medium { color: #e6a817; font-weight: 600; }
    .confidence-low    { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Per-Session Isolation Initialisation ────────────────
# Each user session gets a unique collection name so users cannot see each other's data
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"

session_collection = st.session_state.session_id

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "text_chat_history" not in st.session_state:
    st.session_state.text_chat_history = []

if "image_chat_history" not in st.session_state:
    st.session_state.image_chat_history = []

# ── Check Critical Secrets ──────────────────────────────
gemini_api_key = get_secret("GEMINI_API_KEY")
groq_api_key = get_secret("GROQ_API_KEY")

if not gemini_api_key:
    st.error(
        "⚠️ **Missing GEMINI_API_KEY**: Please add `GEMINI_API_KEY` to your "
        "Streamlit Community Cloud Secrets (`.streamlit/secrets.toml`) or your `.env` file."
    )

if not groq_api_key:
    st.warning(
        "⚠️ **Missing GROQ_API_KEY**: Text answer generation requires `GROQ_API_KEY` in "
        "Streamlit Secrets or your `.env` file."
    )

# ── Sidebar: Document Library ───────────────────────────
with st.sidebar:
    st.markdown("## 📁 Document Library")
    st.markdown("Upload PDFs to build or update your indexed knowledge base.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Digitally created PDFs work best."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if file_name in st.session_state.indexed_files:
            st.info(f"✅ **{file_name}** is already indexed.")
        else:
            if st.button(f"📥 Index '{file_name}'", use_container_width=True):
                with st.spinner("Processing & indexing document..."):
                    upload_dir = os.path.join("data", "uploaded", session_collection)
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, file_name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    ingest_result = ingest_pdf(file_path)

                    if ingest_result["is_scanned"]:
                        st.warning("⚠️ Scanned PDF detected. Text extraction might be partial.")

                    if ingest_result["chunk_count"] == 0:
                        st.error("❌ No text could be extracted from this PDF.")
                    else:
                        new_chunks = [
                            c for c in ingest_result["chunks"]
                            if not chunk_already_exists(c["chunk_id"], collection_name=session_collection)
                        ]

                        if new_chunks:
                            stored = store_chunks(new_chunks, collection_name=session_collection)
                            st.session_state.indexed_files.append(file_name)
                            st.success(f"✅ **{file_name}** indexed! ({stored} chunks stored)")
                        else:
                            st.info("Document chunks are already indexed.")
                            st.session_state.indexed_files.append(file_name)

    st.divider()
    st.markdown("### Indexed Documents")
    stats = get_collection_stats(collection_name=session_collection)
    st.metric("Chunks in your session", stats["total_chunks"])

    if st.session_state.indexed_files:
        for fname in st.session_state.indexed_files:
            st.markdown(f"• 📄 {fname}")
    else:
        st.markdown("*No documents indexed in this session.*")

    st.divider()
    debug_mode = st.checkbox("🔍 Debug Mode (Show Pipeline Details)", value=False)
    if debug_mode:
        os.environ["DEBUG_VISION"] = "1"
    else:
        os.environ["DEBUG_VISION"] = "0"

    st.divider()
    if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
        delete_session_collection(session_collection)
        st.session_state.indexed_files = []
        st.session_state.text_chat_history = []
        st.session_state.image_chat_history = []
        st.success("Session database cleared.")
        st.rerun()

# ── Main Content Area ───────────────────────────────────
st.markdown('<div class="main-header">📑 Document & Vision Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents via text or upload photos to inspect and query against indexed manuals, policies, or specs.</div>', unsafe_allow_html=True)

stats = get_collection_stats(collection_name=session_collection)
if stats["total_chunks"] == 0:
    st.info("👈 **Upload and Index a PDF** using the sidebar to start asking questions.")
else:
    tab1, tab2 = st.tabs(["💬 Text Question", "📸 Image + Question"])

    # ─────────────────────────────────────────────────────────
    # TAB 1: Text-Only Flow
    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Ask a Question from Documents")
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Your question:", placeholder="e.g. What is the policy limit?", key="text_query")
        with col2:
            st.write("")
            st.write("")
            ask_btn = st.button("Ask Text →", key="text_ask_btn", use_container_width=True, type="primary")

        if ask_btn and query:
            with st.spinner("Searching documents & generating answer..."):
                retrieval_res = retrieve_with_confidence(query, top_k=5, collection_name=session_collection)
                chunks = retrieval_res["chunks"]
                confidence = retrieval_res["confidence"]
                warning = retrieval_res["warning"]

                gen_res = generate_answer(query, chunks)
                answer = gen_res["answer"]
                sources = gen_res["sources"]

                st.session_state.text_chat_history.insert(0, {
                    "question": query,
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "warning": warning
                })

        if st.session_state.text_chat_history:
            st.divider()
            for entry in st.session_state.text_chat_history:
                st.markdown(f"**❓ {entry['question']}**")
                conf = entry["confidence"]
                if conf == "HIGH":
                    st.markdown('<span class="confidence-high">● High confidence</span>', unsafe_allow_html=True)
                elif conf == "MEDIUM":
                    st.markdown('<span class="confidence-medium">● Medium confidence</span>', unsafe_allow_html=True)
                elif conf == "LOW":
                    st.markdown('<span class="confidence-low">● Low confidence</span>', unsafe_allow_html=True)

                if entry["warning"]:
                    st.warning(entry["warning"])

                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)

                if entry["sources"]:
                    with st.expander(f"📄 View Sources ({len(entry['sources'])})"):
                        for s in entry["sources"]:
                            st.markdown(f'<div class="source-box"><strong>{s["source"]} — Page {s["page_num"]}</strong> (Score: {s["score"]})<br>{s["text"]}</div><br>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TAB 2: Image Pipeline Flow
    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Upload an Image + Ask a Question")
        st.caption("Upload a photo of an equipment label, error screen, broken part, or receipt. The image text is extracted and matched against your indexed PDFs.")

        img_col1, img_col2 = st.columns([1, 1])

        with img_col1:
            img_file = st.file_uploader(
                "Upload Photo (Required)",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_uploader"
            )
            if img_file:
                st.image(img_file, caption="Uploaded Photo", use_container_width=True)

        with img_col2:
            img_query = st.text_area(
                "What is your question about this photo? (Required)",
                placeholder="e.g. Is this damage covered? What does this error code mean?",
                height=120,
                key="img_question"
            )
            img_submit = st.button("Analyse Photo & Search Documents →", type="primary", use_container_width=True)

        if img_submit:
            if not gemini_api_key:
                st.error("❌ Cannot analyze image: `GEMINI_API_KEY` is missing from configuration.")
            elif not img_file:
                st.error("Please upload an image.")
            elif not img_query.strip():
                st.error("Please provide a question alongside the image.")
            else:
                with st.spinner("Analyzing image, searching documents, and generating answer..."):
                    img_bytes = img_file.getvalue()
                    res = run_image_pipeline(img_bytes, img_file.name, img_query, collection_name=session_collection)

                    st.session_state.image_chat_history.insert(0, {
                        "question": img_query,
                        "filename": img_file.name,
                        "answer": res["answer"],
                        "sources": res["sources"],
                        "confidence": res["confidence"],
                        "warning": res["warning"],
                        "analysis": res["analysis"],
                        "debug_info": res["debug_info"],
                        "error": res.get("error", "")
                    })

        if st.session_state.image_chat_history:
            st.divider()
            for entry in st.session_state.image_chat_history:
                st.markdown(f"**📸 Image: {entry['filename']}** | **❓ {entry['question']}**")

                conf = entry["confidence"]
                if conf == "HIGH":
                    st.markdown('<span class="confidence-high">● High confidence</span>', unsafe_allow_html=True)
                elif conf == "MEDIUM":
                    st.markdown('<span class="confidence-medium">● Medium confidence</span>', unsafe_allow_html=True)
                elif conf == "LOW":
                    st.markdown('<span class="confidence-low">● Low confidence</span>', unsafe_allow_html=True)
                elif conf == "GATE_BLOCKED":
                    st.markdown('<span class="confidence-low">● Not Found in Indexed Documents</span>', unsafe_allow_html=True)

                if entry["warning"]:
                    st.warning(entry["warning"])

                st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)

                # Show Analysis breakdown
                analysis = entry.get("analysis", {})
                if analysis and "error" not in analysis:
                    with st.expander("🔍 Image Understanding Breakdown"):
                        st.markdown(f"**Description:** {analysis.get('description', '')}")
                        if analysis.get('ocr_text'):
                            st.markdown(f"**Extracted Text:** `{analysis.get('ocr_text', '')}`")
                        if analysis.get('entities'):
                            st.markdown(f"**Entities:** {', '.join(analysis.get('entities', []))}")
                        if analysis.get('search_queries'):
                            st.markdown("**Generated Search Queries:**")
                            for q in analysis.get('search_queries', []):
                                st.markdown(f"- *{q}*")

                if entry["sources"]:
                    with st.expander(f"📄 View Cited Passages ({len(entry['sources'])})"):
                        for s in entry["sources"]:
                            st.markdown(f'<div class="source-box"><strong>{s["source"]} — Page {s["page_num"]}</strong> (Score: {s["score"]})<br>{s["text"]}</div><br>', unsafe_allow_html=True)

                # Intermediate debug outputs
                if debug_mode:
                    if entry.get("error"):
                        with st.expander("🛠️ Raw Error Details (Debug Mode)"):
                            st.code(entry["error"], language="text")
                    if entry.get("debug_info"):
                        with st.expander("🛠️ Raw Debug Logs"):
                            st.json(entry["debug_info"])
