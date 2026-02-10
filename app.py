import streamlit as st
import os
import tempfile

from src.data_loader import load_all_pdfs
from src.embedding import EmbeddingPipeline
from src.vectorstore import VectorStore
from src.search import RAGSearch


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Research Assistant (RAG)",
    layout="wide"
)

st.title("📄 AI Research Assistant")
st.write(
    "Upload research papers (PDFs) and ask questions grounded in their content."
)

# ----------------------------
# Session state
# ----------------------------
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "rag" not in st.session_state:
    st.session_state.rag = None


# ----------------------------
# Sidebar – PDF upload
# ----------------------------
st.sidebar.header("📤 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more research papers",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("🔍 Submit & Process Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            # Save PDFs temporarily
            temp_dir = tempfile.mkdtemp()
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())

            # Load documents
            docs = load_all_pdfs(temp_dir)

            # Chunk + embed
            pipeline = EmbeddingPipeline()
            chunks = pipeline.chunk_documents(docs)
            embeddings = pipeline.embed_chunks(chunks)

            # Store in Chroma
            store = VectorStore()
            store.build_from_documents(chunks, embeddings)

            # Initialize RAG
            st.session_state.rag = RAGSearch()
            st.session_state.indexed = True

        st.sidebar.success("✅ Documents processed successfully!")


# ----------------------------
# Main – Question Answering
# ----------------------------
st.header("❓ Ask a Question")

if not st.session_state.indexed:
    st.info("Upload and index PDFs to start asking questions.")
else:
    query = st.text_input(
        "Enter your question about the uploaded papers:",
        placeholder="e.g. What is the main contribution of this paper?"
    )

    if st.button("💡 Get Answer"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.rag.search_and_summarize(query)

            st.subheader("📌 Answer")

            # Split answer and sources (if present)
            if "\n\nSources:\n" in answer:
                answer_text, sources_text = answer.split("\n\nSources:\n", 1)
            else:
                answer_text, sources_text = answer, None

            st.write(answer_text)

            # Display sources cleanly
            if sources_text:
                st.markdown("### 📚 Sources")
                for line in sources_text.split("\n"):
                    st.markdown(f"- {line}")
