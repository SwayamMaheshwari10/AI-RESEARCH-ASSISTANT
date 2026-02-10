# 📄 AI Research Assistant (RAG)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload research papers (PDFs), ask questions grounded in their content, and receive fact-based answers with source citations.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **📤 Multi-PDF Upload** | Upload one or more research papers in PDF format. |
| **🔍 Semantic Retrieval (RAG)** | Relevant document chunks are retrieved using vector embeddings (SentenceTransformers + ChromaDB). |
| **🤖 LLM-Powered Answers** | Answers are generated using Gemini (via LangChain), constrained strictly to retrieved content. |
| **📚 Metadata-Based Citations** | Every answer includes deterministic citations (PDF name + page number) derived directly from retrieval metadata — no hallucinated references. |
| **🧱 Modular Architecture** | Clean separation between ingestion, embedding, vector storage, retrieval, and UI. |
| **🖥️ Interactive Streamlit UI** | Simple and intuitive interface for document upload and question answering. |

---

## 🧠 System Architecture

### Source Code Structure

```
src/
├── data_loader.py   → Load & parse PDFs
├── embedding.py     → Chunk + embed text
├── vectorstore.py   → Store embeddings in ChromaDB
├── search.py        → Retrieval + LLM generation (RAG)
└── app.py           → Streamlit UI (or at project root)
```

### Processing Flow

1. PDFs are uploaded and parsed
2. Text is chunked and embedded using `all-MiniLM-L6-v2`
3. Embeddings are stored in ChromaDB
4. User query is embedded and matched against stored vectors
5. Retrieved chunks are passed to the LLM
6. Answer is generated only from retrieved context
7. Source citations are appended from metadata

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python |
| **UI** | Streamlit |
| **Embeddings** | SentenceTransformers |
| **Vector DB** | ChromaDB |
| **LLM Interface** | LangChain |
| **LLM** | Google Gemini API |
| **Package Manager** | uv |

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

### 2️⃣ Create & activate environment (recommended)

```bash
uv venv
```

### 3️⃣ Install dependencies

```bash
uv pip install -r requirements.txt
```

Or with `uv`:

```bash
uv sync
```

### 4️⃣ Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> ⚠️ **Never commit `.env` files to GitHub.**

---

## ▶️ Run the Application

```bash
uv run streamlit run app.py
```

Or without uv:

```bash
streamlit run app.py
```

### Usage

1. Upload one or more PDFs
2. Click **Index Documents**
3. Ask questions about the papers
4. View answers with citations

---

## 🌱 Future Enhancements

- Conversational memory (multi-turn Q&A)
- Literature review matrix generation
- CSV / Excel export of extracted insights
- Source preview and page highlighting
- Deployment on Streamlit Cloud
