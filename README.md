<div align="center">

# 🤖 HRBot — AI-Powered HR Assistant

**An open-source, privacy-first HR chatbot that answers employee questions from your company's own policy documents — no external API keys required.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 📌 Project Description

**HRBot** is an LLM-powered HR assistant built with **Retrieval-Augmented Generation (RAG)**. It lets employees ask natural-language questions about company policies, leave entitlements, performance reviews, and more — and get instant, accurate answers sourced directly from your uploaded HR documents (PDFs, DOCX, TXT).

All AI inference runs **100% locally** using open-source models from HuggingFace. No data is sent to OpenAI, Google, or any third-party API.

> **Use case:** An employee asks *"How many days of annual leave am I entitled to?"* — HRBot searches the uploaded employee handbook and responds with the exact relevant passage.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Document Ingestion** | Upload PDF/DOCX/TXT HR policy documents via the web UI |
| 🔍 **Semantic Search** | ChromaDB vector store with `all-MiniLM-L6-v2` embeddings for fast similarity retrieval |
| 🤖 **Local LLM** | Runs `google/flan-t5-small` (default) or `microsoft/DialoGPT-medium` — no API key needed |
| 💬 **Chat Interface** | Clean Streamlit UI with quick-action buttons and expandable example prompts |
| 🗄️ **SQLite Backend** | Tracks employees, leave requests, candidates, and performance reviews |
| 🔒 **Privacy-First** | All processing is on-device; documents never leave your machine |
| ⚙️ **Configurable Models** | Swap LLM models in `config.py` to suit your hardware (CPU / GPU) |
| 🔄 **Persistent Vector DB** | Indexed documents survive restarts; re-upload only when policies change |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                      │
│              (app.py — chat + file upload)               │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         RAG Pipeline         │
          │                             │
          │  ┌──────────┐  ┌─────────┐  │
          │  │  Loader  │  │Retriever│  │
          │  │(loader.py│  │(retriever│ │
          │  │ PyMuPDF) │  │  .py)   │  │
          │  └────┬─────┘  └────┬────┘  │
          │       │             │        │
          │  ┌────▼─────────────▼────┐  │
          │  │      Embedder         │  │
          │  │   (embedder.py)       │  │
          │  │  SentenceTransformers │  │
          │  └──────────┬────────────┘  │
          │             │               │
          │  ┌──────────▼────────────┐  │
          │  │   ChromaDB Vector DB  │  │
          │  │  (vectorstore/ dir)   │  │
          │  └──────────┬────────────┘  │
          │             │               │
          │  ┌──────────▼────────────┐  │
          │  │   Local LLM (HF)      │  │
          │  │  flan-t5 / DialoGPT   │  │
          │  └───────────────────────┘  │
          └─────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │    SQLite Database          │
          │  employees / leave /        │
          │  performance / candidates   │
          └─────────────────────────────┘
```

**Flow:**
1. User uploads a PDF → `loader.py` extracts text and splits it into 500-character chunks.
2. `embedder.py` converts chunks into vectors using `all-MiniLM-L6-v2` and stores them in ChromaDB.
3. User types a question → `retriever.py` performs semantic similarity search to find the top-3 relevant chunks.
4. Those chunks are injected into an HR-specific prompt template and passed to the local LLM.
5. The LLM generates an answer that is displayed in the Streamlit chat UI.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | [HuggingFace Transformers](https://huggingface.co/docs/transformers) — `flan-t5-small`, `DialoGPT-medium` |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **RAG Orchestration** | [LangChain](https://www.langchain.com/) |
| **PDF Parsing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |
| **Database** | SQLite 3 (built-in) |
| **Language** | Python 3.9+ |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- (Optional) A CUDA-capable GPU for faster inference

### Step 1 — Clone the repository

```bash
git clone https://github.com/K-vino/HR-chatbot.git
cd HR-chatbot
```

### Step 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `torch` and `transformers` are large packages (~1-3 GB). The first run will also download the selected HuggingFace model. Ensure you have a stable internet connection and sufficient disk space.

### Step 4 — Configure environment variables (optional)

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Step 5 — Run the application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 📦 Requirements.txt Explained

```text
# ── UI Framework ─────────────────────────────────────────
streamlit            # Web UI for the chatbot
python-dotenv        # Load environment variables from .env

# ── LangChain & LLM Orchestration ────────────────────────
langchain>=0.1.16    # RAG pipeline & chain management
langchain-community  # Community integrations (HuggingFacePipeline, Chroma, etc.)
transformers>=4.30.0 # HuggingFace model loading (flan-t5, DialoGPT)
torch                # Deep learning runtime (CPU or CUDA)
accelerate           # Optimised model loading
bitsandbytes         # 8-bit model quantisation (reduces VRAM usage)

# ── Embeddings & Vector Store ─────────────────────────────
sentence-transformers # all-MiniLM-L6-v2 embedding model
chromadb             # Local vector database for semantic search
tiktoken             # Token counting (used internally by LangChain)

# ── File Parsing ──────────────────────────────────────────
PyMuPDF              # PDF text extraction (fitz)
python-docx          # DOCX file reading
unstructured         # Advanced parsing for diverse file types

# ── Text Processing ───────────────────────────────────────
beautifulsoup4       # HTML/XML parsing
lxml                 # XML/HTML parser backend
spacy>=3.6.0         # NLP utilities
textblob             # Sentiment analysis for performance reviews

# ── Database & Security ───────────────────────────────────
bcrypt               # Password hashing
pandas               # Data manipulation & CSV export

# ── Utilities ─────────────────────────────────────────────
requests             # HTTP requests
numpy                # Numerical operations
```

---

## 🖥️ Usage

### 1. Upload an HR Document

1. Launch the app with `streamlit run app.py`.
2. In the **left sidebar**, click **"Browse files"** and select a PDF (employee handbook, leave policy, etc.).
3. HRBot processes and indexes the document — this takes ~30 seconds on first run as the embedding model downloads.

### 2. Ask a Question

Type your question in the chat input and press **"🔎 Ask HRBot"**:

```
What is the company's remote work policy?
How many days of annual leave do I get?
How do I submit a maternity leave request?
```

### 3. Use Quick-Action Buttons

Click **📋 Leave Policies**, **💰 Benefits Info**, or **📞 Contact HR** for instant pre-built queries.

### 4. Switch LLM Models

Edit `config.py` to change the active model:

```python
DEFAULT_LLM = "qa_optimized"  # options: "lightweight", "conversational", "qa_optimized"
```

---

## 📁 Project Structure

```
HR-chatbot/
├── app.py               # Main Streamlit application entry point
├── config.py            # Model, database & app configuration
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
├── .env.example         # Environment variable template
├── LICENSE              # MIT License
├── CONTRIBUTING.md      # Contribution guidelines
├── README.md            # This file
│
├── utils/               # Core AI/RAG utilities
│   ├── __init__.py
│   ├── loader.py        # PDF text extraction & chunking (PyMuPDF)
│   ├── embedder.py      # Embedding generation & ChromaDB management
│   └── retriever.py     # LLM loading, prompt templates & QA chain
│
├── data/                # Uploaded HR documents (auto-created, git-ignored)
│   └── hrbot.db         # SQLite database
│
├── vectorstore/         # Persisted ChromaDB vector index (auto-created, git-ignored)
│
├── models/              # Cached HuggingFace model weights (auto-created, git-ignored)
│
├── logs/                # Application logs (auto-created, git-ignored)
│
└── screenshots/         # UI screenshots for documentation
    └── ...
```

---

## 📸 Screenshots

> **Note:** Upload screenshots of your running application to the `screenshots/` folder and update the paths below.

| Main Chat Interface | Document Upload | Sample Response |
|---|---|---|
| *(add screenshot)* | *(add screenshot)* | *(add screenshot)* |

---

## 💬 Example Questions Users Can Ask

Once you have uploaded an HR policy document, try asking:

**Leave & Time Off**
- *"What are the different types of leave available?"*
- *"How do I apply for maternity/paternity leave?"*
- *"How many sick days am I entitled to per year?"*
- *"Can I carry over unused annual leave to the next year?"*

**Benefits & Compensation**
- *"What health insurance benefits does the company provide?"*
- *"Is there a provident fund or retirement savings scheme?"*
- *"What is the performance bonus structure?"*

**Work Policies**
- *"What are the official working hours?"*
- *"What is the company's remote work / work-from-home policy?"*
- *"What is the dress code policy?"*

**HR Processes**
- *"How does the performance review process work?"*
- *"What is the onboarding process for new employees?"*
- *"How do I raise a grievance with HR?"*
- *"Who do I contact regarding payroll issues?"*

---

## 🔮 Future Improvements

- [ ] **Multi-document support** — query across multiple uploaded policies simultaneously
- [ ] **Authentication** — role-based access (Admin / HR / Employee) with bcrypt login
- [ ] **Conversation history** — maintain multi-turn dialogue context
- [ ] **DOCX & TXT ingestion** — extend loader to non-PDF formats
- [ ] **GPU acceleration toggle** — one-click switch between CPU and CUDA inference
- [ ] **Answer citations** — display the exact source passage used to generate each answer
- [ ] **Admin dashboard** — manage uploaded documents, view query analytics
- [ ] **Email notifications** — notify HR when high-priority questions are asked
- [ ] **Docker support** — containerised deployment with `docker-compose`
- [ ] **REST API** — FastAPI endpoint so HRBot can be embedded in other tools (Slack, Teams)
- [ ] **Multilingual support** — answer questions in the employee's preferred language

---



**Quick start:**

```bash
# 1. Fork the repo and create your branch
git checkout -b feature/your-feature-name

# 2. Make your changes and add tests where applicable

# 3. Commit using a descriptive message
git commit -m "feat: add multi-document query support"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```



</div>
