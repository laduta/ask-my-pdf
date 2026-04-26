# 📄 Ask My PDF — RAG-Powered Document Chat

A production-grade Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and chat with it using AI.

Built with **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini**.

![Ask My PDF Demo](https://laduta.github.io/ask-my-pdf/)

---

## ✨ Features

- 📄 Upload any PDF (text-based or scanned)
- 🔍 Hybrid search — semantic + keyword (BM25)
- 🧠 Parent-child chunking for better context
- 🔬 Answer grounding — no hallucination
- 🖼️ OCR support for scanned documents
- ⚡ Persistent vector store — no reprocessing on restart

---

## 🏗️ Architecture
Browser (HTML/JS)
↓
FastAPI Backend
↓
RAG Pipeline
├── PDF Loader (PyPDF + OCR fallback)
├── Parent-Child Chunker
├── Hybrid Retriever (ChromaDB + BM25)
└── Answer Grounding Check
↓
Google Gemini (LLM + Embeddings)
---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))
- Tesseract OCR (for scanned PDFs)

```bash
sudo apt install tesseract-ocr poppler-utils -y
```

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ask-my-pdf.git
cd ask-my-pdf

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Configuration

Create a `.env` file inside the `backend/` folder:
GOOGLE_API_KEY=your-gemini-api-key-here

### Run

```bash
cd backend
uvicorn main:app --reload
```

Open `frontend/index.html` in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload and process a PDF |
| POST | `/ask` | Ask a question about the PDF |

### Example

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python |
| RAG Pipeline | LangChain |
| Vector Store | ChromaDB |
| Embeddings | Google Gemini Embedding |
| LLM | Google Gemini 2.5 Flash |
| Keyword Search | BM25 (rank_bm25) |
| OCR | Tesseract + pdf2image |
| Frontend | HTML, CSS, JavaScript |

---

## 📚 What I Learned

- How RAG pipelines work end to end
- Parent-child chunking strategy
- Hybrid search combining semantic + keyword retrieval
- Answer grounding to prevent hallucination
- OCR integration for scanned documents
- Building and deploying a FastAPI backend

---

## 🔮 Roadmap

- [ ] Multi-document support
- [ ] User authentication
- [ ] Conversation memory
- [ ] Document management dashboard
- [ ] Fine-tuned embeddings

---

## 👤 Author

**Abubakar Jibrin**
- GitHub: [@laduta](https://github.com/laduta)
- LinkedIn: [Abubakar Jibrin](https://www.linkedin.com/in/abubakar-jibrin-996437279)

---

## 📄 License

MIT License — feel free to use this project as a starting point.
