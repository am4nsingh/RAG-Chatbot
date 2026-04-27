# RAG Chatbot

Two Retrieval-Augmented Generation (RAG) pipelines built with LangChain and Ollama, running fully locally.

---

## Projects

### 1. FAQ Bot (`faqbot.ipynb` + `faqbot_ui.py`)
A conversational chatbot over a local document corpus using vector search and a Streamlit UI.

### 2. Multimodal RAG (`MultiModal.ipynb`)
Advanced RAG pipeline that processes **text, tables, and images** from PDFs.

**Pipeline stages:**
- PDF extraction via `unstructured` (hi-res strategy, table structure inference, image extraction)
- Parallel summarization: images via `llava:7b` (vision), tables via `llama3` (text)
- Parent-child chunking with `RecursiveCharacterTextSplitter`
- Hybrid retrieval: BM25 + ChromaDB vector search (`EnsembleRetriever`)
- Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`)
- Multi-query expansion for improved recall
- Vision-aware answering: `llava:7b` invoked when images are in context, `llama3` otherwise
- Persistent docstore (`LocalFileStore`) + persistent ChromaDB — no re-indexing on restart
- Disk-backed summary cache (SHA-256 keyed) with batched writes
- LLM-as-judge evaluation with deterministic retrieval hit-rate metric

---

## Setup

### Requirements
- [Ollama](https://ollama.com) running locally on `http://127.0.0.1:11434`
- Models pulled:
  ```
  ollama pull llava:7b
  ollama pull llama3
  ```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables (optional overrides)
| Variable | Default | Description |
|---|---|---|
| `MMRAG_ROOT` | current directory | Project root |
| `MMRAG_PDF` | `<root>/sample.pdf` | Path to PDF to index |
| `MMRAG_CHROMA` | `<root>/chromadb` | ChromaDB persist dir |
| `MMRAG_CACHE` | `<root>/cache` | Summary cache dir |
| `MMRAG_DOCSTORE` | `<root>/docstore` | Docstore persist dir |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `MMRAG_DOMAIN` | `the provided document` | Domain label in system prompt |

---

## Usage

### Multimodal RAG
Open `MultiModal.ipynb` and run cells top to bottom. After indexing:

```python
ask("What is the method of joints?")
ask("Summarize the load table on page 5.")
```

`chat_history` persists across cell executions in the same kernel session.

---

## Directory Structure
```
├── faqbot.ipynb          # FAQ RAG pipeline
├── faqbot_ui.py          # Streamlit UI for FAQ bot
├── MultiModal.ipynb      # Multimodal RAG pipeline (text + tables + images)
├── requirements.txt      # Python dependencies
└── README.md
```
