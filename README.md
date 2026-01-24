---
title: RAG Document Q&A Assistant
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# RAG Document Q&A Assistant

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Ireliaaaaaa/rag-document-qa)**

A **Knowledge-Enhanced RAG** (Retrieval-Augmented Generation) system that lets you chat with your PDF documents. Upload any PDF and ask questions—the assistant uses document knowledge when relevant, and general knowledge when not. Unlike traditional RAG systems that fail when queries don't match documents, this assistant always provides helpful responses.

## Features

- **Multi-PDF Support** - Index multiple documents from a folder with incremental indexing
- **Knowledge-Enhanced Mode** - Always responds helpfully; uses documents when relevant, general knowledge otherwise
- **Web Chat Interface** - Clean Gradio UI with conversation history
- **Document Management** - Upload, index, and manage PDFs through the UI
- **Debug Mode** - Optional display of similarity scores and source chunks
- **Source Tracking** - Tracks which document and page each answer comes from

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Google Gemini 2.5 Pro |
| **Embeddings** | Google text-embedding-004 |
| **Vector Store** | ChromaDB (Cosine Similarity) |
| **PDF Processing** | pypdf (lightweight) |
| **Web UI** | Gradio |
| **Tokenization** | tiktoken (cl100k_base) |
| **Evaluation** | LLM-as-a-Judge (Gemini 2.5 Flash) |

## Project Structure

```
My-LLM-APP/
├── app.py                      # Gradio web interface
├── main.py                     # CLI entry point
├── run_evaluation.py           # LLM-as-a-Judge evaluation runner
├── src/
│   ├── pdf_processor/          # PDF parsing & text chunking
│   │   ├── pdf_parser.py       # PDF download and extraction
│   │   └── text_chunker.py     # Text chunking (300-500 tokens)
│   ├── vector_store/           # Vector storage
│   │   ├── chroma_store.py     # ChromaDB operations
│   │   └── gemini_embedding.py # Gemini embedding function
│   ├── rag_qa/                 # Q&A engine
│   │   └── qa_engine.py        # Knowledge-Enhanced QA
│   ├── indexer/                # Document indexer
│   │   └── document_indexer.py # Multi-PDF batch indexing
│   └── evaluation/             # Quality evaluation
│       └── evaluator.py        # LLM-as-a-Judge scoring
├── data/
│   ├── pdfs/                   # PDF storage folder
│   └── indexed_files.json      # Index tracking (auto-generated)
├── chroma_db/                  # Vector database (auto-generated)
├── requirements.txt            # Dependencies
├── .env.example                # Environment variables template
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google API key
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 3. Run the Web App

```bash
python app.py
```

Open the URL shown (typically `http://127.0.0.1:7860`) in your browser.

### 4. Upload Documents & Chat

1. Go to **"Manage Documents"** tab
2. Upload PDF files or click **"Index New Files"** to index PDFs from `data/pdfs/`
3. Go to **"Chat"** tab
4. Start asking questions!

## Usage

### Web Interface (Recommended)

```bash
python app.py
```

Features:
- **Chat Tab**: Conversational Q&A with your documents
- **Manage Documents Tab**: Upload, index, and manage PDFs
- **Debug Mode**: Toggle to see retrieval details and similarity scores

### Command Line Interface

```bash
# Index a PDF and start interactive Q&A
python main.py --pdf "data/pdfs/report.pdf" --reindex

# Ask a single question
python main.py --pdf "data/pdfs/report.pdf" --question "What is the total revenue?"

# Run evaluation with preset questions
python main.py --pdf "data/pdfs/report.pdf" --evaluate
```

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User Query │────▶│   Retrieve   │────▶│   Filter    │
└─────────────┘     │  Top-K Docs  │     │ by Threshold│
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │   Generate   │◀───────────┘
                    │   Response   │
                    └──────┬───────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
    ┌──────▼──────┐                ┌───────▼───────┐
    │  Relevant   │                │  No Relevant  │
    │  Docs Found │                │  Docs Found   │
    └──────┬──────┘                └───────┬───────┘
           │                               │
    ┌──────▼──────┐                ┌───────▼───────┐
    │ Answer with │                │  Answer with  │
    │  Citations  │                │General Knowledge│
    └─────────────┘                └───────────────┘
```

**Key Difference from Traditional RAG:**
- Traditional RAG: "I don't have information about that in the documents."
- Knowledge-Enhanced RAG: Provides helpful response using general knowledge, clearly indicating when information isn't from documents.

## Configuration

### Text Chunking
- `min_tokens`: 300
- `max_tokens`: 500

### Retrieval
- `k`: 5 (top-k documents to retrieve)
- `threshold`: 0.6 (similarity threshold for relevance)

### LLM
- `model`: gemini-2.5-pro
- `temperature`: 0.7 (for natural conversation)

## Evaluation

The system includes an LLM-as-a-Judge evaluation framework:

```bash
# Run full evaluation
python run_evaluation.py

# Quick test (5 questions)
python run_evaluation.py --quick
```

Evaluation dimensions:
- **Faithfulness**: Does the answer follow the retrieved context?
- **Relevancy**: Does it address the user's question?
- **Citation Quality**: Are sources cited naturally?

## Future Improvements

### Planned Features
- [ ] **Conversation Memory** - Context across multiple turns
- [ ] **Streaming Responses** - Real-time token streaming
- [ ] **More File Types** - Word, TXT, Markdown support
- [ ] **REST API** - FastAPI endpoint for integration
- [ ] **Chat Export** - Save conversations as JSON/PDF

### Long-term Goals
- [ ] Cloud deployment (Google Cloud Run)
- [ ] User authentication & multi-tenancy
- [ ] Hybrid search (vector + keyword)
- [ ] Multi-language support
- [ ] Analytics dashboard

## Troubleshooting

**"No documents indexed"**
- Go to "Manage Documents" tab and click "Index New Files" or upload PDFs

**API errors**
- Check your `GOOGLE_API_KEY` in `.env`
- Verify you have API quota at [Google AI Studio](https://aistudio.google.com/)

**Slow PDF processing**
- The `hi_res` strategy is thorough but slower; first-time indexing may take a while

## License

MIT License

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) for LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Gradio](https://gradio.app/) for the web interface
- [Unstructured](https://unstructured.io/) for PDF processing
