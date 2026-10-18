# PDF & Image Document Assistant (RAG)

A Retrieval-Augmented Generation application that ingests PDFs, indexes them with local embeddings and ChromaDB, and allows users to query them via text or uploaded photos (e.g. equipment labels, error screens, parts, receipts, claim photos).

## Features

- **Text PDF Ingestion**: Extracts text page-by-page with `pdfplumber`, creates overlapping chunks, and embeds them locally with `fastembed` (`BAAI/bge-small-en-v1.5`) into `chromadb`.
- **Image Input Pipeline**: 
  - Photos are **never directly embedded or searched**.
  - Analyzed by a vision model (`gemini-3.8-flash` via Google GenAI) to extract verbatim OCR text, descriptions, entities, and document search queries.
  - Multi-query search via the existing ChromaDB retrieval engine with chunk score deduplication.
  - **Relevance Gate**: If top retrieved chunk similarity does not clear the threshold, answer generation is aborted to prevent hallucinations.
  - **Answer Generation**: Synthesizes answers using Groq (`qwen/qwen3.8-27b`) citing exact document names and page numbers.
- **Streamlit Web Interface**: Complete with tabs for text-only queries and photo+question analysis, document upload, and debug inspection.

## Quickstart

### 1. Configure Environment
Copy `.env.example` to `.env` and provide your keys:
```bash
cp .env.example .env
```
Ensure you have `GROQ_API_KEY` and `GEMINI_API_KEY` set in `.env`.

### 2. Run with Streamlit
Activate the virtual environment and start the app:
```bash
source venv/bin/activate
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Running Tests
To run unit and integration tests:
```bash
venv/bin/pytest tests/ -v
```
To run the live pipeline verification script:
```bash
venv/bin/python tests/verify_pipeline.py
```
