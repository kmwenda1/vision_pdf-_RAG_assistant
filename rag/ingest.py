# ─────────────────────────────────────────────────────────
# rag/ingest.py — PDF Reading & Chunking
#
# Copied from insurance-rag and kept identical.
# Takes a PDF file path → extracts text page-by-page →
# cuts into overlapping chunks → returns chunk dicts.
# ─────────────────────────────────────────────────────────

import os
import pdfplumber


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Opens a PDF and extracts the text from every page.

    Returns a list of page dicts:
        {"page_num": 3, "text": "...", "source": "policy.pdf"}

    WHY PAGE BY PAGE?
        We track page numbers so every answer can cite its exact source page.
    """
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text": text.strip(),
                    "source": os.path.basename(file_path),
                })

    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 200) -> list[dict]:
    """
    Cuts each page's text into overlapping chunks.

    Parameters:
        chunk_size: max characters per chunk (800 captures full clauses)
        overlap:    characters shared between adjacent chunks (200 = 25%)
                    prevents splitting a key sentence across two chunks.

    Returns chunk dicts:
        {"text": "...", "page_num": 3, "source": "policy.pdf",
         "chunk_id": "policy.pdf_page3_chunk0"}
    """
    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        source = page["source"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "source": source,
                    "chunk_id": f"{source}_page{page_num}_chunk{chunk_index}",
                })
                chunk_index += 1

            start = end - overlap
            if start >= len(text):
                break

    return chunks


def is_scanned_pdf(pages: list[dict]) -> bool:
    """
    Heuristic: if average chars-per-page < 100 the PDF is probably scanned.
    Warns the user rather than silently failing.
    """
    if not pages:
        return True
    total_text = " ".join(p["text"] for p in pages)
    return len(total_text) / len(pages) < 100


def ingest_pdf(file_path: str) -> dict:
    """
    Main entry point. Runs all three steps and returns:
        {"chunks": [...], "page_count": N, "chunk_count": N,
         "is_scanned": bool, "filename": "policy.pdf"}
    """
    filename = os.path.basename(file_path)
    pages = extract_text_from_pdf(file_path)
    scanned = is_scanned_pdf(pages)
    chunks = chunk_pages(pages)

    return {
        "chunks": chunks,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "is_scanned": scanned,
        "filename": filename,
    }
