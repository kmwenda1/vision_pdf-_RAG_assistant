# ─────────────────────────────────────────────────────────
# rag/retriever.py — Semantic Chunk Search
#
# Supports optional collection_name to isolate per session.
# Embeds the query, searches ChromaDB, returns scored chunks.
# ─────────────────────────────────────────────────────────

from typing import Optional
from rag.embedder import embed_texts, get_collection


def retrieve(query: str, top_k: int = 8, collection_name: Optional[str] = None) -> list[dict]:
    """
    Finds the most relevant document chunks for a given query in collection_name.

    Returns list of:
        {"text": "...", "page_num": 3, "source": "policy.pdf", "score": 0.87}

    score = 1 - cosine_distance  (higher = more relevant)
    """
    collection = get_collection(collection_name)

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for i in range(len(documents)):
        chunks.append({
            "text":     documents[i],
            "page_num": metadatas[i].get("page_num", "?"),
            "source":   metadatas[i].get("source", "unknown"),
            "score":    round(1 - distances[i], 3),
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def retrieve_with_confidence(query: str, top_k: int = 8, collection_name: Optional[str] = None) -> dict:
    """
    Same as retrieve(), plus a confidence label based on the top score.

    Confidence thresholds:
        HIGH   score > 0.25
        MEDIUM score > 0.15
        LOW    score ≤ 0.15

    Returns:
        {"chunks": [...], "confidence": "HIGH", "top_score": 0.87, "warning": None}
    """
    chunks = retrieve(query, top_k, collection_name=collection_name)

    if not chunks:
        return {
            "chunks": [],
            "confidence": "NONE",
            "top_score": 0,
            "warning": "No documents have been indexed yet. Please upload a PDF first.",
        }

    top_score = chunks[0]["score"]

    if top_score > 0.25:
        confidence, warning = "HIGH", None
    elif top_score > 0.15:
        confidence, warning = "MEDIUM", "⚠️ Moderate confidence — verify against the source document."
    else:
        confidence, warning = "LOW", "⚠️ Low confidence. This topic may not be covered in the uploaded documents."

    return {
        "chunks":     chunks,
        "confidence": confidence,
        "top_score":  top_score,
        "warning":    warning,
    }
