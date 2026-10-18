import os
from typing import Optional
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from fastembed import TextEmbedding

# ─────────────────────────────────────────────────────────
# rag/embedder.py — Text → Vectors → ChromaDB
#
# Uses fastembed (local, free) + ChromaDB (local, persistent).
# Supports per-session collection names for tenant/session isolation.
# ─────────────────────────────────────────────────────────

print("Loading embedding model (fastembed)...")
EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Embedding model ready.")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
CHROMA_CLIENT = chromadb.PersistentClient(path=DB_PATH)
DEFAULT_COLLECTION_NAME = "documents"


def get_collection(collection_name: Optional[str] = None):
    name = collection_name or DEFAULT_COLLECTION_NAME
    return CHROMA_CLIENT.get_or_create_collection(
        name=name,
        metadata={"description": "PDF document chunks with embeddings"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert texts to embedding vectors via fastembed (runs locally)."""
    embeddings_gen = EMBEDDING_MODEL.embed(texts)
    return [emb.tolist() for emb in embeddings_gen]


def store_chunks(chunks: list[dict], collection_name: Optional[str] = None) -> int:
    """Embed and store chunks in ChromaDB. Returns count stored."""
    collection = get_collection(collection_name)
    ids       = [c["chunk_id"] for c in chunks]
    texts     = [c["text"]     for c in chunks]
    metadatas = [{"page_num": c["page_num"], "source": c["source"]} for c in chunks]

    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    batch_size = 100
    stored = 0
    for i in range(0, len(chunks), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        stored += len(ids[i:end])
    return stored


def chunk_already_exists(chunk_id: str, collection_name: Optional[str] = None) -> bool:
    collection = get_collection(collection_name)
    result = collection.get(ids=[chunk_id])
    return len(result["ids"]) > 0


def get_collection_stats(collection_name: Optional[str] = None) -> dict:
    name = collection_name or DEFAULT_COLLECTION_NAME
    collection = get_collection(name)
    return {"total_chunks": collection.count(), "collection_name": name}


def delete_session_collection(collection_name: str) -> bool:
    """Deletes only the specified session collection from ChromaDB."""
    try:
        CHROMA_CLIENT.delete_collection(name=collection_name)
        return True
    except Exception:
        return False
