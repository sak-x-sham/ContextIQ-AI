# chroma_rag.py

import os
import uuid
from typing import List, Optional, Dict
import chromadb
from sentence_transformers import SentenceTransformer

# Persistent Chroma client
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# -------------------------
# Embedding model (local, stable)
# -------------------------
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Downloads once locally
_embed_model = None

def _ensure_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for a list of texts."""
    model = _ensure_embed_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [list(map(float, e.tolist())) for e in embs]

# -------------------------
# Collection helpers
# -------------------------
def get_or_create_collection(name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name)

def list_collections() -> List[str]:
    return [c.name for c in client.list_collections()]

def delete_collection(name: str):
    try:
        client.delete_collection(name=name)
        return True
    except Exception as e:
        print(f"Delete Error: {e}")
        return False

# -------------------------
# Document / message storage
# -------------------------
def store_documents(collection_name: str, docs: List[str], metadatas: Optional[List[dict]] = None):
    if not docs:
        return
    col = get_or_create_collection(collection_name)
    embeddings = embed_texts(docs)
    ids = [str(uuid.uuid4()) for _ in docs]
    if metadatas is None:
        metadatas = [{} for _ in docs]
    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

def store_message(chat_id: str, role: str, content: str):
    store_documents(f"chat_{chat_id}", [content], metadatas=[{"role": role}])

def get_documents(collection_name: str) -> Dict[str, List]:
    col = get_or_create_collection(collection_name)
    try:
        res = col.get(include=["ids", "documents", "metadatas"])
        return {
            "ids": res.get("ids", []),
            "documents": res.get("documents", []),
            "metadatas": res.get("metadatas", []),
        }
    except Exception:
        return {"ids": [], "documents": [], "metadatas": []}

# -------------------------
# Retrieval
# -------------------------
def retrieve_chat_context(chat_id: str, query: str, k: int = 3):

    col = get_or_create_collection(f"chat_{chat_id}")

    try:

        query_embedding = embed_texts([query])[0]

        result = col.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        docs = result.get("documents", [[]])[0]

        return [d for d in docs if d.strip()]

    except Exception as e:

        print("Chat Retrieval Error:", e)

        return []

def retrieve_document_context(query: str, k: int = 5):

    col = get_or_create_collection("documents")

    try:

        query_embedding = embed_texts([query])[0]

        result = col.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        docs = result.get("documents", [[]])[0]

        return [d for d in docs if d.strip()]

    except Exception as e:

        print("Document Retrieval Error:", e)

        return []

def retrieve_context(
    chat_id: str,
    query: str,
    k: int = 5
):
    chat_context = retrieve_chat_context(
        chat_id,
        query,
        k=3
    )

    document_context = retrieve_document_context(
        query,
        k=5
    )

    combined = []

    combined.extend(document_context)

    combined.extend(chat_context)

    return combined
# -------------------------
# File helpers
# -------------------------
def chunk_text(text: str, max_words: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def store_file_chunks(chat_id: str, filename: str, text: str) -> int:
    chunks = chunk_text(text, max_words=200)
    if not chunks:
        return 0
    metadatas = [{"source": filename} for _ in chunks]
    store_documents(f"chat_{chat_id}", chunks, metadatas)
    return len(chunks)


def clear_collection(collection_name):
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        print(f"Delete Error: {e}")
        return False
