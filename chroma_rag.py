# import chromadb
# from chromadb.utils import embedding_functions
# import uuid
# import os
# import pdfplumber
# from bs4 import BeautifulSoup
#
#
# # persistent local path
# CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")
#
# # New persistent client
# client = chromadb.PersistentClient(path=CHROMA_DIR)
#
# # Sentence-transformers embedder (local)
# embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L12-v2"
# )
#
# def get_or_create_collection(name: str):
#     # collection names must be simple; we will use "chat_<chat_id>" or "docs"
#     try:
#         return client.get_collection(name=name)
#     except Exception:
#         return client.create_collection(name=name, embedding_function=embedder)
#
# # --- Chat message ops ---
# def store_message(chat_id: str, role: str, content: str):
#     collection_name = f"chat_{chat_id}"
#     col = get_or_create_collection(collection_name)
#     col.add(
#         ids=[str(uuid.uuid4())],
#         documents=[content],
#         metadatas=[{"role": role}],
#     )
#
# def retrieve_context(chat_id: str, query: str, k: int = 4):
#     collection_name = f"chat_{chat_id}"
#     col = get_or_create_collection(collection_name)
#     try:
#         res = col.query(query_texts=[query], n_results=k)
#         # returns list-of-lists; return first list of documents
#         docs = res.get("documents", [[]])
#         return docs[0] if docs and docs[0] else []
#     except Exception:
#         return []
#
# def list_collections():
#     return [c.name for c in client.list_collections()]
#
# # --- Document ingest ops ---
# def store_documents(collection_name: str, docs: list, metadatas: list = None):
#     col = get_or_create_collection(collection_name)
#     ids = [str(uuid.uuid4()) for _ in docs]
#     if not metadatas:
#         metadatas = [{}] * len(docs)
#     col.add(ids=ids, documents=docs, metadatas=metadatas)
#
# def get_documents(collection_name: str, limit: int = 100):
#     col = get_or_create_collection(collection_name)
#     # no direct list API for documents, use query with empty query_texts to fetch stored docs
#     try:
#         res = col.get(include=["documents", "metadatas", "ids"])
#         return {
#             "ids": res.get("ids", []),
#             "documents": res.get("documents", []),
#             "metadatas": res.get("metadatas", []),
#         }
#     except Exception:
#         return {"ids": [], "documents": [], "metadatas": []}
#
# def clear_collection(collection_name: str):
#     # Deletes all items in the collection
#     col = get_or_create_collection(collection_name)
#     # get ids then delete
#     res = col.get(include=["ids"])
#     ids = res.get("ids", [])
#     if ids:
#         col.delete(ids=ids)
#
# def store_file(chat_id, uploaded_file):
#     ext = uploaded_file.name.split(".")[-1].lower()
#
#     if ext == "pdf":
#         text = extract_pdf(uploaded_file)
#     elif ext == "txt":
#         text = uploaded_file.read().decode("utf-8", errors="ignore")
#     elif ext == "html":
#         text = extract_html(uploaded_file)
#     else:
#         return ""
#
#     # split into chunks (safe for RAG)
#     chunks = chunk_text(text)
#
#     # store in chroma
#     col = get_collection(chat_id)
#
#     for i, chunk in enumerate(chunks):
#         col.add(
#             ids=[f"{chat_id}_file_{i}"],
#             documents=[chunk]
#         )
#
#     return text
#
# def extract_pdf(uploaded_file):
#     text = ""
#     with pdfplumber.open(uploaded_file) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() or ""
#     return text
#
#
# def extract_html(uploaded_file):
#     data = uploaded_file.read().decode("utf-8", errors="ignore")
#     soup = BeautifulSoup(data, "html.parser")
#     return soup.get_text(separator="\n")
#
#
# def chunk_text(text, max_len=500):
#     words = text.split()
#     chunks = []
#
#     for i in range(0, len(words), max_len):
#         chunk = " ".join(words[i:i+max_len])
#         chunks.append(chunk)
#
#     return chunks


# chroma_rag.py
import os
import uuid
from typing import List, Optional, Dict
import chromadb
from sentence_transformers import SentenceTransformer

# Persistent Chroma client
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_data")
client = chromadb.PersistentClient(path=CHROMA_DIR)

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
        client.delete_collection(name)
    except Exception:
        pass

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
def retrieve_context(chat_id: str, query: str, k: int = 5):
    col = get_or_create_collection(f"chat_{chat_id}")
    try:
        result = col.query(query_texts=[query], n_results=k)
        docs = result.get("documents", [[]])[0]
        return [d for d in docs if d.strip()]
    except Exception as e:
        print("Retrieval Error:", e)
        return []

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

