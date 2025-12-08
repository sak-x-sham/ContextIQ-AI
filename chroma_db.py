# import chromadb
# from chromadb.config import Settings
#
# chroma_client = chromadb.Client(
#     Settings(chroma_db_impl="duckdb+parquet", persist_directory="chroma_store")
# )
#
# collection = chroma_client.get_or_create_collection(
#     name="chat_memory",
#     metadata={"hnsw:space": "cosine"}
# )
#
# def add_message(chat_id, role, message, embedding):
#     collection.add(
#         ids=[f"{chat_id}_{role}_{hash(message)}"],
#         documents=[message],
#         metadatas=[{"chat_id": chat_id, "role": role}],
#         embeddings=[embedding]
#     )
#
# def query_context(embedding, chat_id, n=5):
#     results = collection.query(
#         query_embeddings=[embedding],
#         n_results=n,
#         where={"chat_id": chat_id}
#     )
#     return results["documents"]


import chromadb
from chromadb.utils import embedding_functions
import uuid

# ---- NEW CHROMA CLIENT (No Deprecation Warning) ----
client = chromadb.PersistentClient(path="chroma_db")

# Sentence Transformer Embedder
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_or_create_collection(chat_id):
    """Create a collection for each chat (or load if exists)."""
    collection_name = f"chat_{chat_id}"

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedder
        )
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedder
        )
    return collection


def store_message(chat_id, role, content):
    """Store message using Chroma RAG."""
    collection = get_or_create_collection(chat_id)

    doc_id = str(uuid.uuid4())  # unique msg id

    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[{"role": role}]
    )


def retrieve_context(chat_id, query, k=5):
    """Retrieve most relevant previous messages."""
    collection = get_or_create_collection(chat_id)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        if results and results["documents"]:
            return results["documents"][0]  # top-k list
    except:
        return []

    return []
