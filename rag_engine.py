# rag_engine.py

import os
import google.generativeai as genai
from chroma_rag import retrieve_context, store_message

# lazy API setup
_model = None
_configured = False


def _ensure_model():
    global _configured, _model
    if not _configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("⚠️ GEMINI_API_KEY is missing. Set it in environment.")

        genai.configure(api_key=api_key)

        # Create Model instance (this is REQUIRED in 0.8.5)
        _model = genai.GenerativeModel("gemini-2.5-flash")

        _configured = True

    return _model

from chroma_rag import retrieve_context, store_message
from groq_client import generate_response

def generate_rag_response(chat_id, query):

    # Retrieve relevant document chunks from ChromaDB
    context = retrieve_context(chat_id, query, k =5)

    # Debug: print retrieved chunks in terminal
    print("\n===== RETRIEVED CONTEXT =====")
    print(context)
    print("=============================\n")

    # Convert list of chunks into one text block
    context_text = "\n\n".join(context)

    prompt = f"""
You are an AI assistant using RAG.

Relevant Context:
{context_text if context_text else "No context found."}

User Question:
{query}

Answer using the provided context whenever possible.
"""

    # Generate response from LLM
    reply = generate_response(prompt)

    # Store conversation in memory
    store_message(chat_id, "user", query)
    store_message(chat_id, "assistant", reply)

    return reply


