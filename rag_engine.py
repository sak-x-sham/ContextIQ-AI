# rag_engine.py

import os
from chroma_rag import retrieve_context, store_message
from chroma_rag import retrieve_context, store_message
from groq_client import generate_response

def generate_rag_response(chat_id, query):

    # Retrieve relevant document chunks from ChromaDB
    context = retrieve_context(chat_id, query, k =5)


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


