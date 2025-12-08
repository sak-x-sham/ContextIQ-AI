# # rag_engine.py
# import os
# import google.generativeai as genai
# from chroma_rag import retrieve_context, store_message, store_document
#
# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in environment")
# genai.configure(api_key=API_KEY)
#
# # choose a working model available on your key:
# LLM_MODEL = "models/gemini-1.5-flash"  # change if your key lists other models
#
# def generate_rag_response(chat_id: str, user_message: str, temperature: float = 0.3, max_tokens: int = 512):
#     # 1) retrieve relevant context
#     context_docs = retrieve_context(chat_id, user_message, k=4)
#     context_text = "\n\n".join(context_docs) if context_docs else ""
#
#     # 2) build prompt
#     prompt = f"""You are a helpful assistant. Use the relevant context below to answer the user's query as accurately as possible.
#
# Relevant context (from previous messages or uploaded docs):
# {context_text}
#
# User Query:
# {user_message}
#
# Answer concisely and clearly. If context is not helpful, be honest and answer based on general knowledge.
# """
#
#     # 3) call Gemini
#     model = genai.GenerativeModel(LLM_MODEL)
#     response = model.generate_content(
#         prompt,
#         generation_config=genai.types.GenerationConfig(
#             temperature=temperature,
#             max_output_tokens=max_tokens
#         )
#     )
#     answer = response.text
#
#     # 4) store conversation in chroma (user then assistant)
#     store_message(chat_id, "user", user_message)
#     store_message(chat_id, "assistant", answer)
#
#     return answer

# rag_engine.py
import os
import google.generativeai as genai
from chroma_rag import retrieve_context, store_message




# def generate_rag_response(chat_id: str, user_message: str, temperature: float = 0.3, max_tokens: int = 512):
#     _ensure_configured()
#
#     # 1) Retrieve context
#     context_docs = retrieve_context(chat_id, user_message, k=4)
#     context_text = "\n\n".join(context_docs) if context_docs else ""
#
#     # 2) Build prompt
#     prompt = f"""You are a helpful assistant. Use the context below to answer the user's query.
#
# Relevant context:
# {context_text}
#
# User query:
# {user_message}
#
# Answer concisely and clearly.
# """
#
#     # 3) Call Gemini
#     model = genai.GenerativeModel(_MODEL)
#     response = model.generate_content(
#         prompt,
#         generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
#     )
#     answer = response.text
#
#     # 4) Store conversation messages in vector DB (as plain docs)
#     store_message(chat_id, "user", user_message)
#     store_message(chat_id, "assistant", answer)
#
#     return answer

# rag_engine.py for Gemini SDK 0.8.5
# rag_engine.py -- Compatible with google-generativeai==0.8.5

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
    context = retrieve_context(chat_id, query, k=5)

    prompt = f"""
You are an AI assistant using RAG.

Relevant stored context:
{ context if context else "No context found." }

User question: {query}

Answer only using the retrieved context when possible.
"""

    reply = generate_response(prompt)

    # store query + reply to memory
    store_message(chat_id, query, reply)

    return reply


