import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    model = genai.GenerativeModel("text-embedding-004")
    result = model.embed_content(text)
    return result.embedding
