# groq_client.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")

client = Groq(api_key=GROQ_API_KEY)

DEFAULT_MODEL = "openai/gpt-oss-20b"

def generate_response(prompt: str):
    """Generate an AI response using Groq."""
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
