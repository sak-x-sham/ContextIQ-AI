# 🧠 ContextIQ AI — Your Personal Knowledge-Aware AI

> *“A conversation doesn’t start from zero — neither should AI.”*

ContextIQ AI is a smart Retrieval-Augmented (RAG) chatbot that learns from your uploaded documents and past interactions — allowing it to respond with **precision, context, and memory**, not just guesses.

It’s built to bridge the gap between **knowledge storage** and **intelligent reasoning**, making it more than a chatbot — it’s a **personal AI knowledge system.**

---

## 🌟 What Makes It Different?

Unlike typical LLM chatbots that forget everything after one message, **ContextIQ AI remembers.**  

It can:

- 📁 Ingest & index documents (PDF, Text, HTML)  
- 🔍 Perform semantic search using embeddings  
- 💬 Answer based on stored knowledge — not hallucination  
- 🧠 Retrieve relevant memory before generating a response  
- 🤝 Maintain context across chat sessions  

This creates a system closer to:

🔹 A research assistant  
🔹 A knowledge search engine  
🔹 A personal memory-powered AI  

Rather than a one-shot text generator.

---

## 🧩 Tech Architecture

| Layer | Technology | Purpose |
|-------|------------|---------|
| LLM | **Groq (LLama 3.1)** | Fast inference & response generation |
| Vector DB | **ChromaDB** | Stores embeddings + memory |
| Embeddings | **Sentence Transformers MiniLM-L6-v2** | Converts text → vector meaning |
| Frontend | **Streamlit** | Simple, fast interactive UI |
| Runtime | Python | Core logic |

This stack allows ContextIQ AI to combine **reasoning + memory**, enabling Retrieval-Augmented Generation (RAG).

---

## 🖼 Screenshots



| UI | Preview |
|----|---------|
| 🏠 Home UI | (<img width="1902" height="930" alt="image" src="https://github.com/user-attachments/assets/54b67add-a1cb-4210-843e-f0ebaaa0d44d" />
) |
| 💬 Chat Interface | (<img width="1911" height="940" alt="image" src="https://github.com/user-attachments/assets/d785fdfe-d361-4be5-8370-81bd05896e07" />
) |
| 📁 Document Upload | (<img width="1900" height="927" alt="image" src="https://github.com/user-attachments/assets/804fb544-694f-4ea8-9d38-4be2870ea61a" />
) |

---
## 🛠️ Future Enhancements

🚧 Coming next:

Vector-based session memory timeline

Cloud deployment + persistent user profiles

Multi-file context blending

Voice input + TTS output

---
## 👤 Author
Saksham Sharma
💡 Builder | Android Dev | AI Explorer | Data Analyst

📍 India

"I don't just use AI — I build it."
---

## 🚀 Getting Started

Clone and run:

```bash
git clone https://github.com/YOUR-USERNAME/ContextIQ-AI.git
cd ContextIQ-AI
pip install -r requirements.txt
streamlit run main.py
