# main.py


import os
import uuid
from dotenv import load_dotenv
import streamlit as st

# load .env before importing rag_engine (rag_engine will lazy-check API)
load_dotenv()

from rag_engine import generate_rag_response
from chroma_rag import retrieve_context, list_collections, store_file_chunks, store_documents
from groq_client import generate_response


# Optional extractors
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# Validate API key exists (rag_engine will also check on call)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not set in .env")
    st.stop()

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("🤖  ContextIQ AI")
st.write("Talk to your personal AI assistant! Your chats are stored & help the model respond better.")

# -----------------------------
# Session initialization
# -----------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_name" not in st.session_state:
    st.session_state.chat_name = "Chat " + st.session_state.chat_id[:5]

# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 Memory Controls")

    if st.button("Show Memory"):
        cols = list_collections()
        st.write(cols)

    if st.button("Clear Memory"):
        import chromadb

        client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_data"))
        for c in client.list_collections():
            try:
                client.delete_collection(c.name)
            except Exception:
                pass
        st.session_state.messages = []
        st.success("All vector memory cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 Upload a PDF / TXT / HTML")

    uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "html"])
    if uploaded_file:
        fname = uploaded_file.name.lower()

        # Extract text safely
        full_text = ""
        if fname.endswith(".pdf"):
            if pdfplumber is None:
                st.error("pdfplumber not installed. Run pip install pdfplumber")
            else:
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    full_text = "\n".join(pages)

        elif fname.endswith(".txt"):
            raw = uploaded_file.read()
            full_text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)

        elif fname.endswith(".html") or fname.endswith(".htm"):
            if BeautifulSoup is None:
                st.error("beautifulsoup4 not installed. Run pip install beautifulsoup4")
            else:
                raw = uploaded_file.read()
                raw = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
                soup = BeautifulSoup(raw, "html.parser")
                full_text = soup.get_text(separator="\n")

        if full_text:
            added_chunks = store_file_chunks(st.session_state.chat_id, uploaded_file.name, full_text)
            st.success(f"Ingested {added_chunks} chunks from {uploaded_file.name}")

            # 🔍 DEBUG — Verify stored vector memory
            debug_preview = retrieve_context(st.session_state.chat_id, "test", k=5)
            st.write("🔎 Debug preview of stored docs:", debug_preview)

        else:
            st.warning("No text extracted from file.")

    st.markdown("---")
    st.markdown("### 💬 Ask something based on stored memory")

    with st.form("user_query_form"):
        user_query = st.text_area("Type your question:")
        submitted = st.form_submit_button("Send")

        if submitted and user_query.strip():
            st.session_state.last_query = user_query
            # optional: show spinner
            with st.spinner("Generating response..."):
                response = generate_rag_response(
                    st.session_state.chat_id,
                    user_query
                )
            # store & display
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.success("Response generated!")
            st.rerun()

    st.markdown("---")
    st.write("Chat name:")
    st.code(st.session_state.chat_name)
    st.write("Chat ID:")
    st.code(st.session_state.chat_id)
    st.markdown("---")
    st.markdown("### Retrieved context (preview)")

    last_query = st.session_state.get("last_query", "")
    if last_query:
        ctx = retrieve_context(st.session_state.chat_id, last_query, k=5)
        if ctx:
            for i, c in enumerate(ctx, 1):
                st.markdown(f"**{i}.** {c[:200]}...")
        else:
            st.write("No context retrieved.")
    else:
        st.write("No query yet.")

# -----------------------------
# Chat UI (original simple layout)
# -----------------------------
st.markdown("### Chat")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")

# -----------------------------
# Input
# -----------------------------
user_text = st.text_input("Ask anything:", key="chat_input")

if st.button("Send"):
    if user_text and user_text.strip():
        st.session_state.last_query = user_text
        answer = generate_rag_response(st.session_state.chat_id, user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

