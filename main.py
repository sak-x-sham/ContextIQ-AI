# main.py


import os
import uuid

import streamlit as st
from dotenv import load_dotenv

# load .env before importing rag_engine (rag_engine will lazy-check API)
from dotenv import load_dotenv
from sympy import false

load_dotenv()  # harmless if .env doesn't exist

from rag_engine import generate_rag_response
from chroma_rag import retrieve_context, list_collections, store_file_chunks
from chat_store import *

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
    st.error(
        "GEMINI_API_KEY environment variable is missing."
    )
    st.stop()

st.set_page_config(page_title="ContextIQ AI", layout="wide")
init_db()
st.title("🤖  ContextIQ AI")
st.write("Talk to your personal AI assistant! Your chats are stored & help the model respond better.")

# -----------------------------
# Session initialization
# -----------------------------

if "chat_id" not in st.session_state:

    existing_chats = load_chats()

    # No chats exist in database yet
    if not existing_chats:

        first_chat_id = str(uuid.uuid4())
        first_chat_name = f"Chat {first_chat_id[:5]}"

        create_chat(
            first_chat_id,
            first_chat_name
        )

        st.session_state.chat_id = first_chat_id
        st.session_state.chat_name = first_chat_name
        st.session_state.messages = []

    # Load first available chat
    else:

        first_chat_id, first_chat_name = existing_chats[0]

        st.session_state.chat_id = first_chat_id
        st.session_state.chat_name = first_chat_name
        st.session_state.messages = load_messages(
            first_chat_id
        )

# -----------------------------
# Sidebar UI
# -----------------------------

with st.sidebar:

    # ==========================
    # KNOWLEDGE BASE
    # ==========================
    st.markdown("### 📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT or HTML",
        type=["pdf", "txt", "html"]
    )

    if uploaded_file:
        fname = uploaded_file.name.lower()
        full_text = ""

        # PDF
        if fname.endswith(".pdf"):
            if pdfplumber is None:
                st.error("pdfplumber not installed.")
            else:
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    if len(full_text) > 2_000_000:
                        st.warning("File too large.")

        # TXT
        elif fname.endswith(".txt"):
            raw = uploaded_file.read()
            full_text = (
                raw.decode("utf-8", errors="ignore")
                if isinstance(raw, bytes)
                else str(raw)
            )

        # HTML
        elif fname.endswith((".html", ".htm")):
            if BeautifulSoup is None:
                st.error("beautifulsoup4 not installed.")
            else:
                raw = uploaded_file.read()
                raw = (
                    raw.decode("utf-8", errors="ignore")
                    if isinstance(raw, bytes)
                    else str(raw)
                )

                soup = BeautifulSoup(raw, "html.parser")
                full_text = soup.get_text(separator="\n")

        if full_text:
            with st.spinner("Indexing document..."):
                added_chunks = store_file_chunks(
                    st.session_state.chat_id,
                    uploaded_file.name,
                    full_text
                )

            st.success(
                f"✅ Added {added_chunks} chunks from {uploaded_file.name}"
            )

        else:
            st.warning("No text extracted from file.")

    # ==========================
    # CHAT MANAGEMENT
    # ==========================
    st.markdown("---")
    st.markdown("### 💬 Chats")

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        new_chat_name = f"Chat {new_chat_id[:5]}"

        create_chat(
            new_chat_id,
            new_chat_name
        )

        st.session_state.chat_id = new_chat_id
        st.session_state.chat_name = new_chat_name
        st.session_state.messages = []

        st.rerun()

    # Chat List
    for chat_id, chat_name in load_chats():

        if st.button(
                chat_name,
                key=f"chat_{chat_id}"
        ):
            st.session_state.chat_id = chat_id
            st.session_state.chat_name = chat_name

            st.session_state.messages = load_messages(
                chat_id
            )

            st.rerun()

    st.info(
        f"**Current Chat:** {st.session_state.chat_name}"
    )

    # Rename Chat
    new_name = st.text_input(
        "Rename Chat",
        value=st.session_state.chat_name
    )

    if st.button("💾 Save Name"):
        rename_chat(
            st.session_state.chat_id,
            new_name
        )

        st.session_state.chat_name = new_name

        st.rerun()

    # Delete Chat
    if st.button(
            "🗑️ Delete Current Chat",
            use_container_width=True
    ):

        current_chat = st.session_state.chat_id

        all_chats = load_chats()

        if len(all_chats) > 1:

            delete_chat(current_chat)

            remaining = load_chats()

            next_chat_id, next_chat_name = remaining[0]

            st.session_state.chat_id = next_chat_id
            st.session_state.chat_name = next_chat_name

            st.session_state.messages = load_messages(
                next_chat_id
            )

            st.rerun()

        else:

            st.warning(
                "At least one chat must exist."
            )

    # Clear Current Conversation
    if st.button(
            "🧹 Clear Conversation",
            use_container_width=True
    ):
        clear_chat_messages(
            st.session_state.chat_id
        )

        st.session_state.messages = []

        st.rerun()

    # ==========================
    # CHAT STATS
    # ==========================
    st.markdown("---")
    st.markdown("### 📊 Chat Stats")

    message_count = len(
        st.session_state.messages
    )

    user_count = sum(
        1
        for m in st.session_state.messages
        if m["role"] == "user"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Messages",
            message_count
        )

    with col2:
        st.metric(
            "Questions",
            user_count
        )
    # ==========================
    # MEMORY
    # ==========================
    st.markdown("---")
    st.markdown("### 🧠 Memory")

    with st.expander("View Stored Collections"):
        try:
            collections = list_collections()

            if collections:
                st.write(collections)
            else:
                st.write("No collections found.")

        except Exception as e:
            st.error(str(e))

    # ==========================
    # RETRIEVAL PREVIEW
    # ==========================
    DEBUG = False
    if DEBUG:
        with st.expander("🔍 Retrieved Context Preview"):

            last_query = st.session_state.get(
                "last_query",
                ""
            )

            if last_query:

                ctx = retrieve_context(
                    st.session_state.chat_id,
                    last_query,
                    k=5
                )

                if ctx:
                    for i, chunk in enumerate(ctx, start=1):
                        st.markdown(
                            f"**{i}.** {chunk[:250]}..."
                        )
                else:
                    st.write("No context retrieved.")

            else:
                st.write("No query yet.")

        # ==========================
        # DEBUG INFO
        # ==========================
        with st.expander("⚙️ Debug Info"):

            st.write("Chat Name")
            st.code(st.session_state.chat_name)

            st.write("Chat ID")
            st.code(st.session_state.chat_id)

    # ==========================
    # DANGER ZONE
    # ==========================
    st.markdown("---")
    st.markdown("### 🗑️ Danger Zone")

    confirm_delete = st.checkbox(
        "I understand this will permanently delete all memory."
    )

    if confirm_delete:

        if st.button("Clear Memory"):

            import chromadb

            client = chromadb.PersistentClient(
                path=os.path.join(
                    os.getcwd(),
                    "chroma_data"
                )
            )

            for collection in client.list_collections():
                try:
                    client.delete_collection(collection.name)
                except Exception:
                    pass

            st.session_state.messages = []

            st.success(
                "All vector memory cleared successfully."
            )

            st.rerun()

    # if st.button("🔥 Factory Reset"):
    #
    #     delete_all_chats()
    #
    #     st.session_state.messages = []
    #
    #     st.success("All chats deleted.")
    #
    #     st.rerun()
# -----------------------------
# Chat UI (original simple layout)
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
# -----------------------------
# Input
# -----------------------------

prompt = st.chat_input("Ask ContextIQ AI...")

if prompt:

    st.session_state.last_query = prompt

    with st.spinner("Thinking..."):

        answer = generate_rag_response(
            st.session_state.chat_id,
            prompt
        )

    # Save to session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save to SQLite
    save_message(
        st.session_state.chat_id,
        "user",
        prompt
    )

    save_message(
        st.session_state.chat_id,
        "assistant",
        answer
    )

    st.rerun()