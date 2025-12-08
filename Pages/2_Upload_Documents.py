# pages/2_Upload_Documents.py
import streamlit as st
from chroma_rag import store_documents, get_or_create_collection
import os
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

st.title("📚 Upload Documents")

collection_name = st.text_input("Collection name (e.g. 'knowledge_base')", value="docs")

uploaded = st.file_uploader("Upload PDF, TXT, or DOCX", accept_multiple_files=True)

def extract_text_from_file(file):
    fname = file.name.lower()
    data = file.read()
    if fname.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif fname.endswith(".docx"):
        doc = Document(BytesIO(data))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    else:
        # txt or fallback
        try:
            return data.decode("utf-8")
        except:
            return ""

if st.button("Ingest files"):
    all_docs = []
    metadatas = []
    for f in uploaded:
        txt = extract_text_from_file(f)
        # naive splitting into chunks
        chunk_size = 1000
        chunks = [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size) if txt[i:i+chunk_size].strip()]
        for ch in chunks:
            all_docs.append(ch)
            metadatas.append({"source": f.name})
    if all_docs:
        store_documents(collection_name, all_docs, metadatas)
        st.success(f"Ingested {len(all_docs)} chunks into {collection_name}")
    else:
        st.warning("No text extracted from files.")
