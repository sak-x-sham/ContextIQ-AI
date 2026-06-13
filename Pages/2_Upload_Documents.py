# pages/2_Upload_Documents.py

import streamlit as st
from chroma_rag import store_documents, get_documents
from io import BytesIO
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="Upload Documents")

st.title("📚 Upload Documents")

# All uploaded documents will go into this collection
collection_name = "documents"

uploaded = st.file_uploader(
    "Upload PDF, TXT, or DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)


# --------------------------------------------------
# File Text Extraction
# --------------------------------------------------
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

        text = "\n".join(
            [p.text for p in doc.paragraphs]
        )

        return text

    elif fname.endswith(".txt"):

        try:
            return data.decode("utf-8")
        except Exception:
            return ""

    return ""


# --------------------------------------------------
# Chunking with overlap
# --------------------------------------------------
def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Ingestion Button
# --------------------------------------------------
if st.button("Ingest Files"):

    if not uploaded:

        st.warning(
            "Please upload at least one file."
        )

        st.stop()

    all_docs = []

    metadatas = []

    total_chunks = 0

    for f in uploaded:

        txt = extract_text_from_file(f)

        if not txt.strip():
            continue

        chunks = chunk_text(txt)

        total_chunks += len(chunks)

        for ch in chunks:

            all_docs.append(ch)

            metadatas.append({
                "source": f.name,
                "type": "document"
            })

    if all_docs:

        store_documents(
            collection_name,
            all_docs,
            metadatas
        )

        st.success(
            f"Successfully ingested {total_chunks} chunks."
        )

        # Verification
        stored = get_documents(
            collection_name
        )

        st.info(
            f"Collection '{collection_name}' now contains "
            f"{len(stored['documents'])} chunks."
        )

    else:

        st.warning(
            "No text could be extracted from uploaded files."
        )


# --------------------------------------------------
# Collection Stats
# --------------------------------------------------
try:

    stored = get_documents(collection_name)

    st.markdown("---")

    st.subheader("📊 Collection Statistics")

    st.write(
        f"Total chunks stored: "
        f"{len(stored['documents'])}"
    )

except Exception:
    pass