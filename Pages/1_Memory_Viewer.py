# pages/1_Memory_Viewer.py
import streamlit as st
from chroma_rag import get_documents, clear_collection, list_collections

st.title("🧠 Memory Viewer")

# List all chroma collections
cols = list_collections()
st.sidebar.markdown("Collections:")
for c in cols:
    st.sidebar.write(c)

sel = st.selectbox("Select collection to view", options=cols)

if sel:
    data = get_documents(sel)
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    ids = data.get("ids", [])
    st.write(f"Total docs: {len(docs)}")
    for i, d in enumerate(docs):
        st.markdown(f"**Doc {i+1} (id: {ids[i]})**")
        st.write(d)
        st.write(metas[i])

    if st.button("Clear this collection"):
        clear_collection(sel)
        st.success(f"Cleared collection {sel}")
        st.experimental_rerun()
