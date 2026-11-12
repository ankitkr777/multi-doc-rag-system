import streamlit as st
from rag_engine import (
    process_uploaded_pdfs,
    build_conversational_rag_chain,
    format_chat_history
)

st.set_page_config(page_title="Multi-Document RAG Assistant", layout="wide")
st.title("Intelligent Multi-Document RAG System")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# Sidebar for multi-PDF upload and ingestion
with st.sidebar:
    st.header("Document Repository")
    uploaded_files = st.file_uploader(
        "Upload PDF reports, policies, or manuals",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Extracting text, chunking, and building vector index..."):
                try:
                    vectorstore = process_uploaded_pdfs(uploaded_files)
                    st.session_state.rag_chain = build_conversational_rag_chain(vectorstore)
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.session_state.messages = []  # Reset chat on new document set
                    st.success(f"Indexed {len(uploaded_files)} document(s) successfully.")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    if st.session_state.indexed_files:
        st.subheader("Active Knowledge Base")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- `{fname}`")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Source References"):
                for src in message["sources"]:
                    st.markdown(f"**{src['file']} — Page {src['page']}**")
                    st.caption(src["excerpt"])

# Chat input handler
if prompt := st.chat_input("Ask a question across your uploaded documents..."):
    if st.session_state.rag_chain is None:
        st.info("Upload and index PDF documents in the sidebar first.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate grounded response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            chat_history = format_chat_history(st.session_state.messages[:-1])
            response = st.session_state.rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })

            answer = response["answer"]
            context_docs = response.get("context", [])

            # Deduplicate source citations
            seen = set()
            unique_sources = []
            for doc in context_docs:
                file_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page", "N/A")
                key = (file_name, page_num)
                if key not in seen:
                    seen.add(key)
                    unique_sources.append({
                        "file": file_name,
                        "page": page_num,
                        "excerpt": doc.page_content[:300].replace("\n", " ") + "..."
                    })

            st.markdown(answer)
            if unique_sources:
                with st.expander("View Source References"):
                    for src in unique_sources:
                        st.markdown(f"**{src['file']} — Page {src['page']}**")
                        st.caption(src["excerpt"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": unique_sources
    })