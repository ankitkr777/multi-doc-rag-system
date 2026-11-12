import os
import tempfile
from typing import List
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_active_groq_model() -> str:
    """Fetches the best currently active chat model available on your Groq API key."""
    api_key = os.getenv("GROQ_API_KEY")
    preferred = [
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant",
    ]
    try:
        res = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        available = {m["id"] for m in res.json().get("data", [])}
        for model_id in preferred:
            if model_id in available:
                return model_id
        for m_id in available:
            if not any(x in m_id for x in ["whisper", "guard", "tts", "orpheus"]):
                return m_id
    except Exception:
        pass
    return "llama-3.1-8b-instant"


def process_uploaded_pdfs(uploaded_files) -> Chroma:
    """Extracts text from multiple uploaded PDFs, preserves metadata, and indexes in Chroma."""
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1
                all_docs.append(page)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_docs:
        raise ValueError("No extractable text found in the uploaded PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="multi_doc_rag",
    )
    return vectorstore


def _format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[{source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_conversational_rag_chain(vectorstore: Chroma):
    """Builds a native LCEL history-aware retrieval chain with strict grounding rules."""
    llm = ChatGroq(
        model=get_active_groq_model(),
        temperature=0.0,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15},
    )

    # 1. Standalone question rewriter using chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question—just reformulate it if needed, "
            "otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_rewriter = contextualize_q_prompt | llm | StrOutputParser()

    def get_standalone_query(input_dict: dict) -> str:
        if input_dict.get("chat_history"):
            return question_rewriter.invoke(input_dict)
        return input_dict["input"]

    # 2. Grounded QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an enterprise document assistant. Answer the user's question ONLY using "
            "the retrieved context provided below.\n\n"
            "Strict Rules:\n"
            "1. If the answer cannot be found in the context, state clearly: "
            "'I cannot find this information in the uploaded documents.' Do not guess or use outside knowledge.\n"
            "2. Cite the document name and page number inline when stating facts (e.g., [Annual_Report.pdf, Page 12]).\n"
            "3. Keep answers clear, structured, and concise.\n\n"
            "Retrieved Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 3. Retrieve context docs first, then generate answer while preserving source docs
    retrieve_docs = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(get_standalone_query(x))
    )

    generate_answer = (
        RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = retrieve_docs | RunnableParallel(
        answer=generate_answer,
        context=lambda x: x["context"],
    )
    return rag_chain


def format_chat_history(messages: List[dict]) -> List:
    """Converts Streamlit session messages into LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history