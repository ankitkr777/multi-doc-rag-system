# Multi-Document RAG Assistant

A production-minded Streamlit application for asking grounded questions across one or more PDF documents. Upload a set of reports, policies, manuals, or similar files; the application extracts and indexes their content, retrieves the most relevant passages for each question, and generates an answer with document and page references.

## Highlights

- Ingest multiple PDFs in a single session
- Preserve document and page metadata through retrieval
- Use semantic search with maximal marginal relevance (MMR) to improve result diversity
- Support follow-up questions with conversation-aware query reformulation
- Enforce grounded responses: when the retrieved documents do not contain the answer, the assistant says so instead of guessing
- Show source excerpts in the interface alongside generated answers
- Select an available Groq chat model dynamically, with sensible fallbacks

## Architecture

```text
PDF uploads
    |
    v
PyPDFLoader -> RecursiveCharacterTextSplitter -> HuggingFace embeddings
    |                                                    |
    +-------------------- metadata ---------------------+
                                                         v
                                                 Chroma vector store
                                                         |
                                                         v
Question + chat history -> query rewrite -> MMR retrieval -> Groq LLM -> answer + sources
```

The pipeline uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings. PDFs are written to temporary files only while PyPDF extracts their pages. The Chroma store is created for the active application session and is not configured for persistence.

## Prerequisites

- Python 3.10 or later
- A [Groq API key](https://console.groq.com/keys)
- Internet access on first run to download the embedding model, if it is not already cached locally

## Quick start

1. Clone the repository and enter the project directory.

   ```bash
   git clone <repository-url>
   cd multi-doc-rag
   ```

2. Create and activate a virtual environment.

   **Windows (PowerShell)**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root.

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

5. Start the application.

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed by Streamlit, upload PDFs in the sidebar, select **Process & Index Documents**, and begin asking questions.

## How it works

1. Each uploaded PDF is parsed page by page.
2. Pages are split into overlapping chunks (1,000 characters with a 200-character overlap) while retaining the original file name and page number.
3. Chunks are embedded and stored in Chroma.
4. For a follow-up question, the model first rewrites it into a standalone search query using the conversation history.
5. An MMR retriever selects up to five relevant, diverse chunks from fifteen candidate matches.
6. The chat model answers only from that retrieved context and is instructed to cite the source document and page inline.

## Project structure

```text
.
├── app.py             # Streamlit interface and session state management
├── rag_engine.py      # PDF ingestion, vector retrieval, and RAG chain
├── requirements.txt   # Runtime dependencies
├── .env               # Local secrets (not committed)
└── .gitignore
```

## Configuration

| Setting | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | API key used by `langchain-groq` to access a chat model. |

At runtime, the application checks the Groq model catalog and prefers these models when available:

1. `openai/gpt-oss-120b`
2. `meta-llama/llama-4-scout-17b-16e-instruct`
3. `qwen/qwen3-32b`
4. `llama-3.1-8b-instant`

If the catalog cannot be reached, it falls back to `llama-3.1-8b-instant`.

## Privacy and operational notes

- Do not commit `.env` files or API keys. The supplied `.gitignore` excludes them.
- The retrieved passages and questions are sent to Groq to generate answers. Review your organization’s data-handling policy before uploading sensitive documents.
- This application does not include OCR. Scanned PDFs without an embedded text layer may produce little or no usable content.
- Indexes are recreated when documents are processed; document state is not intended to survive application restarts.
- Source excerpts improve traceability, but generated answers should still be reviewed for important decisions.

## Troubleshooting

| Symptom | Likely cause | Recommended action |
| --- | --- | --- |
| `GROQ_API_KEY` errors or failed answer generation | Missing, invalid, or expired API key | Confirm the `.env` file is in the project root and contains a valid key. Restart Streamlit after changing it. |
| Model download is slow or fails | First-use download or restricted network | Ensure the host can access Hugging Face model downloads, then retry. |
| No useful answer from a PDF | The PDF is scanned, encrypted, or lacks extractable text | Use a text-based PDF or run OCR before upload. |
| Answers do not address the question | Relevant text was not retrieved | Rephrase the question with terms found in the source documents, or upload the document containing the required material. |

## Development

There is no automated test suite in the repository yet. Before contributing changes, at minimum:

```bash
python -m compileall app.py rag_engine.py
streamlit run app.py
```

Validate the main workflow manually with representative PDFs: upload and index files, ask a direct question, ask a follow-up question, inspect source references, and clear the conversation.

## License

No license has been specified for this repository. Add an explicit license before distributing or reusing the project outside your organization.
