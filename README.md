# Quote RAG

Local RAG pipeline that answers questions grounded in a quotes CSV, using ChromaDB for retrieval and Groq's `llama-3.1-8b-instant` for generation.

## Stack

- **Loader**: `CSVLoader` (langchain_community)
- **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=120)
- **Vector store**: ChromaDB `PersistentClient`, `all-MiniLM-L6-v2` embeddings
- **LLM**: Groq `llama-3.1-8b-instant` via `langchain_groq`
- **Tools**: `calculator`, `contextual_Function` (retrieval)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-community langchain-groq langchain-text-splitters chromadb python-dotenv sentence-transformers
```

Create `.env` (not committed):

```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

## Data

Place `qoute_dataset.csv` in the project root. First run builds the Chroma collection at `./path` and persists it — subsequent runs skip re-embedding if `collection.count() > 0`.

## Usage

```python
question = "your question here"
answer = tools_function(question)
print(answer)
```

`tools_function` retrieves top-5 chunks (L2 distance threshold 1.9), returns `"NOT RELEVENT CONTENT"` if nothing qualifies, otherwise passes retrieved context + question to the LLM.

## Known issues / TODO

- Retrieval threshold (1.9) is not yet empirically validated for this corpus — tune against known query/answer pairs.
- `contextual_Function` and `calculator` are registered as tools (`bind_tools`) but `tools_function` currently calls `contextual_Function` directly rather than letting the LLM decide — tool-calling loop not yet wired end-to-end.
- No conversation memory across turns yet.
- Typos to clean up eventually: `qustion`, `massage`, `qoute` — cosmetic only, not blocking.

## Project structure

```
.
├── venv/
├── .env
├── qoute_dataset.csv
├── path/              # Chroma persistent store
└── notebook.ipynb
```
# Clean-Naive-RAG-
