# Hybrid RAG: PDF Question Answering

This project demonstrates a simple **Hybrid RAG** pipeline over a PDF document.

Hybrid RAG means we retrieve useful context using two methods:

- **Dense retrieval**: semantic/vector search using embeddings and ChromaDB
- **Sparse retrieval**: keyword-based search using BM25

The retrieved chunks are then merged using **Reciprocal Rank Fusion (RRF)** and passed to an LLM for answer generation.

## Pipeline Overview

```text
PDF
 ↓
Load pages
 ↓
Split into chunks
 ↓
Create two search systems
 ├── Dense search: ChromaDB + SentenceTransformer embeddings
 └── Sparse search: BM25 keyword scoring
 ↓
Merge results with RRF
 ↓
Send best chunks to LLM
 ↓
Final answer
```

## Main Components

### 1. PDF Loading

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Why_Language_Models_Hallucinate_Explainer.pdf")
pages = loader.load()
```

This loads the PDF into page-level documents.

Each page contains:

- `page_content`: actual text
- `metadata`: page number, source file, etc.

### 2. Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120
)

texts = text_spliter.split_documents(pages)
chunks = [i.page_content for i in texts]
metadata = [i.metadata for i in texts]
```

The PDF text is split into smaller chunks.

Why this matters:

- LLMs work better with smaller context pieces
- Retrieval becomes more accurate
- Overlap prevents important ideas from being cut off

### 3. Dense Retrieval With ChromaDB

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./use_hybrid_Data")
collection = client.get_or_create_collection(
    name="HYbrid_RAG",
    embedding_function=embedding_function
)
```

Dense retrieval converts text into vectors.

Simple idea:

```text
similar meaning = nearby vectors
```

Example:

```text
"model hallucination"
```

can match:

```text
"LLMs generate false claims"
```

even if the exact words are different.

### 4. Sparse Retrieval With BM25

```python
from rank_bm25 import BM25Okapi

tokens = [c.split() for c in chunks]
bm250 = BM25Okapi(tokens)
```

BM25 is keyword-based retrieval.

Simple idea:

```text
rare matching words = higher score
common words = lower score
```

BM25 is useful when exact terms matter.

### 5. Hybrid Retrieval Function

```python
def Hybrid_Function(query: str):
    result = collection.query(query_texts=[query], n_results=5)
    doc_ = result["documents"][0]
    distance = result["distances"][0]

    threshold = 1.0

    dense_filter = []
    for doc, d in zip(doc_, distance):
        if d < threshold:
            dense_filter.append(doc)

    rm_score = bm250.get_scores(query.split())

    def get_top_index(score, k=10):
        index = list(enumerate(score))
        index_score = sorted(index, key=lambda x: x[1], reverse=True)
        return [i for i, s in index_score[:k]]

    top_index = get_top_index(score=rm_score, k=10)

    sparse_index = []
    for i in top_index:
        if i not in sparse_index:
            sparse_index.append(i)

    sparse_docs = [chunks[i] for i in sparse_index]

    rrf_rank = {}

    for rank, doc in enumerate(dense_filter):
        rrf_rank[doc] = rrf_rank.get(doc, 0) + 1 / (60 + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        rrf_rank[doc] = rrf_rank.get(doc, 0) + 1 / (60 + rank + 1)

    merge = sorted(rrf_rank.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in merge[:5]]

    if not top_docs:
        return "NOT RELEVANT TOPICS"

    return "\n\n".join(top_docs)
```

This function performs the core Hybrid RAG retrieval.

It does three things:

1. Gets semantically similar chunks from ChromaDB
2. Gets keyword-matching chunks from BM25
3. Combines both using RRF

## RRF: Reciprocal Rank Fusion

RRF combines results from different retrievers.

Formula:

```text
score = 1 / (k + rank)
```

In this project:

```python
1 / (60 + rank + 1)
```

Meaning:

- Higher-ranked chunks get more score
- Chunks appearing in both dense and sparse search get extra score
- Final results are sorted by combined score

## LLM Setup

```python
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

key = os.getenv("GROQ_API_KEY")

cha_model = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=key
)
```

The LLM generates the final answer using retrieved context.

Important:

```python
key = os.getenv("GROQ_API_KEY")
```

Do not use:

```python
key = "GROQ_API_KEY"
```

That passes the text `"GROQ_API_KEY"` instead of the real API key.

## Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=gsk_your_real_key_here
```

If you get this error:

```text
AuthenticationError: Invalid API Key
```

check that:

- your `.env` file exists
- the key starts with `gsk_`
- you are using `os.getenv("GROQ_API_KEY")`

## Example Usage

```python
question = "Why Model Hallucinate?"

answers = Hybrid_Function(query=question)

prompt = f"""
You are a reliable LLM.
Answer only from the given content.
Do not use outside knowledge.

content: {answers}
question: {question}
"""

response = cha_model.invoke(prompt)
print(response.content)
```

## Important Notes

- ChromaDB distance works like this:

```text
smaller distance = more similar
```

- BM25 score works like this:

```text
higher score = better keyword match
```

- RRF score works like this:

```text
higher final score = better hybrid result
```

## Common Mistakes

### Passing Raw Query To BM25

BM25 expects tokenized words:

```python
bm250.get_scores(query.split())
```

### Using `.sort()` Incorrectly

This is wrong:

```python
index_score = index.sort(...)
```

because `.sort()` returns `None`.

Use:

```python
index_score = sorted(index, key=lambda x: x[1], reverse=True)
```

### Joining Integers Instead Of Text

BM25 returns indexes, not documents.

Convert indexes to chunks:

```python
sparse_docs = [chunks[i] for i in sparse_index]
```

Then join the text chunks.

## Core Idea

```text
Hybrid RAG = vector meaning search + keyword search + rank fusion + LLM answer
```

Dense search understands meaning.

BM25 catches exact words.

RRF combines both.

The LLM writes the final answer from the retrieved context.
