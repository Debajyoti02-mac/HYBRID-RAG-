# 🔍 Hybrid RAG with ReRanker

A **Retrieval-Augmented Generation** pipeline that combines **dense search**, **sparse search (BM25)**, **Reciprocal Rank Fusion**, and **Cross-Encoder reranking** to deliver precise, context-grounded answers from PDF documents.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)

---

## 🏗️ Architecture

```
PDF Document
    │
    ▼
┌──────────────────┐
│  Text Chunking   │  (RecursiveCharacterTextSplitter)
└──────┬───────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌──────────────┐     ┌──────────────┐
│ Dense Search │     │ Sparse Search│
│  (ChromaDB)  │     │   (BM25)     │
└──────┬───────┘     └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  ▼
       ┌────────────────────┐
       │ Reciprocal Rank    │
       │   Fusion (RRF)     │
       └──────────┬─────────┘
                  ▼
       ┌────────────────────┐
       │  Cross-Encoder     │
       │   ReRanking        │
       └──────────┬─────────┘
                  ▼
       ┌────────────────────┐
       │   Groq LLM         │
       │ (LLaMA 3.1 8B)     │
       └──────────┬─────────┘
                  ▼
            Final Answer
```

---

## ✨ Features

- **Hybrid Retrieval** — Combines semantic (dense) and keyword (sparse) search for better recall
- **BM25 Sparse Search** — Catches exact keyword matches that embeddings might miss
- **Reciprocal Rank Fusion** — Merges results from both retrieval strategies intelligently
- **Cross-Encoder ReRanking** — Re-scores candidates with a powerful cross-encoder model for precision
- **Groq LLM** — Generates concise, context-grounded answers via LLaMA 3.1
- **Streamlit Chat UI** — Clean, chat-style interface with conversation history
- **Suggested Questions** — One-click starter questions for quick exploration

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **PDF Loading** | LangChain `PyPDFLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter` |
| **Embeddings** | `all-MiniLM-L6-v2` (384 dims) |
| **Dense Search** | ChromaDB |
| **Sparse Search** | BM25 (`rank_bm25`) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **ReRanker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Groq — LLaMA 3.1 8B Instant |
| **Frontend** | Streamlit |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Debajyoti02-mac/ReRank-with-Hybrid-RAG-.git
cd ReRank-with-Hybrid-RAG-
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
ReRank-with-Hybrid-RAG-/
├── app.py                          # Streamlit app (main entry point)
├── ReRanker_with_Hybrid.ipynb      # Jupyter notebook (development & experimentation)
├── GK_Questions.pdf                # Source PDF document
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 📖 How It Works

1. **Load & Chunk** — The PDF is loaded and split into overlapping chunks (1000 chars, 120 overlap)
2. **Dense Retrieval** — Query is embedded and matched against ChromaDB vectors (top 5, distance < 1.0)
3. **Sparse Retrieval** — BM25 scores all chunks by keyword overlap (top 10)
4. **Rank Fusion** — Results from both are merged using RRF with k=60
5. **ReRanking** — A Cross-Encoder re-scores the top 5 fused candidates for precision
6. **Answer Generation** — The reranked context is sent to Groq LLM for a concise answer

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub ✅
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add `GROQ_API_KEY` under **Advanced Settings → Secrets**
6. Deploy!

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Debajyoti Hazra**

- GitHub: [@Debajyoti02-mac](https://github.com/Debajyoti02-mac)
