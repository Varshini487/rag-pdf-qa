# 🧠 RAG Pipeline — PDF Q&A App

A full Retrieval Augmented Generation (RAG) pipeline that lets you upload any PDF and ask questions — AI answers from YOUR document, not from its training data.

## 🚀 What is RAG?
RAG = Retrieval Augmented Generation

| Phase | What Happens |
|-------|-------------|
| **Indexing** | Load PDF → Split into chunks → Embed each chunk → Store in FAISS |
| **Retrieval** | Embed question → Find similar chunks → Retrieve top 3 |
| **Generation** | Send chunks + question to GPT → Get grounded answer |

## ✨ Features
- 📄 Load any PDF document
- ✂️ Smart chunking with overlap (no lost context)
- 🔢 OpenAI embeddings for semantic understanding
- ⚡ FAISS vector database for fast similarity search
- 🤖 GPT-3.5 answers ONLY from your document (no hallucination)
- 💬 Interactive Q&A loop in terminal

## 🛠️ Tech Stack
- Python
- LangChain
- OpenAI API (embeddings + GPT-3.5-turbo)
- FAISS (vector database)
- PyPDF (document loader)

## ⚙️ Setup
```bash
pip install langchain langchain-openai langchain-community faiss-cpu pypdf
```
Add `OPENAI_API_KEY` to your environment and run:
```bash
python rag_pipeline.py
```

## 📁 Files
- `rag_pipeline.py` — Full RAG pipeline with Q&A loop
- `rag_colab.ipynb` — Google Colab version (run in browser)

## 👩‍💻 Author
**Varshini Marathi** — Aspiring AI/ML Engineer | B.Tech CSE 2027
📧 marathivarshini5@gmail.com | [LinkedIn](https://linkedin.com/in/varshini-marathi-64a09329a) | [GitHub](https://github.com/Varshini487)
