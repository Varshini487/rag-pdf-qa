# ============================================================
# PROJECT: RAG Pipeline — PDF Q&A App
# Day 10 — RAG, LangChain, FAISS, OpenAI Embeddings
# Author: Varshini Marathi
# GitHub: github.com/Varshini487/rag-pdf-qa
# ============================================================

# pip install langchain langchain-openai langchain-community faiss-cpu pypdf

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"


# ── STEP 1: Load PDF ─────────────────────────────────────────
def load_pdf(pdf_path: str):
    print(f"\n📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s)")
    return documents


# ── STEP 2: Split into Chunks ────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ── STEP 3: Create Vector Store (FAISS) ─────────────────────
def create_vectorstore(chunks):
    print("🧠 Generating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store ready!")
    return vectorstore


# ── STEP 4: Build RAG Chain ──────────────────────────────────
def build_rag_chain(vectorstore):
    prompt_template = """You are a helpful assistant. Answer the question using ONLY 
the context provided below. If the answer is not found in the context, 
say "I could not find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


# ── STEP 5: Interactive Q&A Loop ────────────────────────────
def run_qa_loop(qa_chain):
    print("\n" + "="*55)
    print("🤖 RAG Q&A Bot — Ask anything about your document!")
    print("   Type \'quit\' to exit.")
    print("="*55)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 👋")
            break

        result = qa_chain.invoke({"query": question})
        print(f"\nAI: {result[\'result\']}")
        print(f"📎 Used {len(result[\'source_documents\'])} chunk(s) from the document")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    PDF_PATH = "your_document.pdf"   # Replace with your PDF path

    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    qa_chain = build_rag_chain(vectorstore)
    run_qa_loop(qa_chain)
