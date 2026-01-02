# 🧠 Mini Project 1: Retrieval-Augmented Generation (RAG) System

A hands-on **mini project to deeply understand core RAG concepts** using modern LangChain (v1.x), FAISS vector database, HuggingFace embeddings, and a local LLM via Ollama.

This project is designed as a **foundation-building exercise** before moving to advanced, production-grade RAG systems.

---

## 🚀 Project Objective

To build an **end-to-end RAG pipeline** that:

* Ingests PDF documents
* Chunks and embeds text efficiently
* Stores embeddings in FAISS
* Retrieves relevant context
* Uses a local LLM to answer user queries **only from retrieved context**

---

## 🏗️ Architecture Overview

```
PDF Documents
     ↓
Document Loader (PyMuPDF)
     ↓
Text Chunking (RecursiveCharacterTextSplitter)
     ↓
Embeddings (HuggingFace MiniLM)
     ↓
Vector Store (FAISS)
     ↓
Retriever (Similarity Search)
     ↓
Prompt Template
     ↓
Local LLM (Ollama - Llama3)
     ↓
Grounded Answer
```

---

## 🧰 Tech Stack

* **Python 3.12**
* **LangChain v1.x (LCEL / Runnable chains)**
* **FAISS** – vector database
* **HuggingFace Embeddings** – `sentence-transformers/all-MiniLM-L6-v2`
* **Ollama** – local LLM (`llama3`)
* **PyMuPDF** – PDF loading

---

## 📁 Project Structure

```
MINI_PROJECT1/
│
├── data/
│   └── sample.pdf
│
├── ingest.py        # Document ingestion + FAISS index creation
├── rag_qa.py        # RAG-based question answering
├── faiss_index/     # Stored vector index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv mini1
mini1\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install langchain langchain-core langchain-community \
langchain-ollama langchain-text-splitters \
faiss-cpu sentence-transformers pymupdf
```

### 3️⃣ Install & run Ollama

```bash
ollama pull llama3
ollama run llama3
```

---

## 📥 Ingest Documents

Place PDFs inside the `data/` folder.

Run:

```bash
python ingest.py
```

This will:

* Load PDF pages
* Split text into chunks
* Generate embeddings
* Save FAISS index locally

---

## 💬 Ask Questions (RAG)

Run:

```bash
python rag_qa.py
```

Example queries:

* `What is this document about?`
* `Explain encoder and decoder stacks`
* `Summarize key concepts`

The system answers **only using retrieved context**.

---

## Output 
![alt text](image.png)

## 🧠 Key Learnings

* Difference between **LLMs vs Embedding Models**
* Why embedding model consistency is critical
* How FAISS similarity search works
* Modern LangChain **LCEL / Runnable** pattern
* Common real-world RAG errors and fixes

---

## 🔍 Common Issues Solved

* LangChain v1.x breaking changes
* FAISS dimension mismatch errors
* Ollama vs HuggingFace model confusion
* Performance optimization during ingestion

---

## 👩‍💻 Author

**Sneha Sathe**
Aspiring GenAI / AI Engineer | RAG | LangChain | LLMs

---

⭐ If you find this useful, feel free to star the repo!
