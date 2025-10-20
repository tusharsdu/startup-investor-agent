# 🧠 LLM-based Question Answering System (RAG) — Jupyter Notebook + Streamlit App

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)**–based **Question Answering (QA)** system using a **Large Language Model (LLM)** inside a **Jupyter Notebook** and deploy it as a **Streamlit web app**.

The system retrieves relevant information from a set of documents (PDFs, text files, or notes) and uses an LLM to generate contextually accurate and up-to-date answers.

---

## 📘 Project Overview

The notebook walks through the complete RAG pipeline:
1. **Importing dependencies**
2. **Loading and preprocessing documents**
3. **Creating embeddings** with SentenceTransformers or OpenAI embeddings  
4. **Building a FAISS vector store**
5. **Implementing context retrieval**
6. **Generating answers with an LLM**
7. **Launching the Streamlit interface**

The Streamlit app allows users to **upload documents** and **ask questions** interactively.

---

## 🧩 Technologies Used

| Component | Library / Tool |
|------------|----------------|
| Embeddings | SentenceTransformers / OpenAI `text-embedding-3-large` |
| Vector Store | FAISS |
| LLM | GPT-3.5 / Perpexity sonar pro |
| Framework | LangChain |
| Interface | Streamlit |
| Notebook Environment | Jupyter Notebook |
| Language | Python 3.10+ |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone the following repo 
Create virtual enveronment
pip install -r requirements.txt
streamlit run app.py


