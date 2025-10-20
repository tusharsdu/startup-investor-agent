#Multi-Agent AI Startup Investment Analysis Systemup pitch deck analysis

Uses tools dynamically (APIs: Perplexity sonar-pro, OpenAI GPT-3.5-turbo; datasets: FAISS vector storage; retrieval: Sentence Transformers)
Self-reflects to assess the quality of its analysis output through iterative improvement cycles
Learns across runs by keeping brief memories and notes in FAISS vector database to improve future analyses
What is this project? This is a sophisticated Multi-Agent AI system designed to analyze startup pitch decks and provide comprehensive investment analysis using Model Context Protocol (MCP) architecture. The system employs specialized AI agents that work together to evaluate every aspect of a startup investment opportunity.

Why are we building this? Traditional investment analysis is time-consuming, subjective, and often misses critical details. Our system:

Standardizes investment evaluation across all startups
Accelerates the due diligence process from weeks to minutes
Reduces bias through systematic AI-driven analysis
Improves accuracy by analyzing multiple dimensions simultaneously
Scales investment evaluation for VC firms and angel investors
Core Value Proposition: Transform unstructured pitch deck content into actionable investment intelligence through coordinated AI agents, each specialized in different aspects of startup evaluation.

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


