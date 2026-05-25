# London Hotel RAG Chatbot

A RAG (Retrieval-Augmented Generation) pipeline built with LangChain, HuggingFace embeddings, and Gemini API over a London hotel reviews dataset.

## Stack
- LangChain
- HuggingFace Sentence Transformers
- Google Gemini API
- ChromaDB
- Python

## How it works
1. Loads London hotel reviews from CSV
2. Splits into chunks and creates vector embeddings
3. User asks a question in natural language
4. Retrieves relevant reviews and generates a response via Gemini

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Gemini API key to a `.env` file: `GEMINI_API_KEY=your_key`
4. Run: `python main.py`
