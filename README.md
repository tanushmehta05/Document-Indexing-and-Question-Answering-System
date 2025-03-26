# Document Retrieval and AI-Powered Query System

## Overview
This project builds an AI-powered document retrieval and query system using **Pinecone** for vector storage, **Google Gemini API** (or **OpenAI ChatGPT API**) for embedding generation and response generation, and **NLTK** for text preprocessing.

### Key Features
- **Text Embeddings & Vector Storage**: Uses **Google Gemini API** (or **OpenAI's embedding model**) to generate embeddings and store them in Pinecone.
- **Efficient Document Search**: Finds the most relevant documents using **cosine similarity**.
- **AI-Powered Answers**: Fetches relevant documents and queries the AI model (Gemini or ChatGPT) for context-aware responses.
- **Improved Preprocessing**: Uses **NLTK** for sentence tokenization and stopword removal.

---
## Installation & Setup

### Prerequisites
- Python 3.8+
- Required API Keys:
  - **Pinecone API Key**
  - **Google Gemini API Key** (or **OpenAI API Key** for ChatGPT)

### Install Dependencies
```bash
pip install pinecone-client google-generativeai openai nltk
```

### Setup API Keys
Store API keys in `api.py`:
```python
pinecone_api_key = "your_pinecone_api_key"
gemini_api_key = "your_google_gemini_api_key"
openai_api_key = "your_openai_api_key"  # Only if using OpenAI
```

### Run the Program
```bash
python Gemini_LLP.py  # For Google Gemini API
python ChatGPT_LLP.py  # For OpenAI ChatGPT API
```

---
## Project Structure
```
|-- ML/LLM_open_ai/
    |-- Gemini_LLP.py  # Uses Google Gemini API
    |-- ChatGPT_LLP.py  # Uses OpenAI API
    |-- api.py  # Store API keys
    |-- storage/  # Stores vectorizer and document data
    |-- documents/  # .txt files to be indexed
```

---
## How It Works

1. **Indexing Documents**:
   - Reads `.txt` files from the specified directory.
   - Preprocesses text (removes stopwords, tokenizes sentences).
   - Generates embeddings using **Google Gemini** (or **OpenAI's embedding model**).
   - Stores embeddings in **Pinecone**.

2. **Handling User Queries**:
   - Converts query into embeddings.
   - Searches for the most relevant document in Pinecone.
   - Sends the retrieved document and query to **Google Gemini** (or **ChatGPT API**).
   - Returns an AI-generated response.

---
## Differences & Improvements from Previous Version

| Feature | Old Version (ChatGPT) | New Version (Gemini) |
|---------|----------------|----------------|
| **Embedding Model** | OpenAI’s `text-embedding-ada-002` | Google Gemini’s `embedding-001` |
| **Vector Storage** | Pinecone with TF-IDF | Pinecone with Gemini embeddings |
| **Document Processing** | Basic tokenization | Advanced preprocessing (NLTK) |
| **Error Handling** | Limited retry logic | Better API failure handling |
| **AI Model for Responses** | OpenAI `text-davinci-003` | Google Gemini `gemini-1.5-pro-latest` |

### Outdated Components Fixed
- **Replaced TF-IDF with AI-generated embeddings** for better semantic search.
- **Removed deprecated OpenAI Embedding API** and migrated to the **latest OpenAI/Gemini APIs**.
- **Improved query processing** with more robust error handling.
- **Added NLTK preprocessing** for better document structuring.

---
## Future Enhancements
- **Hybrid Approach**: Allow switching between Gemini and ChatGPT dynamically.
- **UI Interface**: Build a web-based interface for easier interactions.
- **Metadata Filtering**: Enhance search results with additional metadata filters.

---
## Author
Developed by **Tanush**for efficient document retrieval and AI-assisted queries.