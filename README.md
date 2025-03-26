# README: AI-Powered Document Search with Pinecone, OpenAI, and Gemini

## Overview
This project enables document-based query retrieval using **Pinecone for vector storage**, **OpenAI for embeddings & chat completions**, and **Google Gemini for enhanced responses**. It allows users to upload `.txt` files, preprocess them, generate embeddings, store them in Pinecone, and retrieve relevant text passages based on user queries.

## Features
✅ **Embeds text using OpenAI's latest API (`openai.embeddings.create()`)**  
✅ **Uses Pinecone for efficient document retrieval**  
✅ **Integrates Google Gemini for enhanced responses**  
✅ **Preprocesses text (tokenization, stopword removal)**  
✅ **Handles `.txt` files automatically from a directory**  
✅ **Corrects outdated OpenAI API usage and improves performance**  

## Dependencies
Ensure you have the following installed:
```sh
pip install openai pinecone-client google-generativeai nltk
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/LLM_open_ai.git
   cd LLM_open_ai
   ```
2. Add your API keys to `api.py`:
   ```python
   pinecone_api_key = "your_pinecone_api_key"
   gemini_api_key = "your_gemini_api_key"
   openai_api_key = "your_openai_api_key"
   ```
3. Run the script:
   ```sh
   python Gemini_LLP.py
   ```

## How It Works
1. Reads `.txt` files from `C:/Users/ASUS/Documents/ML/LLM_open_ai`
2. Preprocesses text (removes stopwords, tokenizes, and cleans up)
3. Generates embeddings using OpenAI's latest API (`text-embedding-ada-002`)
4. Stores vectors in Pinecone
5. Handles user queries by retrieving the most relevant document passage
6. Uses Google Gemini for answering based on the retrieved text

## Improvements Over Previous Version
### ✅ Fixed Outdated OpenAI API Usage
- **Old Code:** Used `openai.Embedding` (deprecated in `openai>=1.0.0`)
- **New Code:** Uses `openai.embeddings.create(model="text-embedding-ada-002", input=[text])`

### ✅ Replaced TF-IDF with OpenAI Embeddings
- **Old Code:** Used `TfidfVectorizer()` for text representation (less accurate for large datasets)
- **New Code:** Uses **OpenAI's embedding model**, which is more robust for semantic search

### ✅ Enhanced Query Handling
- **Old Code:** Only retrieved the closest document but didn’t refine responses
- **New Code:** Uses Google Gemini to generate precise answers based on context

### ✅ Automatic Indexing in Pinecone
- **Old Code:** Required manual checks and index creation
- **New Code:** Automatically detects if an index exists and creates one if needed

## Running Queries
Once the system is set up, enter your queries:
```sh
💬 Enter your question (or 'exit' to quit): "What is the purpose of AI in healthcare?"
```
The chatbot will fetch relevant documents and provide an answer.

## Notes
- If the index already exists, it will use it instead of re-indexing.
- Queries are enhanced using Google Gemini for a better user experience.

## Future Enhancements
- **Support for more file formats (PDF, DOCX)**
- **Integration with a frontend for UI-based querying**
- **Multi-lingual document support**

---
**🚀 Enjoy using this AI-powered document search tool!**

