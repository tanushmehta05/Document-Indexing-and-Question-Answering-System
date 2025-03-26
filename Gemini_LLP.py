import os
import time
import pinecone
import google.generativeai as genai
import google.api_core.exceptions  
import openai  
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Ensure required nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

from api import pinecone_api_key, gemini_api_key, openai_api_key  

# 🔹 Initialize Pinecone
PINECONE_API_KEY = pinecone_api_key
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# 🔹 List existing indexes
existing_indexes = pc.list_indexes().names()
print("------ Existing indexes ------:\n", existing_indexes)

# 🔹 Set index name
index_name = "document-index-new"

# 🔹 Check if index exists, otherwise create it
if index_name not in existing_indexes:
    print(f"🛠️ Creating new Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # ✅ Adjusted for OpenAI embeddings
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)  
else:
    print(f"✅ Using existing index: {index_name}")

# 🔹 Connect to the Pinecone index
index = pc.Index(index_name)

# 🔹 Initialize Google Gemini API
genai.configure(api_key=gemini_api_key)

# 🔹 Initialize OpenAI API
openai.api_key = openai_api_key  

# 🔹 Directory containing .txt files
TEXT_DIR = "C:/Users/ASUS/Documents/ML/LLM_open_ai"

def preprocess_text(text):
    """Tokenizes and cleans text by removing stopwords."""
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = [
        ' '.join([word for word in sentence.split() if word.lower() not in stop_words])
        for sentence in sentences
    ]
    return " ".join(cleaned_sentences)

def embed_text(text):
    """Generates text embeddings using OpenAI's Ada-002 model (Fixed for new API)."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]  # ✅ Pass input as a list
        )
        embedding_vector = response.data[0].embedding  # ✅ Corrected extraction
        return embedding_vector
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None

def read_txt_files(directory):
    """Reads all .txt files from the specified directory."""
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                documents[filename] = file.read()
    return documents

def build_index_from_txt():
    """Reads .txt files, processes them, and stores embeddings in Pinecone."""
    documents = read_txt_files(TEXT_DIR)
    for doc_id, (filename, text) in enumerate(documents.items()):
        cleaned_text = preprocess_text(text)
        embedding = embed_text(cleaned_text)
        if embedding:
            index.upsert([(str(doc_id), embedding, {"text": text, "filename": filename})])
            print(f"✅ Indexed {filename}")
    print("🔄 Index updated successfully!")

def query_google_api(query):
    """Queries Google's Gemini API for a response."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error querying Gemini API: {e}")
        return "Error processing request."

def handle_queries():
    """Handles user queries by searching Pinecone and fetching Google API responses."""
    while True:
        query = input("\n💬 Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        embedding = embed_text(query)
        if embedding:
            results = index.query(vector=embedding, top_k=5, include_metadata=True)
            if "matches" in results and results["matches"]:
                best_match = results["matches"][0]["metadata"]["text"]
                print(f"\n📜 Best match found in database: {best_match[:200]}...")  
                response = query_google_api(f"Based on this document: {best_match}, answer the question: {query}")
            else:
                response = "No relevant information found."
            print("🤖 Chatbot:", response)

# 🔥 Build Index from .txt files and start querying
build_index_from_txt()
handle_queries()
