import os
import openai
import json
import string
import nltk
from pathlib import Path
import numpy as np
from pinecone import Pinecone, ServerlessSpec  # Import new Pinecone class
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import API keys from api.py
from api import openai_api_key, pinecone_api_key
client = openai.OpenAI(api_key=openai_api_key)

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def read_files_from_directory(directory_path):
    texts = []
    for file_path in Path(directory_path).rglob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def create_pinecone_index(texts, index_name="document-index"):
    processed_texts = [preprocess_text(text) for text in texts]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_texts).toarray()
    dimension = vectors.shape[1]

    # List existing indexes
    existing_indexes = pc.list_indexes().names()

    # Create index if not exists
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(index_name)

    # Prepare data for upsert
    upsert_request = [{"id": str(i), "values": vector.tolist()} for i, vector in enumerate(vectors)]
    index.upsert(vectors=upsert_request)

    return index, vectorizer

def query_openai_api(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(  # Correct API usage
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def handle_queries(index, vectorizer, texts):
    while True:
        q = input("Enter your question (or 'exit' to quit): ")
        if q.lower() == 'exit':
            break
        processed_query = preprocess_text(q)
        query_vector = vectorizer.transform([processed_query]).toarray().tolist()
        result = index.query(vector=query_vector[0], top_k=1, include_metadata=True)

        if not result['matches']:
            print("No relevant document found.")
            continue

        closest_text = texts[int(result['matches'][0]['id'])]
        prompt = f"Relevant content: {closest_text}\n\nUser question: {q}\nAnswer:"
        response = query_openai_api(prompt)
        print('------------')
        print(response)

if __name__ == "__main__":
    Path("storage").mkdir(parents=True, exist_ok=True)  # Ensure storage directory exists
    storage_file = Path("storage/index.json")
    data_dir = Path(r"C:\Users\ASUS\Documents\ML\LLM_open_ai")

    if not storage_file.exists():
        print("Creating index from documents...")
        texts = read_files_from_directory(data_dir)
        if not texts:
            print("No text files found. Exiting.")
            exit()
        index, vectorizer = create_pinecone_index(texts)
        
        with open("storage/vectorizer.json", "w", encoding='utf-8') as f:
            json.dump(vectorizer.get_feature_names_out().tolist(), f)
        
        with open("storage/texts.json", "w", encoding='utf-8') as f:
            json.dump(texts, f)
    else:
        print("Index already exists. Loading index...")
        index = pc.Index("document-index")
        with open("storage/vectorizer.json", "r", encoding='utf-8') as f:
            feature_names = json.load(f)
            vectorizer = TfidfVectorizer(vocabulary=feature_names)

        with open("storage/texts.json", "r", encoding='utf-8') as f:
            texts = json.load(f)

    handle_queries(index, vectorizer, texts)
