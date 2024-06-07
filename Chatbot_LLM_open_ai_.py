
import os
import openai
import json
from pathlib import Path
import numpy as np
from pinecone import Pinecone, ServerlessSpec, QueryRequest, UpsertRequest
from sklearn.feature_extraction.text import TfidfVectorizer

openai_api_key=" " #set your open api key please
# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Function to read files from a directory and combine them into a list of texts
def read_files_from_directory(directory_path):
    texts = []
    for file_path in Path(directory_path).rglob('*.txt'):
        print(f"Reading file: {file_path}")  # Debugging: Print file paths
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Convert tokens back to text
    processed_text = ' '.join(tokens)

    return processed_text

def create_pinecone_index(texts, index_name="document-index"):
    # Check if there are enough non-stop words in the documents after preprocessing
    if not any(len(preprocess_text(text).split()) > 10 for text in texts):
        raise ValueError("Documents do not contain enough non-stop words for indexing.")
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()
    
    dimension = vectors.shape[1]
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='euclidean',
            spec=ServerlessSpec(cloud='gcp', region='us-west1')
        )
    
    index = pc.Index(index_name)

    # Add vectors to the index
    upsert_request = UpsertRequest(
        vectors=[(str(i), vector.tolist()) for i, vector in enumerate(vectors)]
    )
    index.upsert(upsert_request)

    return index, vectorizer

# Function to query the OpenAI API
def query_openai_api(prompt, api_key, model="text-davinci-003"):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Function to handle user queries
def handle_queries(index, vectorizer, texts, api_key):
    while True:
        q = input("Enter your question (or 'exit' to quit): ")
        if q.lower() == 'exit':
            break
        elif not q.strip():  # Validate input
            print("Please enter a valid question.")
            continue
        
        # Convert the query to a vector
        query_vector = vectorizer.transform([q]).toarray().tolist()

        # Search the index for the nearest neighbors
        query_request = QueryRequest(queries=query_vector, top_k=1)
        result = index.query(query_request)
        closest_text = texts[int(result.matches[0].id)]

        prompt = f"Relevant content: {closest_text}\n\nUser question: {q}\nAnswer:"
        response = query_openai_api(prompt, api_key)
        print('------------')
        print(response)

if __name__ == "__main__":
    storage_file = Path("storage/index.json")
    data_dir = Path("C:/Users/ASUS/Documents/ML/LLM_open_ai")  # Updated data directory

    if not storage_file.exists():
        print("Creating index from documents...")
        texts = read_files_from_directory(data_dir)
        
        # Debugging: Print the number of documents read and their content lengths
        print(f"Number of documents read: {len(texts)}")
        for i, text in enumerate(texts):
            print(f"Document {i} length: {len(text.split())} words")
            print(f"Sample text: {text[:200]}")  # Print a sample of the text
        
        try:
            index, vectorizer = create_pinecone_index(texts)
            
            # Save vectorizer
            with open("storage/vectorizer.json", "w", encoding='utf-8') as f:
                json.dump(vectorizer.get_feature_names_out().tolist(), f)
            
            # Save texts
            with open("storage/texts.json", "w", encoding='utf-8') as f:
                json.dump(texts, f)
        except ValueError as e:
            print(e)
    else:
        print("Index already exists. Loading index...")
        index=pc.Index(index_name)
        with open("storage/vectorizer.json", "r", encoding='utf-8') as f:
            feature_names = json.load(f)
            vectorizer = TfidfVectorizer(vocabulary=feature_names)

        with open("storage/texts.json", "r", encoding='utf-8') as f:
            texts = json.load(f)

        if texts:
            handle_queries(index, vectorizer, texts, openai_api_key)
