'''To ensure that there are text files in the directory'''
'''
from pathlib import Path

data_dir = Path(r"C:\Users\ASUS\Documents\ML\LLM_open_ai")  # Update with your directory
txt_files = list(data_dir.rglob("*.txt"))

if txt_files:
    print("Found text files:")
    for file in txt_files:
        print(file)
else:
    print("No text files found!")'''

from api import gemini_api_key
import google.generativeai as genai
genai.configure(api_key=gemini_api_key)  # Use your actual API key
models = genai.list_models()
for model in models:
    print(model.name)
