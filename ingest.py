# Convert Notes → Vectors (Ingestion)
# ingest.py

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()

DATA_PATH = "data"
DB_PATH = "chroma_db"

documents = []
# Load all text & markdown files
for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)

    if file.endswith(".txt") or file.endswith(".md"):
        loader = TextLoader(path)
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        continue

    documents.extend(loader.load())

# chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

print(DB_PATH)
# store in chromadb
db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_PATH)

print("✅ Notes successfully ingested into vector database")
