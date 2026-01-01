from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = "chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

query = input("ask question?")
embeddings.embed_query(query)
results = db.similarity_search(query, k=2)
print("\n🔍 Most Relevant Notes:\n")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}\n")
