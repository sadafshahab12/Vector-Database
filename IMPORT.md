import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Explanation:

os: Helps us interact with files and folders (like reading all files in a folder).

TextLoader: Loads text or markdown files into Python so we can process them.

RecursiveCharacterTextSplitter: Splits long texts into smaller chunks so embeddings can handle them efficiently.

GoogleGenerativeAIEmbeddings: Converts text into vectors (numbers) which represent the meaning of the text.

Chroma: Vector database to store embeddings so we can do semantic search later.

load_dotenv: Loads API keys and secrets from .env file safely.
