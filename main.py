import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

st.set_page_config(page_title="Personal Notes AI", layout="centered")
st.title("📘 Personal Notes Semantic Search")

DB_PATH = "chroma_db"
TEMP_UPLOAD_DIR = "temp_uploads"

# Create temp folder for uploads
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Initialize embeddings and vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
if os.path.exists(DB_PATH):
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    db = None

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

qa_chain = None
if db:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

# --- Upload Notes ---
st.subheader("Upload Notes (PDF, TXT, or MD)")
uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    all_documents = []

    for uploaded_file in uploaded_files:
        # Save temporarily
        file_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load content
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        all_documents.extend(loader.load())

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)

    # Add to vector DB
    if db:
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(chunks, embedding_function=embeddings, persist_directory=DB_PATH)

    db.persist()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    st.success(f"✅ Uploaded and indexed {len(uploaded_files)} files successfully!")

# --- Query ---
if qa_chain:
    query = st.text_input("Ask a question about your notes:")
    if query:
        result = qa_chain(query)
        st.subheader("🤖 Answer")
        st.write(result["result"])

        with st.expander("📚 Sources"):
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                st.write(f"📄 {source}: {doc.page_content[:500]}...")  # first 500 chars
