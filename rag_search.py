from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from dotenv import load_dotenv

load_dotenv()

DB_PATH = "chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperatur=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

query = input("Ask a question:")
result = qa_chain(query)

print("\n🤖 Answer:\n")
print(result["result"])

print("\n📚 Sources:\n")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "Unknown"))
