from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Constants
DOC_PATH = "./docs/mydoc.txt"
VECTOR_STORE_DIR = "./ragdb"

# Step 0: Check if file exists
if not os.path.exists(DOC_PATH):
    raise FileNotFoundError(f"[X] Document not found at path: {DOC_PATH}")

# Step 1: Load the document
loader = TextLoader(DOC_PATH)
documents = loader.load()

# Step 2: Split document into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Step 3: Initialize embedding model (✅ specify model_name)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store the embeddings in Chroma vector DB
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=VECTOR_STORE_DIR
)

print(f"[✓] RAG index created and stored successfully in: {VECTOR_STORE_DIR}")

# ✅ Add get_context function for use in app.py
def get_context(query: str, k: int = 3) -> str:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
    docs = vectordb.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
