from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Or any other LLM you're using
from langchain.prompts import PromptTemplate

# Constants
VECTOR_STORE_DIR = "./ragdb"

# Load vectorstore
embedding_model = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# Define the prompt template
template = """
You are an expert AI assistant. Answer the question using only the context provided below.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Use OpenAI or switch to Groq API or HuggingFace model if needed
llm = OpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Ask a question
query = input("Enter your query: ")
result = qa_chain.run(query)
print("\nAnswer:", result)
