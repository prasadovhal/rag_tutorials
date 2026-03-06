# Load document
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data.txt")
documents = loader.load()

# Chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# embedding: bi-encoder

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector db

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embedding_model)

# retrieval

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LLM
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not in the context, say "Not found in context."

Context:
{context}

Question:
{question}
""")

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini")


# RAG chain
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {
        "context": retriever,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Run Query ---
response = rag_chain.invoke("What is maternity leave duration?")
print(response)

