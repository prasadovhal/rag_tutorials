import pandas as pd

## load data

df = pd.read_csv("gold_dataset_4000.csv")

df.head()

from langchain_core.documents import Document

docs = []

for _, row in df.iterrows():
    docs.append(Document(page_content=row["context"],
                         metadata={"doc_id": row["doc_id"]}))

## create chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

## create embeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

## create FAISS vector database
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# test retrieval

query = "Who developed the theory of relativity?"

docs = retriever.invoke(query)

for d in docs:
    print(d.page_content[:200])
    
## RAG prompt

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import time

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )

def rag_pipeline(query):

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    answer = llm.invoke(prompt).content
    time.sleep(4)
    return answer, context

    
"""
with ollama make sure follow below steps first before running the model:
1. Install ollama from https://ollama.com/, 
    irm https://ollama.com/install.ps1 | iex
2. Run ollama server in terminal
    ollama serve
3. Pull the model you want to use, for example mistral
    ollama pull mistral
4. test manually if the model is working
    ollama run mistral
"""

from langchain_community.llms import Ollama
llm = Ollama(model="mistral")

"""
Alternative models:
llama3
phi3
gemma
"""

# test the ollama model

print(llm.invoke("Who invented the telephone?"))


results_local = []

for _, row in df.iterrows():
    print(_)
    query = row["query"]

    answer, context = rag_pipeline(query)
    
    results_local.append({
        "query": query,
        "answer": answer,
        "contexts": context,
        "ground_truth": row["ground_truth"]
    })
    
