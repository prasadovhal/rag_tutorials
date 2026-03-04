"""
Conversation memory in Retrieval-Augmented Generation (RAG) allows the system to remember previous user interactions so responses remain context-aware across turns. 
This is critical for chatbots, copilots, and assistants.

User Question + Chat History → Retrieve Documents → LLM → Answer
"""

# Step 1 — Parse Documents using Docling

from docling.document_converter import DocumentConverter

converter = DocumentConverter()

doc = converter.convert("rag_guide.pdf")

text = doc.document.export_to_markdown()

print(text[:1000])

# split into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Step 2 — Create Vector Database

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_texts(chunks, embeddings)

# retreiver
retriever = vector_db.as_retriever()


# connect to LLM
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import google.generativeai as genai
from constant import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# RAG answer function

def rag_answer(query):

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    return response.text

rag_answer("What is retrieval augmented generation?")


# Step 4 — Add Basic Conversation Memory

conversation_memory = []

def chat(query, conversation_memory = []):

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    history = "\n".join(conversation_memory)

    prompt = f"""
    Conversation History:
    {history}

    Context:
    {context}

    User Question:
    {query}

    Answer:
    """

    response = model.generate_content(prompt)

    answer = response.text

    conversation_memory.append(f"User: {query}")
    conversation_memory.append(f"Assistant: {answer}")

    # Maintain sliding window of conversation memory
    if len(conversation_memory) > MAX_MEMORY:
        conversation_memory = conversation_memory[-MAX_MEMORY:]
    
    return answer, conversation_memory

answer, conversation_memory = chat("Explain RAG", conversation_memory)
answer, conversation_memory = chat("What are its advantages?", conversation_memory)


# Step 5 — Sliding Window Memory

MAX_MEMORY = 6
# conversation_memory = []

# Step 6 — Conversation Summary Memory (Advanced)

summary_memory = ""

def summarize_memory():

    global summary_memory

    prompt = f"""
    Summarize this conversation briefly.

    {conversation_memory}
    """

    response = model.generate_content(prompt)

    summary_memory = response.text
    
    return summary_memory


summarize_memory()

