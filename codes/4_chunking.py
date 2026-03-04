# manual way of chunking

from pathlib import Path

text = Path("data.txt").read_text()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(text)

####################################################################

# token based chunking

from langchain_core.documents import Document
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def token_chunk(text, max_tokens=300):
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = enc.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

token_based_chunk = token_chunk(text, max_tokens=300)

####################################################################

# Recursive Chunking 

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

####################################################################

# parent doc chunking

# step 1 - create parent chunks
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data.txt")
text = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

parent_docs = parent_splitter.split_documents(text)


# step 2 - create child chunks
from langchain_core.documents import Document

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

child_docs = []

for i, parent in enumerate(parent_docs):
    children = child_splitter.split_text(parent.page_content)

    for child in children:
        child_docs.append(
            Document(
                page_content=child,
                metadata={"parent_id": i}
            )
        )


# step 3 - store child in FAISS

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(child_docs, embedding_model)

# step 4 - store parent on storage for retrieval

from pathlib import Path

parent_store = Path("parent_store")
parent_store.mkdir(exist_ok=True)

for parent_id, parent_doc in enumerate(parent_docs):
    file_path = parent_store / f"{parent_id}.txt"
    file_path.write_text(parent_doc.page_content)

# step 5 - retrieval and expand parent

def retrieve_with_parent(query, k=5):
    retrieved_children = vectorstore.similarity_search(query, k=k)

    parent_ids = {doc.metadata["parent_id"] for doc in retrieved_children}

    parents = []

    for pid in parent_ids:
        file_path = Path("parent_store") / f"{pid}.txt"
        if file_path.exists():
            parents.append(Document(page_content=file_path.read_text()))

    return parents

results = retrieve_with_parent("What is maternity leave?")
print(results[0])



###########################################################################

