import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load & split
loader = TextLoader("data.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma DB
vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


##### Qdrant

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # local

vectorstore = Qdrant.from_documents(
    docs,
    embedding_model,
    location=":memory:",
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


##### Pinecone

from constant import pinecone_api_key
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LCPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Create index if not exists
if "rag-index" not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name="rag-index",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index (IMPORTANT: use pc.Index)
index = pc.Index("rag-index")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LangChain wrapper
vectorstore = LCPinecone(
    index=index,
    embedding=embedding_model,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

### weaviate
# needs docker

from langchain_community.vectorstores import Weaviate
import weaviate

client = weaviate.connect_to_local()

vectorstore = Weaviate.from_documents(
    docs,
    embedding_model,
    client=client,
    index_name="RagIndex"
)

### Milvus

from langchain_community.vectorstores import Milvus

vectorstore = Milvus.from_documents(
    docs,
    embedding_model,
    connection_args={"host": "localhost", "port": "19530"}
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


### Elasticsearch

from langchain_community.vectorstores import ElasticsearchStore

vectorstore = ElasticsearchStore.from_documents(
    docs,
    embedding_model,
    es_url="http://localhost:9200",
    index_name="rag-index"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})