import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Load
loader = TextLoader("data.txt")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# Bi-Encoder embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)

# Retrieve top 20 candidates
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})


# Add Cross-Encoder
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# rag pipeline with re-ranking

def cross_encoder_rag(query):
    # Step 1: Retrieve top 20
    retrieved_docs = retriever.invoke(query)

    # Step 2: Prepare pairs
    pairs = [(query, doc.page_content) for doc in retrieved_docs]

    # Step 3: Score with cross-encoder
    scores = cross_encoder.predict(pairs)

    # Step 4: Sort by score
    reranked = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Step 5: Take top 3
    top_docs = [doc.page_content for doc, score in reranked[:3]]

    return top_docs

# LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If not found, say "Not found in context."

Context:
{context}

Question:
{question}
""")

def generate_answer(query):
    top_docs = cross_encoder_rag(query)

    context = "\n\n".join(top_docs)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query
    })

print(generate_answer("What is maternity leave duration?"))