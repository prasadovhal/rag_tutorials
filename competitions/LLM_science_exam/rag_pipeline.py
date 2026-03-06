import pandas as pd

## load data

DATA_PATH = "D:/Study/Git_repo/rag_tutorials/competitions/LLM_science_exam/data/"

train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)

train.head()
test.head()

wiki_corpus = pd.read_csv(DATA_PATH + "wiki_stem_corpus.csv")
wiki_corpus.head()
wiki_corpus = wiki_corpus[["content_id", "page_title", "text"]]

documents = []

for _, row in wiki_corpus.iterrows():
    doc = {
        "id": row["content_id"],
        "text": row["page_title"] + " " + row["text"]
    }
    documents.append(doc)
    
###########################################################

# doc chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = []
chunk_to_doc = []

for doc in documents:

    pieces = splitter.split_text(doc["text"])

    for p in pieces:
        chunks.append(p)
        chunk_to_doc.append(doc["id"])
        
########################################################################

# create embeddings

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-base-en")

embeddings = embedding_model.encode(chunks, show_progress_bar=True)


########################################################################

# vector database

import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.bin")

import pickle

with open("chunk_meta.pkl", "wb") as f:
    pickle.dump({
        "chunks": chunks,
        "ids": chunk_to_doc
    }, f)

########################################################################

# Dense retrieval function

def dense_search(query, k=10):

    q_embedding = embed_model.encode([query])

    scores, indices = index.search(q_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(chunk_to_doc[idx])

    return results

########################################################################

# Add BM25 Retrieval

from rank_bm25 import BM25Okapi

tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def bm25_search(query, k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]

    return [chunk_to_doc[i] for i in top_indices]


########################################################################

# hybrid retrieval

def hybrid_search(query):

    dense_docs = dense_search(query, 10)
    bm25_docs = bm25_search(query, 10)

    candidates = list(set(dense_docs + bm25_docs))

    return candidates

########################################################################

# cross encoder re-ranking

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

## re-rank candidates

def rerank(query, candidate_texts, top_k=5):

    pairs = [(query, doc) for doc in candidate_texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_texts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:top_k]]

######################################################################

