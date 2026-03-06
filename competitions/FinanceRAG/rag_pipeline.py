# load corpus

import json

corpus = []

DATA_PATH = "D:/Study/Git_repo/rag_tutorials/competitions/FinanceRAG/data/"

with open(DATA_PATH + "finqa_corpus.jsonl/corpus.jsonl", "r") as f:
    for line in f:
        corpus.append(json.loads(line))

print(corpus[0])

documents = [doc["text"] for doc in corpus]
doc_ids = [doc["_id"] for doc in corpus]

###############################################################

# chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = []
chunk_doc_map = []

for doc in corpus:

    pieces = splitter.split_text(doc["text"])

    for piece in pieces:
        chunks.append(piece)
        chunk_doc_map.append(doc["_id"])

###############################################################

# create embeddings and vector database

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-base-en")

embeddings = embedding_model.encode(chunks)

###############################################################

# Build FAISS vector index

import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

###############################################################

# Dense retrieval function

def dense_search(query, k=10):

    q_embedding = embedding_model.encode([query])

    scores, indices = index.search(q_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(chunk_doc_map[idx])

    return results

# load queries

queries = []

with open(DATA_PATH + "finqa_queries.jsonl/queries.jsonl", "r") as f:
    for line in f:
        queries.append(json.loads(line))
        
print(queries[0])

###############################################################

# run dense retrieval on queries

predictions = {}

for q in queries:
    docs = dense_search(q["text"])
    predictions[q["_id"]] = docs

###############################################################

# Evaluate Using Ground Truth

import pandas as pd

qrels = pd.read_csv(
    DATA_PATH + "FinQA_qrels.tsv",
    sep="\t"
)

ground_truth = dict(
    zip(qrels["query_id"], qrels["corpus_id"])
)

###############################################################

## calculate recall@k

correct = 0
total = 0

for qid, docs in predictions.items():

    if qid not in ground_truth:
        continue   # skip queries without ground truth

    total += 1

    if ground_truth[qid] in docs:
        correct += 1

recall = correct / total

print("Recall@10:", recall)

###############################################################

# Add BM25 Retrieval

## Build index

from rank_bm25 import BM25Okapi

tokenized_chunks = [chunk.lower().split() for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)

## search

def bm25_search(query, k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]

    return [chunk_doc_map[i] for i in top_indices]

###############################################################

# hybrid retrieval

def hybrid_search(query):

    dense_docs = dense_search(query, 10)
    bm25_docs = bm25_search(query, 10)

    candidates = list(set(dense_docs + bm25_docs))

    return candidates

###############################################################

# cross encoder re-ranking

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

## re-rank candidates

def rerank(query, candidate_texts):

    pairs = [(query, doc) for doc in candidate_texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_texts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:5]]

###############################################################

def retrieve(query):

    # Step 1: get candidate doc_ids
    candidate_doc_ids = hybrid_search(query)

    # Step 2: convert doc_ids → text
    candidate_texts = [
        docid_to_text[d]
        for d in candidate_doc_ids
        
    ]

    # Step 3: rerank using cross-encoder
    top_texts = rerank(query, candidate_texts)

    # Step 4: convert text → doc_id
    text_to_docid = {v: k for k, v in docid_to_text.items()}

    top_doc_ids = [
        text_to_docid[t]
        for t in top_texts
    ]

    return top_doc_ids


docid_to_text = {}

for doc in corpus:
    docid_to_text[doc["_id"]] = doc["title"] + " " + doc["text"]

###############################################################

predictions = {}

for q in queries:

    query_id = q["_id"]
    query = q["text"]

    docs = retrieve(query)

    predictions[query_id] = docs

###############################################################

# create submission file

rows = []

for qid, docs in predictions.items():

    rows.append({
        "query_id": qid,
        "doc_id": docs[0]
    })

submission = pd.DataFrame(rows)

submission.to_csv("submission.csv", index=False)