### Hybrid Retrieval (Dense + BM25)

"""
Ranking by score

score = term_frequency × inverse_document_frequency
         normalized by document length

in RAG context

Query → BM25 → candidate docs
Query → Dense → candidate docs
Merge → Re-rank → LLM

"""
## load data

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load raw text
text = Path("data.txt").read_text()

# Sentence-aware chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

## build dense index

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding model
dense_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
dense_embeddings = dense_model.encode(chunks)

dimension = dense_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)  # cosine similarity (with normalized vectors)

# Normalize for cosine similarity
faiss.normalize_L2(dense_embeddings)

faiss_index.add(np.array(dense_embeddings))

## BM25 Index

from rank_bm25 import BM25Okapi

# Tokenize chunks
tokenized_chunks = [chunk.lower().split() for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)

## Hybrid Retrieval function

def hybrid_retrieve(query, k_dense=5, k_bm25=5):
    
    # ---- Dense retrieval ----
    query_embedding = dense_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    dense_scores, dense_indices = faiss_index.search(query_embedding, k_dense)
    dense_results = [chunks[i] for i in dense_indices[0]]
    
    # ---- BM25 retrieval ----
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
    bm25_results = [chunks[i] for i in bm25_indices]
    
    # ---- Merge results ----
    merged = list(set(dense_results + bm25_results))
    
    return merged

## cross encoder re-ranking

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

## final retrieval with re-ranking

def retrieve_with_rerank(query, top_k=3):
    
    # Step 1: Hybrid candidates
    candidates = hybrid_retrieve(query, k_dense=5, k_bm25=5)
    
    # Step 2: Prepare pairs
    pairs = [(query, doc) for doc in candidates]
    
    # Step 3: Score with cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Step 4: Sort by score
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # Step 5: Return top_k
    return [doc for doc, score in ranked[:top_k]]

## testing

results = retrieve_with_rerank("What is maternity leave duration?")

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(res)