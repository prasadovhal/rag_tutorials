import pandas as pd

## load data
DATA_PATH = "D:/Study/Git_repo/rag_tutorials/hallucination/"

df = pd.read_csv(DATA_PATH + "gold_dataset_4000.csv")

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

import time

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate

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

    answer = llm.invoke(prompt)
    # answer = llm.invoke(prompt).content for google genai or chatgpt
    time.sleep(1)
    return answer, context

    
"""
with ollama make sure follow below steps first before running the model:
1. Install ollama from https://ollama.com/, 
    irm https://ollama.com/install.ps1 | iex
2. Pull the model you want to use, for example mistral
    ollama pull mistral
3. Run ollama server in terminal
    ollama serve
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

# on actual data
results_local = []

for _, row in df.iloc[:150, :].iterrows():
    print(_)
    query = row["query"]

    answer, context = rag_pipeline(query)
    
    results_local.append({
        "user_input": query,
        "response": answer,
        "retrieved_contexts": [context],
        "reference": row["ground_truth"]
    })

import json
with open(DATA_PATH + "model_prediction_results_local.json", "w") as f:
    json.dump(results_local, f, indent=4)
#####################################################################

######## Evaluation starts ####################

## convert results to dataframe
import json

# Reading the file
with open(DATA_PATH + "model_prediction_results_local.json", "r") as f:
    results_local = json.load(f)
    
from datasets import Dataset

results_local = pd.DataFrame(results_local)
# results_local.rename(columns={"query": "user_input",
#                               "answer": "response",
#                               "contexts": "retrieved_contexts",
#                               "ground_truth": "reference"
#                               }, inplace=True)  
# results_local["retrieved_contexts"] = results_local["retrieved_contexts"].apply(lambda x: [x])

## temp fix

# results_local["retrieved_contexts"] = results_local["retrieved_contexts"].apply(lambda x: x[0]).apply(lambda x: x[0])

eval_dataset = Dataset.from_pandas(results_local.iloc[:20, :])

eval_dataset.to_csv(DATA_PATH + "rag_evaluation_results.csv", 
                    index=False)

#############################################

## evaluation using RAGAS

"""
Metrics computed:
    Faithfulness
    Answer Relevance
    Context Precision
    Context Recall
    Context Relevance

"""

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# wrap the LLM for RAGAS evaluation as using other than defualt openai models
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

metrics=[
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm),
    ContextPrecision(),
    ContextRecall(),
    ContextRelevance()
]

result = evaluate(
    eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings
)


print(result)

#############################################

# Answer Correctness (DeepEval)

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

## create evaluation model
from deepeval.models import OllamaModel
evaluation_model = OllamaModel(
    model="mistral"
)

metric = AnswerRelevancyMetric(model=evaluation_model)

scores = []
results_local = results_local.to_dict(orient="records")

for r in results_local[:20]:

    test_case = LLMTestCase(
        input=r["user_input"],
        actual_output=r["response"],
        expected_output=r["reference"]
    )

    metric.measure(test_case)

    scores.append(metric.score)

print("Answer Correctness:", sum(scores)/len(scores))

#############################################

# Hallucination Score

import re

def hallucination_score(answer, context):
    # Lowercase and remove punctuation
    clean_answer = re.sub(r'[^\w\s]', '', answer.lower())
    clean_context = re.sub(r'[^\w\s]', '', context.lower())
    
    words = clean_answer.split()
    context_words = set(clean_context.split()) # Use a set for O(1) lookup
    
    unsupported = sum(1 for w in words if w not in context_words)
    return unsupported / len(words) if words else 0

scores = []

for r in results_local:
    score = hallucination_score(r["response"], r["retrieved_contexts"][0])
    scores.append(score)

hallucination_rate = sum(scores)/len(scores)

print("Hallucination Score:", hallucination_rate)

### using cosine similarity between answer and context embeddings
# this seems more correct

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def hallucination_score_similarity(answer, contexts):

    context = " ".join(contexts)

    emb1 = model.encode(answer)
    emb2 = model.encode(context)

    sim = cosine_similarity([emb1],[emb2])[0][0]

    return 1 - sim

scores = []

for r in results_local:
    score = hallucination_score_similarity(r["response"], r["retrieved_contexts"])
    scores.append(score)

hallucination_rate = sum(scores)/len(scores)

print("Hallucination Score:", hallucination_rate)

#############################################

# Retrieval Accuracy

correct = 0
total_samples = len(df)

for i, row in df.iterrows():
    docs = retriever.invoke(row["query"])
    
    # 1. Standardize the target ID (from your CSV)
    target = str(row["doc_id"]).strip().lower()
    
    # 2. Extract and clean IDs from retrieved docs
    # We use .get() to avoid KeyErrors if metadata is missing
    retrieved_ids = [str(d.metadata.get("doc_id", d.metadata.get("source", ""))).strip().lower() for d in docs]
    
    # 3. Perform a "Smart Match" 
    # This checks if the target is exactly in the list OR 
    # if the target is a SUBSTRING of any retrieved ID (handles doc_1 vs doc_1_chunk_0)
    match_found = False
    for r_id in retrieved_ids:
        if target == r_id or target in r_id or os.path.basename(r_id).startswith(target):
            match_found = True
            break
            
    if match_found:
        correct += 1
    elif i < 5: # Debug the first 5 misses
        print(f"Debug Row {i}:")
        print(f"  Expected: '{target}'")
        print(f"  Actual in Metadata: {retrieved_ids}")

final_accuracy = correct / total_samples
print(f"\nAdjusted Retrieval Accuracy: {final_accuracy:.4f}")


#############################################

# Context Quality

from sentence_transformers import SentenceTransformer, util

# Load a lightweight, fast model
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_context_quality(contexts, ground_truth):
    # 1. Combine chunks into one block of text
    context_text = " ".join(contexts)
    
    # 2. Encode both into vectors (embeddings)
    context_emb = model.encode(context_text, convert_to_tensor=True)
    truth_emb = model.encode(ground_truth, convert_to_tensor=True)
    
    # 3. Calculate Cosine Similarity
    score = util.cos_sim(context_emb, truth_emb)
    return score.item()

scores = []

for r in results_local:
    score = semantic_context_quality(r["retrieved_contexts"], r["reference"])
    scores.append(score)

avg_context_quality = sum(scores) / len(scores)

print("Context Quality:", avg_context_quality)


## using bert score
from bert_score import score

def calculate_bertscore(contexts, ground_truth):
    candidate = [" ".join(contexts)]
    reference = [ground_truth]
    
    # P = Precision, R = Recall, F1 = F1 Score
    P, R, F1 = score(candidate, reference, lang="en", verbose=True)
    return F1

scores = []

for r in results_local:
    score = calculate_bertscore(r["retrieved_contexts"], r["reference"])
    scores.append(score)

avg_context_quality = sum(scores) / len(scores)

print("Context Quality:", avg_context_quality)

#############################################

# Confidence Scoring

scores = vectorstore.similarity_search_with_score(query)

confidence = 1 - scores[0][1]

print("Confidence:", confidence)



# guiderails trigger

def guardrail(answer, confidence):

    if confidence < 0.5:
        return "LOW_CONFIDENCE"

    if len(answer) < 5:
        return "INVALID_RESPONSE"

    return "OK"


#############################################

# LLM as a judge

from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

import re

def LLM_judge(query, answer, ground_truth):
    # 1. Stricter Prompt
    prompt = f"""
    You are an expert evaluator. Compare the Answer against the Ground Truth based on the Query.
    
    Query: {query}
    Answer: {answer}
    Ground Truth: {ground_truth}

    Give a score between 0.0 and 1.0. 
    1.0 is perfectly correct. 0.0 is completely wrong.
    Output ONLY the numerical score. Do not provide any explanation.
    """

    raw_output = llm.invoke(prompt).strip()
    
    match = re.search(r"([0-9]*\.?[0-9]+)", raw_output)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    else:
        print(f"Warning: Could not find a score in output: {raw_output}")
        return 0.0
    
scores = []

for r in results_local:
    score = LLM_judge(r['user_input'], r['response'], r["reference"])
    scores.append(score)

avg_context_quality = sum(scores) / len(scores)

print("Context Quality:", avg_context_quality)

#############################################

# TruLens Observability

from trulens_eval import Tru

tru = Tru()

tru.run_dashboard()

#############################################

# LangSmith Experiment Tracking

from langsmith import Client

client = Client()

client.create_run(
    name="rag-evaluation",
    inputs={"query":query},
    outputs={"answer":answer}
)

#############################################

# Latency Measurement

import time

latencies = []

for _, row in df.iloc[:5, :].iterrows():

    start = time.time()

    docs = retriever.invoke(row["query"])
    context = " ".join([d.page_content for d in docs])

    prompt = f"""
    Context: {context}
    Question: {row['query']}
    """

    answer = llm.invoke(prompt).strip()

    end = time.time()

    latency = end - start
    latencies.append(latency)

avg_latency = sum(latencies) / len(latencies)

print("Average Latency:", avg_latency)


"More Detailed Latency Breakdown"

retrieval_latencies = []
generation_latencies = []

for _, row in df.iterrows():

    start_r = time.time()
    docs = retriever.get_relevant_documents(row["query"])
    retrieval_latencies.append(time.time() - start_r)

    context = " ".join([d.page_content for d in docs])

    prompt = f"{context}\n{row['query']}"

    start_g = time.time()
    answer = llm(prompt)
    generation_latencies.append(time.time() - start_g)

print("Retrieval latency:", sum(retrieval_latencies)/len(retrieval_latencies))
print("Generation latency:", sum(generation_latencies)/len(generation_latencies))

#############################################

# Token Usage

import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(enc.encode(text))

token_stats = []

for r in results_local:

    input_tokens = count_tokens(r["contexts"] + r["query"])
    output_tokens = count_tokens(r["answer"])

    token_stats.append({
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens
    })

avg_input = sum(t["input"] for t in token_stats)/len(token_stats)
avg_output = sum(t["output"] for t in token_stats)/len(token_stats)
avg_total = sum(t["total"] for t in token_stats)/len(token_stats)

print("Avg input tokens:", avg_input)
print("Avg output tokens:", avg_output)
print("Avg total tokens:", avg_total)


#############################################

# Cost Per Query

input_price = 0.01 / 1000
output_price = 0.03 / 1000

costs = []

for t in token_stats:

    cost = (t["input"] * input_price) + (t["output"] * output_price)
    costs.append(cost)

avg_cost = sum(costs)/len(costs)

print("Average Cost Per Query:", avg_cost)


#############################################

# Drift Detection

previous_metrics = {
    "hallucination_rate": 0.08,
    "retrieval_accuracy": 0.92
}

current_metrics = {
    "hallucination_rate": hallucination_rate,
    "retrieval_accuracy": retrieval_accuracy
}

drift_report = {}

for key in previous_metrics:

    change = abs(current_metrics[key] - previous_metrics[key])

    if change > 0.05:
        drift_report[key] = "DRIFT DETECTED"
    else:
        drift_report[key] = "STABLE"

print(drift_report)


"Embedding Drift Detection"

from sklearn.metrics.pairwise import cosine_similarity

old_embeddings = embedding_model.embed_query("old query example")
new_embeddings = embedding_model.embed_query("new query example")

similarity = cosine_similarity([old_embeddings], [new_embeddings])[0][0]

print("Embedding similarity:", similarity)

#############################################

# Failure clusters

## Step 1 — Collect Failure Cases

failures = []

for r in results_local:

    score = hallucination_score(r["answer"], r["contexts"])

    if score > 0.4:
        failures.append(r["query"])

## Step 2 — Embed Failure Queries

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

failure_embeddings = model.encode(failures)

## Step 3 — Cluster Failures

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

clusters = kmeans.fit_predict(failure_embeddings)

## Step 4 — Inspect Clusters

cluster_map = {}

for query, c in zip(failures, clusters):

    if c not in cluster_map:
        cluster_map[c] = []

    cluster_map[c].append(query)

for c in cluster_map:
    print("Cluster", c)
    print(cluster_map[c][:5])
    
#############################################