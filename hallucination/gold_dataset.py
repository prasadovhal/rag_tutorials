gold_dataset = [
    {"query":"Who invented the telephone?",
    "answer":"Alexander Graham Bell"},
    {"query":"Capital of France?",
    "answer":"Paris"}
]

for item in gold_dataset:
    prediction = rag_pipeline(item["query"])
    print(prediction, item["answer"])