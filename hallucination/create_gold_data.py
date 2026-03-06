from datasets import load_dataset
import pandas as pd

wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:20000]"
)

rows = []

for row in wiki:

    text = row["text"]

    if len(text) < 500:
        continue

    sentences = text.split(". ")

    context = ". ".join(sentences[:3])

    rows.append({
        "query": "What is described in the following context?",
        "context": context,
        "ground_truth": sentences[0],
        "doc_id": row["id"]
    })

    if len(rows) >= 4000:
        break

df = pd.DataFrame(rows)

df.to_csv("gold_dataset_4000.csv", index=False)