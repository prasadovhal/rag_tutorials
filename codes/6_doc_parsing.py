"""
Poor parsing → poor chunks → poor embeddings → poor retrieval.

Document → "Parse" → Chunk → Embed → Vector DB → Retrieve → LLM

what parsing does
1️⃣ Layout reconstruction
2️⃣ Remove noise
3️⃣ Preserve document structure
4️⃣ Table extraction

libraries used for parsing
1️⃣ PyMuPDF (langchain's default)
2️⃣ Docling

"""

# PyMuPDF

from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("policy.pdf")
documents = loader.load()

print(documents[0].page_content)

# docling

from docling.document_converter import DocumentConverter

converter = DocumentConverter()

result = converter.convert("docling_full_tutorial_sample.pdf")

print(result.document.export_to_markdown())

# Working With Document Object

doc = result.document

print(type(doc))

for element in doc.iterate_items():
    print(element)
    
# Export Formats

doc.export_to_markdown()

import json

with open("document.json", "w") as f:
    json.dump(doc.model_dump(), f, indent=2)
    
doc.export_to_text()


# extract tables

tables = doc.tables

for table in tables:
    df = table.export_to_dataframe()
    print(df)
    
for table in doc.tables:
    df = table.export_to_dataframe()
    print(df)
    
# extract sections

for item, level in doc.iterate_items():
    if hasattr(item, "text"):
        print(item.text)
    elif item.label == "table":
        print("TABLE DETECTED")
        

# chunking docs for RAG

markdown = doc.export_to_markdown()

# then use RecursiveCharacterTextSplitter


## using docling with langchain

from langchain_core.documents import Document

mk_down = doc.export_to_markdown()

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(mk_down)

print(chunks[:3])


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = FAISS.from_texts(chunks, embeddings)

results = vectordb.similarity_search("What is the methodology?")

for r in results:
    print(r.page_content)
    

## extract images

for item, level in doc.iterate_items():
    if item.label == "figure":
        print(item)
        
# extract figure captions

for item, level in doc.iterate_items():
    if item.label == "caption":
        print(item.text)
        

# inspect all detected elements

for item, level in doc.iterate_items():
    print(level, item.label)
    

# extract everything safely

content = []

for item, level in doc.iterate_items():

    if hasattr(item, "text") and item.text:
        content.append(item.text)

for table in doc.tables:
    content.append(table.export_to_markdown())

print(content)

# get metadata

print(doc.model_dump().keys())
print(doc.origin)