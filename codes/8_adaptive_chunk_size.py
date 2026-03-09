from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
DATA_PATH = "D:/Study/Git_repo/rag_tutorials/codes/"

doc = converter.convert(DATA_PATH + "rag_guide.pdf")
text = doc.document.export_to_markdown()


#######################################################

# Structure Based Chunking
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header1"),
        ("##", "Header2")
    ]
)

chunks = splitter.split_text(text)

#######################################################

# Semantic Chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(embedding_model)
docs = text_splitter.create_documents([text])

#######################################################

# Token Budget Adaptive Chunking 
# depend on context window of the LLM and the number of retrieved documents, we can compute the chunk size to fit in the context window


MODEL_CONTEXT = {
    "llama3": 8192,
    "gpt4": 128000,
    "claude": 200000
}


def compute_chunk_size(model_name, retrieval_docs=4):
    
    context_window = MODEL_CONTEXT[model_name]

    # allocate 40% for documents
    doc_budget = int(context_window * 0.4)

    chunk_size = doc_budget // retrieval_docs

    return chunk_size

chunk_size = compute_chunk_size("llama3")

print(chunk_size)

#######################################################

# Importance Based Chunking

IMPORTANT_KEYWORDS = [
    "definition",
    "equation",
    "theorem",
    "formula",
    "important",
    "note",
    "key point"
]

def is_important(sentence):
    
    s = sentence.lower()
    
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in s:
            return True
            
    return False

from langchain_text_splitters import RecursiveCharacterTextSplitter

def importance_chunker(text, chunk_size=500):

    sentences = text.split(".")
    
    chunks = []
    current_chunk = ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    for sentence in sentences:
        
        sentence = sentence.strip() + "."

        if is_important(sentence):
            
            # save previous chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # important sentence becomes its own chunk
            chunks.append(sentence)

        else:
            
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

test_text = """

Definition: Reinforcement learning is a learning method based on rewards.

Agents interact with environment.

Equation: Q(s,a) = r + γ max Q(s',a')

Training continues until convergence.

Important: exploration vs exploitation tradeoff is critical.
"""

chunks = importance_chunker(test_text)

for i, c in enumerate(chunks):
    print(f"\nChunk {i+1}\n", c)
    
    
## Note

"""
    structure chunking
    + importance chunking
    + semantic chunking
    + token budget chunking
    
"""