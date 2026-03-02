# rag_tutorials
Basic RAG understanding, to advance tutorials

## Set up Python & Poetry

1. cd transformers_tutorial
2. install poetry
`(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`
3. run `C:\Users\user_name\AppData\Roaming\Python\Scripts`
4. check poetry version `poetry --version`
5. set `poetry config virtualenvs.in-project true`
6. run `poetry install`
7. set venv 
   - for windows `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` or `.venv\Scripts\activate`
   - for linux/mac `source .venv/bin/activate`

## changes you need to make

1. Create `constant.py` file inside `codes/` folder.
2. Add the following keys inside it:
   - `GOOGLE_API_KEY = "your_google_api_key"`
   - `OPENAI_KEY = "your_openai_key"`
   - `HUGGINGFACE_API_KEY = "your_huggingface_api_key"`
   - `LANGFUSE_SECRET_KEY = "your_langfuse_secret_key"`
   - `LANGFUSE_PUBLIC_KEY = "your_langfuse_public_key"`
   - `LANGFUSE_BASE_URL = "https://cloud.langfuse.com"`


# Topics included in RAG

1. encoding methods
    - Bi-encoder
    - Cross-encoder
2. vectorDB
3. chunking
    - token-based
    - recursive text
    - parent doc chunking
4. doc parsing
5. conversation memory
6. Advance
    - Hybrid-RAG
        - re-ranking
    - Graph-RAG
        - Knowledge graph
    - Agentic RAG
    - Reflection
    - self-ask

# RAG Steps

1. Document Loading: Load PDFs, CSVs, Web pages, DB data.
2. Chunking: Break long docs into smaller pieces.
3. Embedding: Convert chunks into vectors.
4. Store in Vector DB: So we can search via similarity.
5. Retrieval: Convert query → embedding → similarity search.
6. Augment Prompt: Attach retrieved chunks to prompt.
7. Generate Answer: LLM answers using retrieved context.


## What it includes
