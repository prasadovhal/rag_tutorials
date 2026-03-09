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
