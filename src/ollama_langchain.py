from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
# Initialize the Ollama model
llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")
msg = HumanMessage( "What is LangChain?")
messages=[msg]
# Run a test query
response = llm.invoke(messages)
print(response)
