from langchain_ollama import OllamaLLM
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel

# ✅ Define Input Schema
class AddNumbersInput(BaseModel):
    a: int
    b: int

# ✅ Define the function
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together and returns the sum."""
    return a + b

# ✅ Create the tool
add_numbers_tool = StructuredTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Adds two numbers together and returns the sum."
)

# ✅ Initialize DeepSeek-Coder Model (Running Locally via Ollama)
print("Initializing LLM...")
llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")  # Adjust the base_url if needed
print("LLM initialized.")

# ✅ Use STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION instead of OPENAI_FUNCTIONS
print("Initializing agent...")
agent = initialize_agent(
    tools=[add_numbers_tool],
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ✅ FIXED
    llm=llm,
    verbose=True
)
print("Agent initialized.")

# ✅ Test the agent
print("Running agent...")
response = agent.invoke({"input": "What is the sum of 2 and 3?"})
print("Agent Response:", response)
