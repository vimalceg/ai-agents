from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool

# Define the function that acts as a tool
def add_two_numbers(a: int, b: int):
    return f"The sum of {a} and {b} is {a + b}."

# Register the new tool
add_tool = Tool(
    name="AddTwoNumbers",  # Must match exactly with what the agent expects
    func=add_two_numbers,
    description="Use this tool to add two numbers and get the result."
)

# Initialize Ollama model
llm = Ollama(model="llama3.1",base_url="http://localhost:11434")  # Make sure you have the model downloaded

# Create an agent with the tool
agent = initialize_agent(
    tools=[add_tool],  # Register the tool
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Run the agent with the new tool
response = agent.invoke("Add 5 and 3")
print(response)
