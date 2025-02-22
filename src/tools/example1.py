from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool
from datetime import datetime

# Define the function that acts as a tool
def get_current_time():
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Register it as a LangChain Tool
time_tool = Tool(
    name="GetCurrentTime",  # Must match exactly with what the agent expects
    func=get_current_time,
    description="Use this tool to get the current date and time in YYYY-MM-DD HH:MM:SS format."
)

# Initialize Ollama model
llm = Ollama(model="llama3.1",base_url="http://localhost:11434")  # Make sure you have the model downloaded

# Create an agent with the tool
agent = initialize_agent(
    tools=[time_tool],  # Register the tool
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Run the agent
response = agent.invoke("What is the current time?")
print(response)
