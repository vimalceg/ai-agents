from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool
from datetime import datetime

# Define the function that acts as a tool
# Accept a single argument and return a greeting with the current time

def greet_user(name):
    return f"Hello, {name}!,"

# Define the function that acts as a tool to get the current time

def get_current_time():
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ."

# Register the tools as LangChain Tools
greet_tool = Tool(
    name="GreetUser",  # Must match exactly with what the agent expects
    func=greet_user,
    description="Use this tool to greet the user and get the current date and time in YYYY-MM-DD HH:MM:SS format."
)

time_tool = Tool(
    name="GetCurrentTime",  # Must match exactly with what the agent expects
    func=get_current_time,
    description="Use this tool to get the current date and time in YYYY-MM-DD HH:MM:SS format."
)

# Initialize Ollama model
llm = Ollama(model="mistral", base_url="http://localhost:11434")  # Make sure you have the model downloaded

# Create an agent with the tools
agent = initialize_agent(
    tools=[greet_tool, time_tool],  # Register the tools
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Run the agent
response = agent.invoke("Hi I am Vimal. I want to know the current time.")
print(response)
response = agent.invoke("greet user vimal")
print(response)
