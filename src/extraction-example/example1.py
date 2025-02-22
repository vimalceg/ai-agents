from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import json

class Person(BaseModel):
    """Information about Person."""
    name: Optional[str] = Field(default=None, description="The name of the person.")
    hair_color: Optional[str] = Field(default=None, description="The hair color of the person.")
    height_in_meters: Optional[str] = Field(default=None, description="The height of the person in meters.")

# Adjusted prompt to enforce JSON output
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Extract the relevant information from the text and return it as a valid JSON object. "
            "If an attribute is missing, return null. "
            "The output must be strictly JSON:\n"
            '{{ "name": null | string, "hair_color": null | string, "height_in_meters": null | string }}',
        ),
        ("human", "{text}"),
    ]
)

llm = OllamaLLM(model="llama2", base_url="http://localhost:11434")

text = "The person's name is John. He has brown hair and is 1.8 meters tall."
prompt = prompt_template.invoke({"text": text})

# Generate response
response = llm.invoke(prompt)

# Extract the actual content (LangChain models return objects)
response_text = response.content if hasattr(response, "content") else str(response)

# Parse the response into the Pydantic model
try:
    data = json.loads(response_text.strip())  # Ensure clean JSON
    person = Person(**data)
    print(person)
except json.JSONDecodeError:
    print("Failed to parse JSON response:", response_text)
