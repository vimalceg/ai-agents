from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# Define a schema with Optional fields to detect missing data
class Address(BaseModel):
    street: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City name")
    zip_code: Optional[str] = Field(None, description="ZIP code")

class Person(BaseModel):
    name: Optional[str] = Field(None, description="Full name")
    age: Optional[int] = Field(None, description="Age in years")
    email: Optional[str] = Field(None, description="Email address")
    addresses: Optional[List[Address]] = Field(None, description="List of addresses")

# Create an output parser
parser = PydanticOutputParser(pydantic_object=Person)

# Define a prompt that asks for missing information
prompt = PromptTemplate(
    template="Extract structured information from the following text:\n\n{text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = OllamaLLM(model_name="llama2", base_url="http://localhost:11434")

chain = prompt | llm | parser

# Function to handle missing fields
def request_missing_info(text):
    extracted_data = chain.invoke({"text": text})

    # Identify missing fields
    missing_fields = []
    if not extracted_data.name:
        missing_fields.append("name")
    if extracted_data.age is None:
        missing_fields.append("age")
    if not extracted_data.email:
        missing_fields.append("email")
    if not extracted_data.addresses:
        missing_fields.append("addresses")
    
    # If data is missing, ask the user
    if missing_fields:
        print(f"Missing fields detected: {', '.join(missing_fields)}")
        for field in missing_fields:
            value = input(f"Please provide the missing {field}: ")
            setattr(extracted_data, field, value if field != "age" else int(value))

    return extracted_data

# Sample input with missing fields
input_text = """
John Doe lives in New York. His email is johndoe@example.com.
"""

# Run extraction with missing data handling
final_data = request_missing_info(input_text)

# Output the completed structured data
print(final_data)
