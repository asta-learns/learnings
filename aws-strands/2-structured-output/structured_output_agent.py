"""Structured Output Example.

This example shows how to use a Strands agent with a Pydantic schema to return
validated structured data instead of plain text.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.litellm import LiteLLMModel


# Load environment variables 
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class PersonInfo(BaseModel):
    """Schema for the person information extracted from text."""

    name: str = Field(..., description="Full name of the person.")
    age: int = Field(..., description="Age in years.")
    email: str = Field(..., description="Email address.")
    occupation: str = Field(..., description="Job or profession.")


# Configure the model used by the agent.
model = LiteLLMModel(
    client_args={
        "api_key": GEMINI_API_KEY,
    },
    model_id="gemini/gemini-3.1-flash-lite",
    params={
        "max_tokens": 1024,
        "temperature": 0.7,
    },
)


# Create the agent.
agent = Agent(model=model)


# Request structured output using the schema defined above.
response = agent(
    "John is 28 years old and his email is hello@john.com, and he lives in Chennai, he works as a software engineer.",
    structured_output_model=PersonInfo,
)

try:
    person_info: PersonInfo = response.structured_output
    print("\n--- Structured Output ---")
    print(f"Name: {person_info.name}")
    print(f"Age: {person_info.age}")
    print(f"Email: {person_info.email}")
    print(f"Occupation: {person_info.occupation}")
except Exception as e:
    print(f"Error validating structured output: {e}")

