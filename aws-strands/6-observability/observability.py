"""
Observability using Langfuse, and OpenTelemetry.
"""

import os
from langfuse import get_client
from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel

load_dotenv()

# Get keys for your project from the project settings page: https://cloud.langfuse.com
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
base_url = os.getenv("LANGFUSE_BASE_URL")

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

model = GeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    model_id="gemini-3.1-flash-lite",
    params={
        "max_output_tokens": 1024,
        "temperature": 0.7,
    },
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful game assistant. You will answer questions about the game and provide helpful information to the user."
)

response = agent("Which is the best horror game on steam for this month? How many players are playing it right now?")
