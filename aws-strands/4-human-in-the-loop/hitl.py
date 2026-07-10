"""
A simple human-in-the-loop (HITL) example. 
"""

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import handoff_to_user

load_dotenv()

# Configure the model used by the agent.
model = LiteLLMModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY")
    },
    model_id="gemini/gemini-3.1-flash-lite",
    params={
        "max_tokens": 1024,
        "temperature": 0.85,
    },
)

# Make the agent ask the user before running a command.
system_prompt = "You're a friendly assistant. Ask for user approval before any command-line execution. If the user does not approve, do not execute the command. If the user approves, execute the command."

# Attach the handoff tool so the agent can pause and ask the user.
agent = Agent(
    model=model,
    tools=[handoff_to_user],
    system_prompt=system_prompt,
)

# Ask for the directory listing command.
query = "Use the command to list all the files & folders in the current directory."

response = agent(query)
