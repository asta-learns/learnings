"""
Access tools from an MCP server in an agent.
"""

import os

from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient


load_dotenv()

# Use the Gemini key from your .env file.
model = LiteLLMModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY")
    },
    model_id="gemini/gemini-3.1-flash-lite",
    params={
        "max_tokens": 1024,
        "temperature": 0.85,
    }
)

# Change this to True if you want to pass the MCP client directly to the agent.
USE_DIRECT_MCP_CLIENT = False

# Create the MCP client for the AWS documentation server.
mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
        )
    )
)


def create_agent_with_direct_client() -> Agent:
    # This keeps the MCP client inside the agent.
    return Agent(
        model=model,
        tools=[mcp_client],
    )


def create_agent_with_loaded_tools() -> Agent:
    # This opens the client, loads the tools, and then passes the tools to the agent.
    with mcp_client:
        aws_tools = mcp_client.list_tools_sync()
        return Agent(
            model=model,
            tools=aws_tools,
        )


if USE_DIRECT_MCP_CLIENT:
    agent = create_agent_with_direct_client()
else:
    agent = create_agent_with_loaded_tools()


query = "List the basic available AWS services and provide a brief description of each for someone new to AWS."
response = agent(query)
print(response)
