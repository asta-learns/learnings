"""
Use two MCP clients in one agent.

This file shows a simple flow:
    1. Create a model.
    2. Connect to two MCP servers.
    3. Load tools from both servers.
    4. Give all tools to one agent.
    5. Ask the agent a question.
"""

from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient
import os

import warnings

# Hide extra warnings so the example stays easy to read.
warnings.filterwarnings("ignore")

load_dotenv()

# Create the model used by the agent.
model = LiteLLMModel(
    client_args={"api_key": os.getenv("GEMINI_API_KEY")},
    model_id="gemini/gemini-3.1-flash-lite",
    params={"max_tokens": 1024, "temperature": 0.85},
)

# This client would connect to a local file system MCP server.
# I kept this here as an example, as I could not figure out how to work with this. 
file_system_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=[
                "@modelcontextprotocol/server-filesystem",
                os.getcwd(),
            ],
        )
    )
)

# This client gives the agent web search tools.
duckduckgo_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=[
                "duckduckgo-mcp-server@latest",
            ],
        )
    )
)

# This client gives the agent page fetching tools.
fetch_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=[
                "mcp-fetch-server",
            ],
        )
    )
)

with duckduckgo_mcp_client, fetch_mcp_client:
    # Load tools from both servers.
    duckduckgo_tools = duckduckgo_mcp_client.list_tools_sync()
    fetch_tools = fetch_mcp_client.list_tools_sync()

    # Merge all tools into one list.
    all_tools = duckduckgo_tools + fetch_tools

    print("=== DuckDuckGo MCP Client Tools ===")
    print(f"Loaded {len(duckduckgo_tools)} tools from the DuckDuckGo MCP client.")

    # Print every tool so you can see what the agent gets.
    for tool in all_tools:
        print(f"- {tool}")

    # Create one agent that can use tools from both MCP servers.
    agent = Agent(
        model=model,
        tools=all_tools,
    )

    # Ask the agent a simple search question.
    ddg_query = "What is the capital of France?"
    response_1 = agent(ddg_query)
    print(f"=== Response to DuckDuckGo Query: '{ddg_query}' ===")
    print(response_1)

    # Ask the agent to fetch content from a URL.
    fetch_query = "Fetch the latest news about AI from https://news.google.com/topics/CAAqJAgKIh5DQkFTRUFvSEwyMHZNRzFyZWhJRlpXNHRSMElvQUFQAQ?ceid=IN:en&oc=3"
    response_2 = agent(fetch_query)
    print(f"=== Response to Fetch Query: '{fetch_query}' ===")
    print(response_2)
