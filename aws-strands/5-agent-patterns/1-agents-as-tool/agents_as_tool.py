"""
Multi-agent Pattern: Agents as Tool

We define agent as a tool.

I can define my own tools for each specialized agent, and then use them in the agent's tool list.
But for now I chose not to.
"""

import os
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.litellm import LiteLLMModel
from strands_tools import http_request


load_dotenv()

model = LiteLLMModel(
        client_args={
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
        model_id="gemini/gemini-3.1-flash-lite",
        params={
            "max_tokens": 1024,
            "temperature": 0.8,
        },
)

@tool
def game_recommendation_assistant(query: str) -> str:
    """
    A tool that provides game recommendations based on a query.

    Args:
        query (str): The input query for game recommendations.

    Returns:
        str: A string containing the recommended games.
    """

    agent = Agent(model=model,
                  tools=[http_request],
                  system_prompt="You are a game recommendation agent. Provide relevant game recommendations based on the user's query."
            )
    
    response = agent(query)
    return response


@tool
def anime_recommendation_assistant(query: str) -> str:
    """
    A tool that provides anime recommendations based on a query.

    Args:
        query (str): The input query for anime recommendations.

    Returns:
        str: A string containing the recommended anime.
    """

    agent = Agent(model=model,
                  tools=[http_request],
                  system_prompt="You are an anime recommendation agent. Provide relevant anime recommendations based on the user's query."
            )
    
    response = agent(query)
    return response


@tool
def book_recommendation_assistant(query: str) -> str:
    """
    A tool that provides book recommendations based on a query.

    Args:
        query (str): The input query for book recommendations.

    Returns:
        str: A string containing the recommended books.
    """

    agent = Agent(model=model,
                  tools=[http_request],
                  system_prompt="You are a book recommendation agent. Provide relevant book recommendations based on the user's query."
            )
    
    response = agent(query)
    return response

