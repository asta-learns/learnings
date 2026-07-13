"""
Use the orchestrator agent as pass in the specialized agents as tools.
The orchestrator agent will then call the specialized agents as needed to provide recommendations based on the user's query.
"""

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel

from agents_as_tool import game_recommendation_assistant, anime_recommendation_assistant, book_recommendation_assistant


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

agent = Agent(model=model,
              tools=[game_recommendation_assistant, anime_recommendation_assistant, book_recommendation_assistant],
              system_prompt="You are an orchestrator agent. You will call the specialized agents as tools to provide recommendations based on the user's query."
)

game_query = "Can you recommend some good games related to VALORANT?"
anime_query = "Can you recommend some good anime related to Naruto?"
book_query = "Can you recommend some good books similar to The Laws of Human Nature?"

print("--- Game Recommendations ---")
print(agent(game_query))

print("\n--- Anime Recommendations ---")
print(agent(anime_query))

print("\n--- Book Recommendations ---")
print(agent(book_query))
