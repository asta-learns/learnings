"""
Implementation of workflow multiagent pattern.

Use Case: Cross-Media Recommendation System
    - Recommend a anime, and game that matches the user's query.
    - And a book that blends the anime and game together.
"""

import os
import logging
from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel
from strands_tools import workflow


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),]
)

logging.info("Initializing model for the agent...")
model = GeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY"),
    },
    model_id="gemini-3.1-flash-lite",
    params={
        "max_output_tokens": 1024,
        "temperature": 0.7,
    }
)
logging.info("Model initialized successfully.")

logging.info("Creating Orchestrator agent for cross-media recommendation system...")
orchestrator = Agent(
    model=model,
    system_prompt="""
    You are an orchestrator. You will be given a user's query, and you will ask the anime expert and game expert for recommendations.
    Then, you will ask the book expert to recommend a book that blends the anime and game together.
    """,
    tools=[workflow],
)
logging.info("Orchestrator agent created successfully.")

query = "Recommend an anime and a game that matches the user's query, and a book that blends the anime and game together, in topic of financial literacy and investment strategies."

# Creating workflow for cross-media recommendation
logging.info("Creating workflow for cross-media recommendation...")
results = orchestrator.tool.workflow(
    action="create",
    workflow_id="cross_media_recommendation",
    tasks=[
        {
            "task_id": "get_anime_recommendation",
            "description": f"Get an anime recommendation based on the user's query: {query}",
            "system_prompt": "You are an anime expert. You will be given a user's query, and you will recommend an anime that matches the query.",
            "tools": [],
            "priority": 5,
        },
        {
            "task_id": "get_game_recommendation",
            "description": f"Get a game recommendation based on the user's query: {query}",
            "system_prompt": "You are a game expert. You will be given a user's query, and you will recommend a game that matches the query.",
            "tools": [],
            "priority": 5,
        },
        {
            "task_id": "get_book_recommendation",
            "description": f"Get a book recommendation that blends the anime and game together",
            "system_prompt": "You are a book expert. You will be given an anime and a game, and you will recommend a book that blends the two together perfectly.",
            "dependencies": ["get_anime_recommendation", "get_game_recommendation"],
            "tools": [],
            "priority": 3,
        },
    ]
)

logging.info("Starting cross-media recommendation workflow...")
result = orchestrator.tool.workflow(action="start", workflow_id="cross_media_recommendation")
logging.info("Workflow done successfully.")
