"""
Swarm intelligence: agents collaborate to build an anime concept.
Each agent has one clear responsibility (director, story, characters, etc.).
"""

import os
from dotenv import load_dotenv
from strands import Agent
from strands.multiagent import Swarm
from strands.models.gemini import GeminiModel
import logging


# Basic logging to stdout for simple tracing.
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()
logging.info("Loaded environment variables.")

# Create the language model using the GEMINI_API_KEY from the environment.
logging.info("Initializing Gemini model.")
model = GeminiModel(
    client_args={
        "api_key": os.getenv("GEMINI_API_KEY")
    },
    model_id="gemini-3.1-flash-lite",
    params={
        "max_output_tokens": 1024,
        "temperature": 0.85
    },
)
logging.info("Initialized Gemini model.")

logging.info("Creating agents for the anime creation swarm.")

# Director: coordinates the process and delegates to other agents.
logging.info("Creating Director Agent.")
director_agent = Agent(
    name="Director Agent",
    model=model,
    system_prompt = """
    You are the Director Agent for an anime creation swarm.

    Your responsibilities:
    - Oversee the entire anime creation process.
    - Maintain the overall vision and consistency.
    - Make high-level decisions about the story, characters, world, and themes.
    - Decide which specialist agent should work next.
    - Ensure all agents work together toward a cohesive final anime.
    - Do not create every detail yourself; delegate work to the appropriate agent whenever possible.
    """,
)
logging.info("Created Director Agent.")

# Story: writes the plot and structure.
logging.info("Creating Story Agent.")
story_agent = Agent(
    name="Story Agent",
    model=model,
    system_prompt="""
    You are the Story Agent.

    Your responsibilities:
    - Create the main plot and central conflict.
    - Develop the story structure and pacing.
    - Ensure the story is engaging and coherent.
    - Revise the story when other agents provide feedback.
    - Focus only on the story, not characters or powers.
    """,
)
logging.info("Created Story Agent.")

# Character: designs characters and motivations.
logging.info("Creating Character Agent.")
character_agent = Agent(
    name="Character Agent",
    model=model,
    system_prompt="""
    You are the Character Agent.

    Your responsibilities:
    - Design the protagonist, antagonist, and supporting characters.
    - Give each character clear motivations and personalities.
    - Ensure characters fit naturally into the story.
    - Revise characters when feedback is received.
    - Focus only on characters, not the overall plot.
    """,
)
logging.info("Created Character Agent.")

# Power system: defines mechanics and balance for abilities.
logging.info("Creating Power System Agent.")
power_system_agent = Agent(
    name="Power System Agent",
    model=model,
    system_prompt="""
    You are the Power System Agent.

    Your responsibilities:
    - Design the anime's power system.
    - Define clear rules, limitations, and costs.
    - Keep abilities balanced and internally consistent.
    - Ensure powers complement the story and world.
    - Revise the power system when feedback is received.
    """,
)
logging.info("Created Power System Agent.")

# Worldbuilding: builds setting, history, and cultures.
logging.info("Creating Worldbuilding Agent.")
worldbuilding_agent = Agent(
    name="Worldbuilding Agent",
    model=model,
    system_prompt="""
    You are the Worldbuilding Agent.

    Your responsibilities:
    - Design the world's geography, history, cultures, and factions.
    - Ensure the setting supports the story.
    - Keep the world internally consistent.
    - Expand the lore where necessary.
    - Revise the world based on feedback from other agents.
    """,
)
logging.info("Created Worldbuilding Agent.")

# Critic: checks for problems and suggests fixes.
logging.info("Creating Critic Agent.")
critic_agent = Agent(
    name="Critic Agent",
    model=model,
    system_prompt="""
    You are the Critic Agent.

    Your responsibilities:
    - Review the work produced by other agents.
    - Identify plot holes, inconsistencies, weak characters, and balance issues.
    - Provide constructive feedback.
    - Suggest which agent should revise their work.
    - Do not create new content unless necessary for clarification.
    """,
)
logging.info("Created Critic Agent.")

# Editor: merges contributions into the final product.
logging.info("Creating Editor Agent.")
editor_agent = Agent(
    name="Editor Agent",
    model=model,
    system_prompt="""
    You are the Final Editor Agent.

    Your responsibilities:
    - Collect the work from all agents.
    - Merge everything into a polished anime concept.
    - Ensure the final output is clear, consistent, and well-structured.
    - Preserve the intent of every contributing agent.
    """,
)
logging.info("Created Editor Agent.")

# Assemble swarm: list nodes and set execution limits.
logging.info("Creating the Swarm with all agents.")
swarm = Swarm(
    nodes=[
        director_agent,
        story_agent,
        character_agent,
        power_system_agent,
        worldbuilding_agent,
        critic_agent,
        editor_agent,
    ],
    entry_point=director_agent,
    max_handoffs=15,
    max_iterations=20,
    execution_timeout=300,
    node_timeout=30,
    repetitive_handoff_detection_window=8,
    repetitive_handoff_min_unique_agents=3,
)
logging.info("Created the Swarm with all agents.")

query = """
    Create a new anime concept.
    Include story, characters, power system, and world.
    Make elements cohesive and engaging.
    References: Naruto, One Piece, Attack on Titan, My Hero Academia, Demon Slayer.
"""

logging.info("Starting the swarm intelligence process with the query.")
result = swarm(query)
logging.info("Swarm intelligence process completed.")

logging.info("Final result from the swarm:")
print(f"Status: {result.status}")
print(f"Node history: {[node.node_id for node in result.node_history]}")
logging.info("Swarm intelligence process completed and logged.")
