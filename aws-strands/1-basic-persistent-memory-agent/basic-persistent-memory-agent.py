"""
A simple persistent memory agent.

Module Overview:
    This module implements a stateful conversational agent that retains memory
    across multiple runs using file-based session management. The agent leverages
    the Strands framework to combine a language model with session persistence,
    enabling context-aware interactions.

Key Components:
    - create_persistent_session(): Initializes file-based session storage.
    - create_agent(): Constructs the agent with model and session manager.
    - main(): Entry point that demonstrates agent capabilities.

Beginner Note:
    A "persistent agent" combines three concepts:
    1. A language model (Gemini) for understanding and responding.
    2. A session manager that saves conversations to disk.
    3. The Strands framework that ties everything together.
"""

import os
from pathlib import Path

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from strands import Agent  # pyright: ignore[reportMissingImports]
from strands.models import LiteLLMModel  # pyright: ignore[reportMissingImports]
from strands_tools import http_request  # pyright: ignore[reportMissingImports]
from strands.session.file_session_manager import FileSessionManager  # pyright: ignore[reportMissingImports]


# Load environment variables from a local .env file before any settings are read.
load_dotenv()


def create_persistent_session(session_id: str) -> FileSessionManager:
    """
    Create and configure a file-based session manager.

    Initializes a FileSessionManager that persists agent state and conversation
    history to disk. All session data is stored in a 'session_data' subdirectory
    relative to this script's location.

    Returns:
        FileSessionManager: Configured session manager with persistent storage
            initialized for the specified session ID.

    Note:
        Each session is uniquely identified by its session_id. Using the same
        session_id across runs allows the agent to recall previous conversations.
    """
    # Determine the directory where this script is located.
    base_dir = Path(__file__).parent.resolve()
    # Define the subdirectory where all session data will be stored on disk.
    storage_dir = str(base_dir / "session_data")

    # Initialize the session manager with a persistent storage backend.
    # The session_id acts as a unique identifier for this conversation thread.
    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=storage_dir,
    )

    return session_manager


def create_agent(session_id: str) -> Agent:
    """
    Create and configure a stateful Strands agent.

    Assembles a complete agent by combining three key components:
    1. A language model (Gemini Flash Lite) via LiteLLM for inference.
    2. A file-based session manager for state persistence.
    3. Strands framework orchestration.

    The returned agent can maintain context across multiple conversation turns
    because its session state is written to disk after each interaction.

    Returns:
        Agent: A fully initialized Strands agent with persistent memory and
            ready-to-use inference capabilities.

    Environment Requirements:
        - GEMINI_API_KEY: Must be set in the .env file for API authentication.
    """

    # Initialize the language model with Gemini as the backbone inference engine.
    # LiteLLM acts as a unified interface to multiple LLM providers.
    model = LiteLLMModel(
        client_args={
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
        # Using Gemini 3.1 Flash Lite for a balance of speed and capability.
        model_id="gemini/gemini-3.1-flash-lite",
        params={
            # Limit output length to keep responses concise.
            "max_tokens": 1024,
            # Set creativity level (0.0=deterministic, 1.0=highly creative).
            "temperature": 0.7,
        },
    )

    # Create the session manager responsible for persisting conversation history.
    session_manager = create_persistent_session(session_id)

    # Assemble the final agent by combining model, configuration, and session.
    # Strands handles routing, tool calls, and multi-turn orchestration.
    return Agent(
        model=model,
        name="KnowAsta",
        description=(
            "A stateful agent with persistent memory that recalls and responds to "
            "context from previous conversations."
        ),
        session_manager=session_manager,
    )


def main() -> None:
    """
    This function serves as the primary entry point when the script is run
    directly. It instantiates the persistent memory agent and demonstrates
    its ability to recall prior conversational context.

    Workflow:
        1. Initializes the agent with persistent session storage.
        2. Sends two queries to the agent (first stores context, second queries it).
        3. Outputs the agent's response to the console.

    Output:
        Prints the agent's response string directly to stdout.
    """
    # Instantiate the agent with all dependencies and session persistence enabled.
    session_id = "know_asta"
    persistent_agent = create_agent(session_id=session_id)

    print("=== Persistent Memory Agent Demo ===")
    print("=== Conversation Start ===")

    print("\nQuery 1: My name is Asta.")
    response_1 = persistent_agent("My name is Asta.")

    print("\nQuery 2: What is my name?")
    response_2 = persistent_agent("What is my name?")

    print("\n=== Conversation End ===")
    print(f"The agent was able to remember the name because its memory is persisted in the session '{session_id}'.")


if __name__ == "__main__":
    # Guard clause: only execute main() when this module is run directly,
    # not when it is imported as a module in another script.
    main()
