"""
Weather agent demo built with Strands.

Overview:
    This file shows the smallest complete pattern for creating and running a
    weather assistant agent.

What the program does:
    1. Loads environment variables so the API key can be read from a .env file.
    2. Creates a Strands agent that can call the National Weather Service API
       through the HTTP tool.
    3. Runs a sample user request in the main entry point and prints the result.

Beginner note:
    An "agent" is a program that combines a model, instructions, and tools so
    it can decide when to call external services and then explain the result in
    natural language.
"""

import os

from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from strands import Agent  # pyright: ignore[reportMissingImports]
from strands.models import LiteLLMModel  # pyright: ignore[reportMissingImports]
from strands_tools import http_request  # pyright: ignore[reportMissingImports]


# Load environment variables from a local .env file before any settings are read.
load_dotenv()


# This prompt tells the agent how to behave and how to use the weather API.
WEATHER_SYSTEM_PROMPT = """You are a friendly and helpful weather assistant with HTTP capabilities.

Your primary function is to provide accurate weather forecasts for locations in the United States by using the National Weather Service API.

Follow these steps to fulfill a user's request:
1. First, if you don't have grid coordinates, use the points API endpoint to get them.
   - For latitude and longitude: https://api.weather.gov/points/{latitude},{longitude}
   - For a US zipcode: https://api.weather.gov/points/{zipcode}
2. The points API will return a `forecast` URL. Use this URL to make a second HTTP request to get the actual weather forecast.
3. Process the forecast data and present it to the user in a clear, easy-to-understand format.

When displaying your response:
- Highlight key information like temperature, precipitation, and any weather alerts.
- Explain technical terms in simple language.
- If you encounter an error, apologize and explain that you couldn't retrieve the weather information.
"""


def create_weather_agent() -> Agent:
    """
    Create and configure the weather agent.

    The returned agent has everything it needs to answer weather questions:
    the model, the system instructions, and the HTTP tool used to call the
    National Weather Service API.

    Returns:
        Agent: A fully configured Strands agent ready to handle weather queries.
    """

    # The model talks to Gemini through LiteLLM and uses the API key from the environment.
    model = LiteLLMModel(
        client_args={
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
        model_id="gemini/gemini-3.1-flash-lite",
        params={
            "max_tokens": 1024,
            "temperature": 0.7,
        },
    )

    # The agent combines the model, instructions, and HTTP tool into one runnable unit.
    return  Agent(
        model=model,
        system_prompt=WEATHER_SYSTEM_PROMPT,
        tools=[http_request],
        name="WeatherAgent",
        description=(
            "A weather agent that provides forecasts for locations in the "
            "United States using the National Weather Service API."
        ),
    )


def main() -> None:
    """
    This function is the program entry point. It creates the weather agent,
    sends one example question, and prints the agent's response to the terminal.
    """

    # Build the agent once so the rest of the program can use it.
    weather_agent = create_weather_agent()

    # This sample prompt demonstrates how a user can ask for a weather comparison.
    response = weather_agent(
        "What's the weather like in New York City today? And compare it to the weather in Los Angeles."
    )

    # Print the final answer so the user can read it in the console.
    print(response)


if __name__ == "__main__":
    # Only run the demo when this file is executed directly.
    main()
