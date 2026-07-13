"""
Implementing graph-based workflow multi-agent pattern.

Use Case: Anime Council!!

    A multi-agent system where anime characters independently answer the user question from their personality,
    and a Judge agent evaluates the responses to select the most compelling or best-supported answer.

    We have 5 characters:
                            Erwin Smith (Attack on Titan)
                            L Lawliet (Death Note)
                            Asta (Black Clover)
                            Roymen Sukuna (Jujutsu Kaisen)
                            Gintoki Sakata (Gintama)
"""

import os
import logging
from dotenv import load_dotenv
from strands import Agent
from strands.multiagent import GraphBuilder
from strands.models.gemini import GeminiModel

from strands.multiagent.base import MultiAgentBase, NodeResult, Status, MultiAgentResult
from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message


class QueryPasser(MultiAgentBase):
    """
    A deterministic custom node that passes the query to all agents.
    Acts like the starting point of the graph.
    """
    def __init__(self, name, query):
        super().__init__()
        self.name = name
        self.query = query

    async def invoke_async(self, task, invocation_state, **kwargs):
        agent_result = AgentResult(
            state=None,
            metrics=None,
            stop_reason="end_turn",
            message=Message(
                role="query_passer",
                content=[ContentBlock(text=self.query)],
            ),
        )

        return MultiAgentResult(
            status=Status.COMPLETED,
            results={self.name: NodeResult(result=agent_result)},
        )


load_dotenv()

logging.getLogger("strands.multiagent").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

model = GeminiModel(
            client_args={
                "api_key": os.getenv("GEMINI_API_KEY"),
            },
            model_id="gemini-3.1-flash-lite",
            params={
                "max_output_tokens": 1024,
                "temperature": 0.8,
            },
)

query = "Should you sacrifice a few to save many?" 

query_passer = QueryPasser(name="Query Passer", query=query)

agent_erwin = Agent(
                name="Erwin Smith",
                model=model,
                system_prompt="""
                            You are Erwin Smith, the former Commander of the Survey Corps in Attack on Titan.
                            You are known for your strategic mind, leadership skills, and unwavering dedication to humanity's survival.
                            You are analytical, decisive, and often prioritize the greater good over personal feelings.
                            Your responses should reflect your tactical thinking, sense of duty, and ability to inspire others.""",
)

agent_lawliet = Agent(
                name="L Lawliet",
                model=model,
                system_prompt="""
                            You are L Lawliet, the world-renowned detective from Death Note.
                            You are highly intelligent, analytical, and possess exceptional deductive reasoning skills.
                            You are known for your eccentric behavior, meticulous attention to detail, and ability to solve complex cases.
                            Your responses should reflect your logical thinking, keen observation, and unique problem-solving approach.""",
)

agent_asta = Agent(
                name="Asta",
                model=model,
                system_prompt="""
                            You are Asta, the determined and energetic protagonist of Black Clover.
                            You are known for your unwavering determination, boundless energy, and relentless pursuit of your goals.
                            You are optimistic, courageous, and never give up, even in the face of adversity.
                            Your responses should reflect your enthusiasm, determination, and positive outlook, as well as your willingness to take on challenges head-on.""",
)

agent_sukuna = Agent(
                name="Ryomen Sukuna",
                model=model,
                system_prompt="""
                            You are Ryomen Sukuna, the powerful and malevolent curse from Jujutsu Kaisen.
                            You are known for your immense strength, cunning nature, and ruthless demeanor.
                            You are feared by many and have a dark sense of humor. Your responses should reflect your confidence, dominance, and strategic thinking.
                            You are a master manipulator and enjoy toying with your opponents, often using psychological tactics to gain the upper hand.
                            Your responses should reflect your cunning, confidence, and strategic thinking, as well as your dark sense of humor.""",
)

agent_gintoki = Agent(
                name="Gintoki Sakata",
                model=model,
                system_prompt="""
                            You are Gintoki Sakata, the lazy yet skilled samurai from Gintama.
                            You are known for your laid-back attitude, love for sweets, and exceptional swordsmanship.
                            You have a strong sense of justice and are willing to fight for what you believe in, but you often do so in a humorous and unconventional manner.
                            Your responses should reflect your wit, sarcasm, and ability to find humor in any situation, as well as your underlying sense of honor and loyalty.""",
)

agent_judge = Agent(
                name="Judge",
                model=model,
                system_prompt="""
                            You are the Judge, an impartial evaluator of responses from various anime characters.
                            Your role is to assess the quality, persuasiveness, and relevance of each character's response to a given question.
                            You are fair, analytical, and objective in your evaluations, and you prioritize clarity, coherence, and logical reasoning in your judgments.
                            Your responses should reflect your ability to critically analyze arguments, provide constructive feedback, and make informed decisions based on the merits of each response.""",
)

graph_builder = GraphBuilder()

graph_builder.add_node(query_passer, "query_passer_node")
graph_builder.add_node(agent_erwin, "agent_erwin_node")
graph_builder.add_node(agent_lawliet, "agent_lawliet_node")
graph_builder.add_node(agent_asta, "agent_asta_node")
graph_builder.add_node(agent_sukuna, "agent_sukuna_node")
graph_builder.add_node(agent_gintoki, "agent_gintoki_node")
graph_builder.add_node(agent_judge, "agent_judge_node")

graph_builder.add_edge("query_passer_node", "agent_erwin_node")
graph_builder.add_edge("query_passer_node", "agent_lawliet_node")
graph_builder.add_edge("query_passer_node", "agent_asta_node")
graph_builder.add_edge("query_passer_node", "agent_sukuna_node")
graph_builder.add_edge("query_passer_node", "agent_gintoki_node")

graph_builder.add_edge("agent_erwin_node", "agent_judge_node")
graph_builder.add_edge("agent_lawliet_node", "agent_judge_node")
graph_builder.add_edge("agent_asta_node", "agent_judge_node")
graph_builder.add_edge("agent_sukuna_node", "agent_judge_node")
graph_builder.add_edge("agent_gintoki_node", "agent_judge_node")

graph_builder.set_entry_point("query_passer_node")
graph_builder.set_execution_timeout(300)

graph = graph_builder.build()

result = graph(query)

# Access the results
print(f"\nStatus: {result.status}")
print(f"Execution order: {[node.node_id for node in result.execution_order]}")
