"""
Chapter 2: Hello Agent - A simple LangGraph agent with custom tools.

This module defines a conversational agent with two tools (get_current_time and calculate).
The agent uses Databricks-hosted Claude as its LLM and is served via MLflow's AgentServer.

Key concepts:
- @tool decorator turns Python functions into tools the LLM can call
- @invoke() and @stream() register request handlers with the AgentServer
- create_agent() builds a LangGraph ReAct agent from tools + model
"""

from datetime import datetime
from typing import AsyncGenerator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import process_agent_astream_events

# Enable automatic MLflow tracing for all LangChain/LangGraph operations.
# Every LLM call and tool invocation is logged to your MLflow experiment.
mlflow.langchain.autolog()


# ---------------------------------------------------------------------------
# Tools
#
# Each @tool-decorated function becomes a tool the LLM can invoke.
# The docstring is critical — it's what the LLM reads to decide when and
# how to call the tool. Include clear descriptions and argument examples.
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks what time it is or what today's date is."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool when the user asks you to do math, compute something,
    or evaluate a numerical expression.

    Args:
        expression: A mathematical expression to evaluate. Examples: '2 + 2', '15 * 3.14', '(100 - 32) * 5/9'
    """
    try:
        # Restrict eval to safe math builtins only — no access to __import__, open, etc.
        allowed_names: dict[str, object] = {
            "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
        }
        result: object = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Agent Initialization
# ---------------------------------------------------------------------------

# Collect all tools into a list that will be passed to the agent.
# To add a new tool, define it with @tool above and append it here.
TOOLS: list = [get_current_time, calculate]

# Configure the LLM — this uses Databricks Foundation Model APIs.
# The ChatDatabricks class handles authentication automatically:
# - Locally: uses your Databricks CLI profile or env vars
# - Deployed: uses the app's service principal credentials
MODEL: ChatDatabricks = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")


def init_agent() -> CompiledStateGraph:
    """Create a LangGraph ReAct agent with our tools and model.

    Returns a compiled LangGraph state graph that can process messages
    and decide which tools to call based on the user's input.
    """
    return create_agent(tools=TOOLS, model=MODEL)


# ---------------------------------------------------------------------------
# MLflow AgentServer Handlers
#
# These two functions are registered with the AgentServer via decorators:
# - @invoke() handles non-streaming requests (returns complete response)
# - @stream() handles streaming requests (yields events as they happen)
#
# The request/response format follows the OpenAI Responses API spec.
# See: https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/
# ---------------------------------------------------------------------------


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests by collecting all stream events into a single response.

    This delegates to the streaming handler and collects only the final
    output items (type="response.output_item.done"), discarding intermediate
    text deltas. The result is a complete ResponsesAgentResponse.
    """
    outputs: list = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming requests by yielding events as the agent produces them.

    This is the core execution path:
    1. Initialize the LangGraph agent with tools and model
    2. Convert the incoming Responses API messages to LangChain format
    3. Stream the agent's execution, yielding events for each text chunk
       and tool call result
    """
    # Create a fresh agent instance for each request
    agent: CompiledStateGraph = init_agent()

    # Convert Responses API input format to LangChain message format.
    # request.input is a list of ResponsesAgentInput items (role + content).
    # to_chat_completions_input() converts them to ChatCompletions-style dicts.
    messages: dict[str, list] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input])
    }

    # Stream the agent's execution. stream_mode=["updates", "messages"] gives us:
    # - "updates": complete tool results and final messages (for output items)
    # - "messages": token-by-token text chunks (for streaming text deltas)
    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
