from datetime import datetime
from typing import AsyncGenerator

import mlflow
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import process_agent_astream_events

mlflow.langchain.autolog()

# ---------------------------------------------------------------------------
# Tools - Simple Python functions decorated with @tool
# The docstring is what the LLM reads to decide when to use each tool.
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
        # Only allow safe math operations
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

# Collect all tools into a list
TOOLS = [get_current_time, calculate]

# Configure the LLM - this uses Databricks Foundation Model APIs
MODEL = ChatDatabricks(endpoint="databricks-claude-sonnet-4")


def init_agent():
    """Create a LangGraph agent with our tools and model."""
    return create_agent(tools=TOOLS, model=MODEL)


# ---------------------------------------------------------------------------
# MLflow AgentServer handlers
# ---------------------------------------------------------------------------


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests. Collects all stream events and returns the final response."""
    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming requests. Yields events as they are produced by the agent."""
    agent = init_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
