import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Optional, Sequence, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    AsyncCheckpointSaver,
    AsyncDatabricksStore,
    ChatDatabricks,
)
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.utils import (
    get_databricks_host_from_env,
    get_or_create_thread_id,
    get_lakebase_access_error_message,
    process_agent_astream_events,
)
from agent_server.utils_memory import get_user_id, memory_tools

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
sp_workspace_client = WorkspaceClient()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
LAKEBASE_INSTANCE_NAME = os.getenv("LAKEBASE_INSTANCE_NAME") or None
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
EMBEDDING_DIMS = 1024

if not LAKEBASE_INSTANCE_NAME:
    raise ValueError(
        "LAKEBASE_INSTANCE_NAME is required. Set it in your .env.local file.\n"
        "Create a Lakebase instance: databricks lakebase create-database-instance <name> --capacity CU_1"
    )

SYSTEM_PROMPT = """You are a helpful assistant. Use the available tools to answer questions.

You have access to memory tools that allow you to remember information about users:
- Use get_user_memory to search for previously saved information about the user
- Use save_user_memory to remember important facts, preferences, or details the user shares
- Use delete_user_memory to forget specific information when asked

Always check for relevant memories at the start of a conversation to provide personalized responses.

**Always save** when the user explicitly asks you to remember something.
**Proactively save** preferences, role/expertise, ongoing projects, recurring constraints.
**Don't save** temporary facts, trivial details, or highly sensitive personal information."""


# ---------------------------------------------------------------------------
# Custom Tools
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate. Examples: '2 + 2', '15 * 3.14'
    """
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Stateful Agent State
# The add_messages annotation tells LangGraph to APPEND new messages to
# the existing list (short-term memory) rather than replacing it.
# ---------------------------------------------------------------------------


class StatefulAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


# ---------------------------------------------------------------------------
# Agent Initialization
# ---------------------------------------------------------------------------


async def init_agent(
    checkpointer: Optional[Any] = None,
    store: Optional[BaseStore] = None,
):
    """Create a LangGraph agent with short-term and long-term memory.

    Args:
        checkpointer: AsyncCheckpointSaver for short-term memory (conversation history)
        store: AsyncDatabricksStore for long-term memory (user facts/preferences)
    """
    tools = [get_current_time, calculate] + memory_tools()
    model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        state_schema=StatefulAgentState,
    )


# ---------------------------------------------------------------------------
# MLflow AgentServer handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    thread_id = get_or_create_thread_id(request)
    request.custom_inputs = dict(request.custom_inputs or {})
    request.custom_inputs["thread_id"] = thread_id

    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    user_id = get_user_id(request)
    custom_outputs = {"thread_id": thread_id}
    if user_id:
        custom_outputs["user_id"] = user_id

    return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = get_or_create_thread_id(request)
    user_id = get_user_id(request)

    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if user_id:
        config["configurable"]["user_id"] = user_id

    try:
        # Open both short-term (checkpointer) and long-term (store) connections
        async with AsyncCheckpointSaver(
            instance_name=LAKEBASE_INSTANCE_NAME,
        ) as checkpointer:
            await checkpointer.setup()

            async with AsyncDatabricksStore(
                instance_name=LAKEBASE_INSTANCE_NAME,
                embedding_endpoint=EMBEDDING_ENDPOINT,
                embedding_dims=EMBEDDING_DIMS,
            ) as store:
                await store.setup()
                config["configurable"]["store"] = store

                agent = await init_agent(checkpointer=checkpointer, store=store)

                async for event in process_agent_astream_events(
                    agent.astream(input_state, config, stream_mode=["updates", "messages"])
                ):
                    yield event

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ["lakebase", "pg_hba", "postgres", "database instance"]):
            logger.error(f"Lakebase access error: {e}")
            raise HTTPException(
                status_code=503, detail=get_lakebase_access_error_message(LAKEBASE_INSTANCE_NAME)
            ) from e
        raise
