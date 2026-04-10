"""
Chapter 5: Agent with Short-Term and Long-Term Memory.

This module builds on Chapter 3 by adding two types of memory:

- Short-term memory (AsyncCheckpointSaver): Persists conversation history
  per thread ID so the agent remembers what was said earlier in a session.
  Uses LangGraph's checkpointing with Lakebase as the storage backend.

- Long-term memory (AsyncDatabricksStore): Stores user facts and preferences
  across sessions using semantic search. The agent has explicit tools to
  save, retrieve, and delete memories.

Both require a Lakebase instance and the databricks-langchain[memory] extra.
"""

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
from langgraph.graph.state import CompiledStateGraph
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
    get_lakebase_access_error_message,
    get_or_create_thread_id,
    process_agent_astream_events,
)
from agent_server.utils_memory import get_user_id, memory_tools

logger: logging.Logger = logging.getLogger(__name__)

# Enable automatic MLflow tracing for all LangChain/LangGraph operations
mlflow.langchain.autolog()

# Service principal workspace client for shared resource access
sp_workspace_client: WorkspaceClient = WorkspaceClient()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LLM model endpoint hosted on Databricks Foundation Model APIs
LLM_ENDPOINT_NAME: str = "databricks-claude-sonnet-4-5"

# Lakebase instance name — required for both short-term and long-term memory.
# Set via LAKEBASE_INSTANCE_NAME in .env.local or app.yaml.
LAKEBASE_INSTANCE_NAME: Optional[str] = os.getenv("LAKEBASE_INSTANCE_NAME") or None

# Embedding model for long-term memory semantic search.
# databricks-gte-large-en produces 1024-dimensional vectors.
EMBEDDING_ENDPOINT: str = "databricks-gte-large-en"
EMBEDDING_DIMS: int = 1024

if not LAKEBASE_INSTANCE_NAME:
    raise ValueError(
        "LAKEBASE_INSTANCE_NAME is required. Set it in your .env.local file.\n"
        "Create a Lakebase instance: databricks lakebase create-database-instance <name> --capacity CU_1"
    )

# System prompt with memory usage instructions.
# This is critical — it tells the LLM when to save, retrieve, and skip memories.
SYSTEM_PROMPT: str = """You are a helpful assistant. Use the available tools to answer questions.

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
        allowed_names: dict[str, object] = {
            "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
        }
        result: object = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Stateful Agent State
#
# The key difference from stateless agents: the `add_messages` annotation
# tells LangGraph to APPEND new messages to the existing list rather than
# replacing it. This is what enables short-term memory — the full
# conversation history accumulates across turns within a thread.
# ---------------------------------------------------------------------------


class StatefulAgentState(TypedDict, total=False):
    """Agent state schema with message accumulation for conversation history."""
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


# ---------------------------------------------------------------------------
# Agent Initialization
# ---------------------------------------------------------------------------


async def init_agent(
    checkpointer: Optional[Any] = None,
    store: Optional[BaseStore] = None,
) -> CompiledStateGraph:
    """Create a LangGraph agent with short-term and long-term memory.

    Args:
        checkpointer: AsyncCheckpointSaver for short-term memory. When provided,
            LangGraph saves/loads message history per thread_id in Lakebase.
        store: AsyncDatabricksStore for long-term memory. When provided, the
            memory tools can save/retrieve user facts with semantic search.

    Returns:
        A compiled LangGraph agent with memory capabilities
    """
    # Combine custom tools with long-term memory tools (get/save/delete)
    tools: list = [get_current_time, calculate] + memory_tools()
    model: ChatDatabricks = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,  # Short-term: persists conversation history
        store=store,                # Long-term: enables memory tool access
        state_schema=StatefulAgentState,  # Uses add_messages for history accumulation
    )


# ---------------------------------------------------------------------------
# MLflow AgentServer Handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests. Returns the thread_id and user_id in custom_outputs
    so the chat UI can maintain session continuity."""
    # Ensure thread_id is set so the streaming handler can use it
    thread_id: str = get_or_create_thread_id(request)
    request.custom_inputs = dict(request.custom_inputs or {})
    request.custom_inputs["thread_id"] = thread_id

    # Collect all completed output items from the stream
    outputs: list = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    # Return thread_id (and user_id if available) so the client can reuse them
    user_id: Optional[str] = get_user_id(request)
    custom_outputs: dict[str, str] = {"thread_id": thread_id}
    if user_id:
        custom_outputs["user_id"] = user_id

    return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming requests with both short-term and long-term memory.

    This function opens two Lakebase connections:
    1. AsyncCheckpointSaver — loads/saves conversation history for this thread
    2. AsyncDatabricksStore — provides semantic search over long-term user memories
    """
    # Extract thread_id (for short-term memory) and user_id (for long-term memory)
    thread_id: str = get_or_create_thread_id(request)
    user_id: Optional[str] = get_user_id(request)

    # Tag the MLflow trace with the session ID for grouping in the UI
    mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    # Build the input state — messages in LangChain format + custom inputs
    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    # LangGraph config — thread_id tells the checkpointer which conversation to load.
    # user_id tells the memory tools which user's memories to access.
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if user_id:
        config["configurable"]["user_id"] = user_id

    try:
        # Open the short-term memory connection (checkpointer).
        # AsyncCheckpointSaver is an async context manager that manages the
        # Lakebase connection pool and creates tables on first use.
        async with AsyncCheckpointSaver(
            instance_name=LAKEBASE_INSTANCE_NAME,
        ) as checkpointer:
            await checkpointer.setup()  # Create checkpoint tables if they don't exist

            # Open the long-term memory connection (store).
            # AsyncDatabricksStore uses an embedding model for semantic search
            # over stored memories.
            async with AsyncDatabricksStore(
                instance_name=LAKEBASE_INSTANCE_NAME,
                embedding_endpoint=EMBEDDING_ENDPOINT,
                embedding_dims=EMBEDDING_DIMS,
            ) as store:
                await store.setup()  # Create store tables if they don't exist

                # Make the store accessible to memory tools via config
                config["configurable"]["store"] = store

                # Create the agent with both memory backends
                agent: CompiledStateGraph = await init_agent(
                    checkpointer=checkpointer, store=store
                )

                # Stream agent execution and convert events to Responses API format
                async for event in process_agent_astream_events(
                    agent.astream(input_state, config, stream_mode=["updates", "messages"])
                ):
                    yield event

    except Exception as e:
        # Provide a helpful error message for Lakebase connection issues
        error_msg: str = str(e).lower()
        if any(kw in error_msg for kw in ["lakebase", "pg_hba", "postgres", "database instance"]):
            logger.error(f"Lakebase access error: {e}")
            raise HTTPException(
                status_code=503,
                detail=get_lakebase_access_error_message(LAKEBASE_INSTANCE_NAME),
            ) from e
        raise
