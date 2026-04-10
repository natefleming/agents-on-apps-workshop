"""
Utility functions for the agent server.

This module provides:
- get_user_workspace_client(): On-behalf-of (OBO) authentication helper
- get_databricks_host_from_env(): Resolve the Databricks workspace URL
- process_agent_astream_events(): Convert LangGraph stream events to
  MLflow ResponsesAgentStreamEvent format
"""

import logging
from typing import Any, AsyncGenerator, AsyncIterator, Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain.chat_models import json
from langchain.messages import AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)


def get_user_workspace_client() -> WorkspaceClient:
    """Get a workspace client authenticated as the requesting user (OBO).

    When running in a Databricks App, the platform injects the user's
    OAuth token via the 'x-forwarded-access-token' HTTP header. This
    function extracts that token and creates a WorkspaceClient that
    acts on behalf of the logged-in user.
    """
    token: Optional[str] = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def get_databricks_host_from_env() -> Optional[str]:
    """Get the Databricks workspace host URL from the SDK's configured auth.

    Returns the host URL (e.g., 'https://my-workspace.cloud.databricks.com')
    or None if it cannot be determined.
    """
    try:
        w: WorkspaceClient = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Convert LangGraph stream events into MLflow ResponsesAgentStreamEvent objects.

    LangGraph's agent.astream() with stream_mode=["updates", "messages"] produces
    two types of events that we need to convert to the Responses API format:

    1. ("updates", data) — Complete node outputs containing tool call results
       and final assistant messages. We convert these into
       "response.output_item.done" events via output_to_responses_items_stream().

    2. ("messages", data) — Streaming text chunks (individual tokens) from the
       LLM. We convert these into "response.content_part.delta" events via
       create_text_delta().

    Args:
        async_stream: The async iterator from agent.astream()

    Yields:
        ResponsesAgentStreamEvent objects in Responses API format
    """
    async for event in async_stream:
        event_type: str = event[0]
        event_data: Any = event[1]

        if event_type == "updates":
            # "updates" contain complete messages from agent graph nodes.
            # Each node (e.g., "agent", "tools") produces a dict of outputs.
            for node_data in event_data.values():
                messages: list = node_data.get("messages", [])
                if len(messages) > 0:
                    # ToolMessage content must be a string for serialization.
                    # If a tool returned a dict/list, convert it to JSON.
                    for msg in messages:
                        if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                            msg.content = json.dumps(msg.content)

                    # Convert LangChain messages to Responses API output items
                    for item in output_to_responses_items_stream(messages):
                        yield item

        elif event_type == "messages":
            # "messages" contain streaming text chunks (token by token).
            # Each chunk is an AIMessageChunk with partial text content.
            try:
                chunk: Any = event_data[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    # Create a text delta event for this token
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=content, item_id=chunk.id)
                    )
            except Exception as e:
                logging.exception(f"Error processing agent stream event: {e}")
