"""
Utility functions for the stateful agent server.

This module provides:
- get_or_create_thread_id(): Extract or generate a thread ID for short-term memory
- get_user_workspace_client(): On-behalf-of (OBO) authentication helper
- get_databricks_host_from_env(): Resolve the Databricks workspace URL
- get_lakebase_access_error_message(): User-friendly Lakebase error messages
- process_agent_astream_events(): Convert LangGraph stream events to Responses API format
"""

import logging
import os
from typing import Any, AsyncGenerator, AsyncIterator, Optional

import uuid_utils
from databricks.sdk import WorkspaceClient
from databricks_langchain.chat_models import json
from langchain.messages import AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)


def get_or_create_thread_id(request: ResponsesAgentRequest) -> str:
    """Get thread ID from request or generate a new one.

    Thread IDs identify a conversation session for short-term memory.
    The checkpointer uses this to load/save the conversation history.

    Resolution priority:
    1. custom_inputs.thread_id — explicitly provided by the client
    2. context.conversation_id — from MLflow ChatContext (set by chat UI)
    3. Auto-generated UUIDv7 — new conversation
    """
    ci: dict[str, Any] = dict(request.custom_inputs or {})

    if "thread_id" in ci and ci["thread_id"]:
        return str(ci["thread_id"])

    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)

    # UUIDv7 is time-ordered, making thread IDs sortable by creation time
    return str(uuid_utils.uuid7())


def get_user_workspace_client() -> WorkspaceClient:
    """Get a workspace client authenticated as the requesting user (OBO).

    Extracts the user's OAuth token from the 'x-forwarded-access-token'
    header injected by the Databricks Apps platform.
    """
    token: Optional[str] = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def get_databricks_host_from_env() -> Optional[str]:
    """Get the Databricks workspace host URL from the SDK's configured auth."""
    try:
        w: WorkspaceClient = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


def get_lakebase_access_error_message(lakebase_instance_name: str) -> str:
    """Generate a user-friendly error message for Lakebase connection failures.

    Provides different guidance depending on whether the agent is running
    in a Databricks App (need to configure app resources) or locally
    (need to check auth and instance name).
    """
    if os.getenv("DATABRICKS_APP_NAME"):
        app_name: Optional[str] = os.getenv("DATABRICKS_APP_NAME")
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            f"The App Service Principal for '{app_name}' may not have access.\n\n"
            "To fix this:\n"
            "1. Go to the Databricks UI and navigate to your app\n"
            "2. Click 'Edit' > 'App resources' > 'Add resource'\n"
            "3. Add your Lakebase instance as a resource\n"
            "4. Grant the necessary permissions on your Lakebase instance."
        )
    else:
        return (
            f"Failed to connect to Lakebase instance '{lakebase_instance_name}'. "
            "Please verify:\n"
            "1. The instance name is correct\n"
            "2. You have the necessary permissions\n"
            "3. Your Databricks authentication is configured correctly"
        )


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Convert LangGraph stream events into MLflow ResponsesAgentStreamEvent objects.

    Handles two event types from agent.astream(stream_mode=["updates", "messages"]):

    1. ("updates", data) — Complete node outputs (tool results, final messages).
       Converted to "response.output_item.done" events.

    2. ("messages", data) — Streaming text chunks (tokens).
       Converted to "response.content_part.delta" events.
    """
    async for event in async_stream:
        event_type: str = event[0]
        event_data: Any = event[1]

        if event_type == "updates":
            # Complete messages from agent graph nodes
            for node_data in event_data.values():
                messages: list = node_data.get("messages", [])
                if len(messages) > 0:
                    # Ensure ToolMessage content is a string for serialization
                    for msg in messages:
                        if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                            msg.content = json.dumps(msg.content)
                    for item in output_to_responses_items_stream(messages):
                        yield item

        elif event_type == "messages":
            # Streaming text chunks (token by token)
            try:
                chunk: Any = event_data[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=content, item_id=chunk.id)
                    )
            except Exception as e:
                logging.exception(f"Error processing agent stream event: {e}")
