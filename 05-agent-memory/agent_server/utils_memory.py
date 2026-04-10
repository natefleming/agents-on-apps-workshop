"""
Long-term memory tools for the agent.

This module provides three tools that the LLM can call to manage per-user
long-term memory stored in Lakebase via AsyncDatabricksStore:

- get_user_memory: Semantic search over stored memories
- save_user_memory: Persist a key-value memory for the user
- delete_user_memory: Remove a specific memory

Memories are namespaced by user_id, so each user has isolated storage.
The AsyncDatabricksStore uses embeddings (databricks-gte-large-en) for
semantic search — the agent can find relevant memories even when the
query wording differs from the stored content.
"""

import json
import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

logger: logging.Logger = logging.getLogger(__name__)


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user ID from the request.

    The user_id identifies who the memories belong to. It can come from:
    1. custom_inputs.user_id — explicitly set by the client
    2. context.user_id — provided by the MLflow ChatContext

    Returns None if no user_id is available (memory tools will be disabled).
    """
    custom_inputs: dict = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def memory_tools() -> list:
    """Create the three long-term memory tools: get, save, delete.

    These tools receive a RunnableConfig parameter that LangGraph automatically
    injects at runtime. The config contains the user_id and store that were
    set in the agent's config dict during stream_handler().

    Returns:
        A list of three @tool-decorated async functions
    """

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory.

        Uses semantic search — finds relevant memories even if the query
        wording differs from the stored content.

        Args:
            query: A natural language query describing what you're looking for.
        """
        # Extract user_id and store from the LangGraph runtime config
        user_id: Optional[str] = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Memory not available - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Memory not available - store not configured."

        # Each user's memories are stored in a separate namespace
        namespace: tuple[str, str] = ("user_memories", user_id.replace(".", "-"))

        # Semantic search: finds the 5 most relevant memories for this query
        results = await store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        # Format results as a readable list for the LLM
        memory_items: list[str] = [
            f"- [{item.key}]: {json.dumps(item.value)}" for item in results
        ]
        return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(
        memory_key: str, memory_data_json: str, config: RunnableConfig
    ) -> str:
        """Save information about the user to long-term memory.

        The memory is stored with an embedding for semantic retrieval later.

        Args:
            memory_key: A short, descriptive key for this memory (e.g., 'language_preference').
            memory_data_json: A JSON object with the memory data (e.g., '{"language": "Python"}').
        """
        user_id: Optional[str] = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot save memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot save memory - store not configured."

        namespace: tuple[str, str] = ("user_memories", user_id.replace(".", "-"))

        try:
            # Parse the JSON string into a dict — store requires dict values
            memory_data: Any = json.loads(memory_data_json)
            if not isinstance(memory_data, dict):
                return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"

            # aput() stores the value with an auto-generated embedding for search
            await store.aput(namespace, memory_key, memory_data)
            return f"Successfully saved memory '{memory_key}' for user."
        except json.JSONDecodeError as e:
            return f"Failed to save memory: Invalid JSON - {e}"

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the user's long-term memory.

        Args:
            memory_key: The key of the memory to delete.
        """
        user_id: Optional[str] = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot delete memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot delete memory - store not configured."

        namespace: tuple[str, str] = ("user_memories", user_id.replace(".", "-"))
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory '{memory_key}' for user."

    return [get_user_memory, save_user_memory, delete_user_memory]
