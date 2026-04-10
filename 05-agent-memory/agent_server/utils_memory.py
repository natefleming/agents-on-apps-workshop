import json
import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from mlflow.types.responses import ResponsesAgentRequest

logger = logging.getLogger(__name__)


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user ID from request custom_inputs or context."""
    custom_inputs = dict(request.custom_inputs or {})
    if "user_id" in custom_inputs:
        return custom_inputs["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


def memory_tools():
    """Create the three long-term memory tools: get, save, delete."""

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory.

        Use this to recall preferences, facts, or details the user has shared in past conversations.

        Args:
            query: A natural language query describing what you're looking for.
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Memory not available - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Memory not available - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))
        results = await store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        memory_items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
        return f"Found {len(results)} relevant memories:\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
        """Save information about the user to long-term memory.

        Use this to remember preferences, facts, or important details the user shares.

        Args:
            memory_key: A short, descriptive key for this memory (e.g., 'language_preference').
            memory_data_json: A JSON object with the memory data (e.g., '{"language": "Python"}').
        """
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot save memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot save memory - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))

        try:
            memory_data = json.loads(memory_data_json)
            if not isinstance(memory_data, dict):
                return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"
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
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id:
            return "Cannot delete memory - no user_id provided."

        store: Optional[BaseStore] = config.get("configurable", {}).get("store")
        if not store:
            return "Cannot delete memory - store not configured."

        namespace = ("user_memories", user_id.replace(".", "-"))
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory '{memory_key}' for user."

    return [get_user_memory, save_user_memory, delete_user_memory]
