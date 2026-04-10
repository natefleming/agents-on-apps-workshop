"""
Chapter 3: Agent with MCP — Combines custom tools with MCP server tools.

This module extends Chapter 2 by connecting to Databricks MCP servers,
which provide pre-built tools (like a Python code interpreter) that the
agent can call without any custom tool code.

Key concepts:
- DatabricksMultiServerMCPClient connects to one or more MCP servers
- mcp_client.get_tools() discovers available tools at runtime
- Custom tools and MCP tools are combined into a single tool list
"""

from datetime import datetime
from typing import AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
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

from agent_server.utils import (
    get_databricks_host_from_env,
    get_user_workspace_client,
    process_agent_astream_events,
)

# Enable automatic MLflow tracing for all LangChain/LangGraph operations
mlflow.langchain.autolog()

# Create a service principal workspace client for MCP server connections.
# This is used when the app accesses shared resources (not user-specific).
sp_workspace_client: WorkspaceClient = WorkspaceClient()


# ---------------------------------------------------------------------------
# Custom Tools — Same as Chapter 2
# These handle simple, well-defined tasks directly in Python.
# ---------------------------------------------------------------------------


@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks what time it is or what today's date is."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool for simple, quick math calculations.

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


# List of custom tools defined in this file
CUSTOM_TOOLS: list = [get_current_time, calculate]


# ---------------------------------------------------------------------------
# MCP Server Configuration
#
# MCP (Model Context Protocol) servers expose tool catalogs over HTTP.
# DatabricksMultiServerMCPClient connects to one or more servers and
# merges all their tools into a single list the agent can use.
# ---------------------------------------------------------------------------


def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    """Initialize an MCP client connected to Databricks MCP servers.

    The system.ai UC Functions MCP server provides tools like python_exec
    (a code interpreter) that are available in every Databricks workspace.

    Args:
        workspace_client: Authenticated WorkspaceClient for API access

    Returns:
        A multi-server MCP client that can discover tools from all connected servers
    """
    host_name: Optional[str] = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient(
        [
            # UC Functions MCP server for system.ai (code interpreter, etc.)
            DatabricksMCPServer(
                name="system-ai",
                url=f"{host_name}/api/2.0/mcp/functions/system/ai",
            ),
            # Uncomment to add more MCP servers:
            #
            # SQL MCP server — query a SQL warehouse
            # DatabricksMCPServer(
            #     name="sql",
            #     url=f"{host_name}/api/2.0/mcp/sql?warehouse_id=YOUR_WAREHOUSE_ID",
            # ),
            #
            # UC Functions MCP server — call functions in a specific catalog/schema
            # DatabricksMCPServer(
            #     name="my-uc-functions",
            #     url=f"{host_name}/api/2.0/mcp/functions/my_catalog/my_schema",
            # ),
        ]
    )


# ---------------------------------------------------------------------------
# Agent Initialization
# ---------------------------------------------------------------------------

# Configure the LLM
MODEL: ChatDatabricks = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")


async def init_agent(workspace_client: Optional[WorkspaceClient] = None) -> CompiledStateGraph:
    """Create a LangGraph agent with both custom tools and MCP-discovered tools.

    This async function:
    1. Connects to the MCP server(s) and discovers available tools
    2. Combines them with our custom Python tools
    3. Creates a LangGraph ReAct agent that can call any of them

    Args:
        workspace_client: Optional WorkspaceClient override (for OBO auth)

    Returns:
        A compiled LangGraph agent ready to process messages
    """
    # Discover tools from MCP servers (this makes an HTTP call to each server)
    mcp_client: DatabricksMultiServerMCPClient = init_mcp_client(
        workspace_client or sp_workspace_client
    )
    mcp_tools: list = await mcp_client.get_tools()

    # Combine custom tools with MCP-discovered tools into one list.
    # The LLM sees all tool descriptions and decides which to call.
    all_tools: list = CUSTOM_TOOLS + mcp_tools

    return create_agent(tools=all_tools, model=MODEL)


# ---------------------------------------------------------------------------
# MLflow AgentServer Handlers
# ---------------------------------------------------------------------------


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests by collecting all stream events."""
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

    To use on-behalf-of authentication (so MCP calls use the logged-in
    user's identity), uncomment the user_workspace_client line and pass
    it to init_agent().
    """
    # Uncomment for OBO auth:
    # user_workspace_client: WorkspaceClient = get_user_workspace_client()
    # agent = await init_agent(workspace_client=user_workspace_client)
    agent: CompiledStateGraph = await init_agent()

    # Convert Responses API input to LangChain message format
    messages: dict[str, list] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input])
    }

    # Stream agent execution and convert events to Responses API format
    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
