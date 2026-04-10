from datetime import datetime
from typing import AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool
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

mlflow.langchain.autolog()
sp_workspace_client = WorkspaceClient()

# ---------------------------------------------------------------------------
# Custom Tools - Same as Chapter 2
# These are simple Python functions that the agent can call directly.
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
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# Custom tools list
CUSTOM_TOOLS = [get_current_time, calculate]

# ---------------------------------------------------------------------------
# MCP Server Configuration
# Connect to Databricks MCP servers to get additional tools (like code execution).
# ---------------------------------------------------------------------------


def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    """
    Initialize MCP client connected to Databricks MCP servers.

    The system.ai MCP server provides tools like python_exec (code interpreter)
    that are available in every Databricks workspace.
    """
    host_name = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="system-ai",
                url=f"{host_name}/api/2.0/mcp/functions/system/ai",
            ),
            # Uncomment to add more MCP servers:
            #
            # DatabricksMCPServer(
            #     name="sql",
            #     url=f"{host_name}/api/2.0/mcp/sql?warehouse_id=YOUR_WAREHOUSE_ID",
            # ),
            # DatabricksMCPServer(
            #     name="my-uc-functions",
            #     url=f"{host_name}/api/2.0/mcp/functions/my_catalog/my_schema",
            # ),
        ]
    )


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

MODEL = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    """
    Create a LangGraph agent with both custom tools and MCP tools.

    Custom tools (get_current_time, calculate) handle simple tasks.
    MCP tools (python_exec, etc.) handle complex tasks like code execution.
    """
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    mcp_tools = await mcp_client.get_tools()

    # Combine custom tools with MCP-discovered tools
    all_tools = CUSTOM_TOOLS + mcp_tools

    return create_agent(tools=all_tools, model=MODEL)


# ---------------------------------------------------------------------------
# MLflow AgentServer handlers
# ---------------------------------------------------------------------------


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests."""
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
    """Handle streaming requests."""
    # To use on-behalf-of authentication, uncomment the next line:
    # user_workspace_client = get_user_workspace_client()
    agent = await init_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
