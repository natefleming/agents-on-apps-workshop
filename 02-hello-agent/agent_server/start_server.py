"""
Server bootstrap — initializes and runs the MLflow AgentServer.

This file:
1. Loads environment variables from .env.local (for local development)
2. Imports the agent module to register @invoke/@stream handlers
3. Creates the AgentServer with a built-in chat UI proxy
4. Sets up git-based version tracking for MLflow model versioning

You rarely need to modify this file. Focus on agent.py instead.
"""

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load environment variables BEFORE importing the agent module.
# This ensures DATABRICKS_CONFIG_PROFILE, MLFLOW_EXPERIMENT_ID, etc.
# are available when the agent initializes its Databricks connections.
load_dotenv(dotenv_path=".env.local", override=True)

# Importing the agent module triggers the @invoke() and @stream() decorators,
# which register the handler functions with MLflow's global registry.
# This MUST happen before creating the AgentServer.
import agent_server.agent  # noqa: E402, F401

# Create the AgentServer. "ResponsesAgent" indicates we use the OpenAI
# Responses API format for input/output. enable_chat_proxy=True serves
# a built-in chat UI at the root URL (http://localhost:8000).
agent_server: AgentServer = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Expose the FastAPI app at module level so uvicorn can find it when
# running with multiple workers (e.g., --workers 4).
app = agent_server.app  # noqa: F841

# Link MLflow model versions to git commits (may fail in deployed environments without .git)
try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    pass


def main() -> None:
    """Entry point for `uv run start-server`."""
    agent_server.run(app_import_string="agent_server.start_server:app")
