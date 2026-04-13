"""
Server bootstrap — initializes and runs the MLflow AgentServer.

This file:
1. Loads environment variables from .env.local (including LAKEBASE_PROJECT/LAKEBASE_BRANCH)
2. Imports the agent module to register @invoke/@stream handlers
3. Creates the AgentServer with a built-in chat UI proxy
4. Sets up git-based version tracking for MLflow model versioning

You rarely need to modify this file. Focus on agent.py instead.
"""

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load environment variables BEFORE importing the agent module.
# This ensures LAKEBASE_PROJECT/LAKEBASE_BRANCH and auth config are available.
load_dotenv(dotenv_path=".env.local", override=True)

# Importing the agent module registers @invoke() and @stream() handlers
import agent_server.agent  # noqa: E402, F401

# Create the AgentServer with built-in chat UI at http://localhost:8000
agent_server: AgentServer = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Expose FastAPI app at module level for multi-worker support
app = agent_server.app  # noqa: F841

# Link MLflow model versions to git commits (may fail in deployed environments without .git)
try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    pass


def main() -> None:
    """Entry point for `uv run start-server`."""
    agent_server.run(app_import_string="agent_server.start_server:app")
