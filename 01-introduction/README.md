# Chapter 1: Introduction to Agents on Apps

## What is an "Agent on an App"?

An **Agent on an App** is an AI agent deployed as a Databricks App. Instead of deploying your agent behind a Model Serving endpoint, you deploy it as a full application that:

- Runs as a **FastAPI server** with the MLflow `AgentServer`
- Includes a **built-in chat UI** (or can serve a custom frontend)
- Supports **local development** with hot-reload, so you iterate fast on your laptop
- Deploys to **Databricks Apps** with a single CLI command
- Gets **automatic tracing** via MLflow for observability and debugging
- Supports **on-behalf-of (OBO) authentication** so users interact with Databricks resources using their own credentials

This is the recommended approach for building and deploying agents on Databricks.

## Architecture Overview

```
+-----------------------------------------------------------+
|                    Databricks App                           |
|                                                             |
|  +-----------------+     +-------------------------------+  |
|  |   Chat UI        |     |   MLflow AgentServer (FastAPI) | |
|  |   (Next.js)      |---->|   /invocations endpoint       | |
|  |   Port 8000      |     |                               | |
|  +-----------------+     |   +-------------------------+ | |
|                           |   |   Your Agent Code       | | |
|                           |   |   (LangGraph/LangChain) | | |
|                           |   |                         | | |
|                           |   |   @invoke() / @stream() | | |
|                           |   +-------------------------+ | |
|                           +---------|--------|------------+ |
+----------------------------|--------|--------|-------------+
                              |        |        |
                     +--------+   +----+   +----+--------+
                     |            |         |             |
              +------v---+  +----v----+  +-v-----------+ |
              | LLM      |  | MCP     |  | Custom      | |
              | Endpoint  |  | Servers |  | Tools       | |
              | (Claude,  |  | (Code   |  | (@tool)     | |
              |  GPT, etc)|  |  Exec,  |  |             | |
              +----------+  |  SQL)   |  +-------------+ |
                             +---------+                   |
```

## Key Components

### 1. MLflow AgentServer

The `AgentServer` from `mlflow.genai.agent_server` is the backbone. It:

- Wraps your agent code in a FastAPI application
- Exposes a `/invocations` endpoint that accepts the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) format
- Handles both streaming and non-streaming responses
- Automatically traces all agent interactions to MLflow
- Optionally serves a chat UI via `enable_chat_proxy=True`

### 2. The `@invoke()` and `@stream()` Decorators

These decorators register your agent functions with the AgentServer:

```python
from mlflow.genai.agent_server import invoke, stream

@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Called for non-streaming requests"""
    ...

@stream()
async def streaming(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Called for streaming requests - yields events as they happen"""
    ...
```

### 3. LangGraph + LangChain

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework for building stateful, multi-step agents. Combined with LangChain, it provides:

- **Agent orchestration**: The agent decides which tools to call and when
- **Tool integration**: Easily add Python functions as tools via `@tool`
- **MCP support**: Connect to MCP servers for pre-built tool catalogs
- **Streaming**: Native support for streaming token-by-token responses

### 4. Databricks Foundation Models

Your agent uses Databricks-hosted LLMs via the `ChatDatabricks` class:

```python
from databricks_langchain import ChatDatabricks

model = ChatDatabricks(endpoint="databricks-claude-sonnet-4")
```

This works both locally (using your Databricks CLI credentials) and when deployed (using the app's service principal).

### 5. MCP (Model Context Protocol) Servers

MCP is an open protocol that lets your agent connect to external tool servers. Databricks provides built-in MCP servers for:

- **`system.ai`**: Code interpreter (Python execution), and other AI tools
- **SQL**: Query SQL warehouses
- **Vector Search**: Search vector indexes
- **Genie**: Natural language data exploration
- **UC Functions**: Call Unity Catalog functions

You can also build and connect to custom MCP servers.

## Project Structure

Every agent app in this workshop follows this structure:

```
my-agent/
├── app.yaml                 # Databricks App deployment config
├── pyproject.toml           # Python dependencies and entry points
├── .env.example             # Environment variable template
├── .env.local               # Your local environment (git-ignored)
├── agent_server/
│   ├── __init__.py
│   ├── agent.py             # Your agent logic (this is where you code)
│   ├── start_server.py      # AgentServer initialization (rarely modified)
│   └── utils.py             # Helper functions
```

### `app.yaml` - Deployment Configuration

Tells Databricks how to run your app:

```yaml
command: ["uv", "run", "start-app"]

env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_EXPERIMENT_ID
    valueFrom: "experiment"    # Bound from Databricks App resources
```

### `pyproject.toml` - Dependencies and Scripts

Manages Python packages and defines entry points:

```toml
[project.scripts]
start-app = "scripts.start_app:main"        # Start backend + chat UI
start-server = "agent_server.start_server:main"  # Start backend only
```

### `agent.py` - Your Agent

This is where you spend most of your time. It defines:

1. What **model** your agent uses
2. What **tools** your agent has access to
3. How **requests are processed** (the `@invoke` and `@stream` functions)

### `start_server.py` - Server Bootstrap

Initializes the AgentServer. You rarely need to modify this:

```python
from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer

load_dotenv(dotenv_path=".env.local", override=True)

import agent_server.agent  # Register @invoke/@stream decorators

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app
```

## Development Workflow

### Local Development

```bash
# 1. Authenticate with Databricks
databricks auth login

# 2. Set up environment
cp .env.example .env.local
# Edit .env.local with your MLflow experiment ID

# 3. Install dependencies
uv sync

# 4. Run locally with chat UI
uv run start-app
# Open http://localhost:8000

# 5. Or run just the backend with hot-reload
uv run start-server --reload
```

### Testing via API

```bash
# Streaming
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "hello" }], "stream": true }'

# Non-streaming
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "hello" }] }'
```

### Deploy to Databricks

```bash
# 1. Create the app
databricks apps create my-agent

# 2. Add resources (MLflow experiment, serving endpoints) via the app's edit page

# 3. Sync code to workspace
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/my-agent"

# 4. Deploy
databricks apps deploy my-agent \
  --source-code-path /Workspace/Users/$DATABRICKS_USERNAME/my-agent

# 5. Query (must use OAuth token, not PAT)
databricks auth token
curl -X POST <app-url>/invocations \
  -H "Authorization: Bearer <oauth-token>" \
  -H "Content-Type: application/json" \
  -d '{ "input": [{ "role": "user", "content": "hello" }], "stream": true }'
```

## What's Next

In [Chapter 2](../02-hello-agent/), you'll build your first agent with custom Python tools and test it locally.
