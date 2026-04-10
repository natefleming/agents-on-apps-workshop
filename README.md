# Agents on Apps Workshop

Build, test, and deploy AI agents on Databricks Apps. This workshop takes you from zero to a fully deployed agent with tools and MCP server integration.

## What You'll Learn

- How Databricks Apps provides a local-first development experience for AI agents
- How to build a LangGraph agent with custom tools using the MLflow AgentServer
- How to connect your agent to MCP (Model Context Protocol) servers for powerful, pre-built tool capabilities
- How to test locally and deploy to Databricks
- How to use Databricks Asset Bundles for declarative, repeatable deployments

## Prerequisites

- A Databricks workspace with access to Foundation Model APIs (e.g., `databricks-claude-sonnet-4-5`)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [nvm](https://github.com/nvm-sh/nvm) with Node 20 LTS (for the chat UI)
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) installed and configured

## Workshop Chapters

| Chapter | Title | What You'll Build |
|---------|-------|-------------------|
| [01](./01-introduction/) | Introduction to Agents on Apps | Understand the architecture and key concepts |
| [02](./02-hello-agent/) | Hello Agent: Your First LangGraph Agent | A working agent with custom Python tools (weather, calculator) |
| [03](./03-agent-with-mcp/) | Adding MCP Server Tools | Extend your agent with Databricks' built-in code interpreter via MCP |
| [04](./04-deploy-with-bundles/) | Deploying with Asset Bundles | Declarative deployment with `databricks.yml` and multi-target support |

## Quick Start

Each chapter is self-contained with its own working code. To jump straight into building:

```bash
# 1. Clone this repo
git clone <this-repo-url>
cd agents-on-apps-workshop

# 2. Start with Chapter 2 for your first working agent
cd 02-hello-agent

# 3. Set up authentication
databricks auth login

# 4. Set up environment
cp .env.example .env.local
# Edit .env.local with your experiment ID (see chapter README for details)

# 5. Run locally
uv run start-app
# Open http://localhost:8000
```

## Reference

- [Author an Agent (Clone from GitHub)](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) -- Official getting started guide
- [Agent Framework Authentication](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-authentication) -- SP auth vs user-scoped auth
- [Agent Framework Tools](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-tool) -- MCP servers, UC functions, vector search
- [Databricks Apps Auth](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth) -- OBO tokens, user API scopes
- [Databricks Apps Deploy](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy) -- Deploy via CLI
- [Databricks Apps Resources](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/resources) -- Configure app resources and permissions
- [Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) -- Declarative deployment
- [MLflow ResponsesAgent Docs](https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/) -- Agent input/output format
- [Databricks App Templates (GitHub)](https://github.com/databricks/app-templates) -- Official template repository
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) -- Agent orchestration framework
