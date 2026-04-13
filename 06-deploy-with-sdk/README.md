# Chapter 6: Deploying with the Databricks Python SDK

In previous chapters, we deployed using the Workspace UI, CLI commands, or Asset Bundles. In this chapter, you'll deploy the same agent **programmatically** using the [Databricks Python SDK](https://docs.databricks.com/aws/en/dev-tools/sdk-python). This is the approach you'd use when deployment is part of a larger Python workflow, CI/CD pipeline, or custom tooling.

## What You'll Learn

- How to create and deploy a Databricks App using `WorkspaceClient().apps`
- How to configure app resources (endpoints, experiments) in Python code
- How to manage the full app lifecycle: create, deploy, update, stop, delete
- When to choose SDK deployment over the other three methods

## What Changed from Chapter 4

The agent code (`agent_server/`) is **identical** to Chapters 3-4. The only difference is **how we deploy it**:

| | CLI (Ch 2-3) | Asset Bundles (Ch 4) | Python SDK (Ch 6) |
|---|---|---|---|
| Config | `app.yaml` | `databricks.yml` | Python code |
| Create app | `databricks apps create` | `databricks bundle deploy` | `w.apps.create()` |
| Sync code | `databricks sync .` | `databricks bundle deploy` | `databricks sync .` |
| Deploy | `databricks apps deploy` | `databricks bundle deploy` | `w.apps.deploy()` |
| Resources | Manual in UI | Declarative YAML | Python objects |
| Best for | Quick iteration | Production, multi-env | CI/CD, custom tooling |

## Step 1: Understand the SDK Deployment Script

The deployment logic lives in `scripts/deploy_app.py`. Here's the core workflow:

### Create an App with Resources

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App, AppDeployment, AppResource,
    AppResourceExperiment, AppResourceServingEndpoint, EnvVariable,
)

w = WorkspaceClient()

# Define what resources the app needs
resources = [
    AppResource(
        name="experiment",
        experiment=AppResourceExperiment(
            experiment_name="/Users/me/agents-workshop-sdk",
            permission="CAN_MANAGE",
        ),
    ),
    AppResource(
        name="llm",
        serving_endpoint=AppResourceServingEndpoint(
            name="databricks-claude-sonnet-4-5",
            permission="CAN_QUERY",
        ),
    ),
]

# Create the app (waits until it's ready)
app = w.apps.create_and_wait(
    App(
        name="my-agent",
        description="My agent app",
        resources=resources,
    )
)
```

### Deploy the App

```python
# Deploy from a workspace path
deployment = w.apps.deploy_and_wait(
    app_name="my-agent",
    app_deployment=AppDeployment(
        source_code_path="/Workspace/Users/me/my-agent",
        env_vars=[
            EnvVariable(name="MLFLOW_TRACKING_URI", value="databricks"),
        ],
    ),
)
print(f"Deployed: {deployment.status.state.value}")
```

### Manage the App Lifecycle

```python
# Check status
app = w.apps.get("my-agent")
print(f"URL: {app.url}")
print(f"Status: {app.app_status.state.value}")

# List all deployments
for d in w.apps.list_deployments("my-agent"):
    print(f"  {d.deployment_id}: {d.status.state.value}")

# Stop the app
w.apps.stop_and_wait("my-agent")

# Restart the app
w.apps.start_and_wait("my-agent")

# Delete the app
w.apps.delete("my-agent")
```

### Additional Resource Types

The SDK supports all the same resource types as Asset Bundles:

```python
from databricks.sdk.service.apps import (
    AppResourceDatabase,
    AppResourceGenieSpace,
    AppResourceSqlWarehouse,
)

# Lakebase database (for memory - Chapter 5)
AppResource(
    name="database",
    database=AppResourceDatabase(
        database_name="databricks_postgres",
        instance_name="my-lakebase",
        permission="CAN_CONNECT_AND_CREATE",
    ),
)

# Genie space
AppResource(
    name="genie",
    genie_space=AppResourceGenieSpace(
        name="My Genie Space",
        space_id="abc123",
        permission="CAN_RUN",
    ),
)

# SQL warehouse
AppResource(
    name="warehouse",
    sql_warehouse=AppResourceSqlWarehouse(
        name="my-warehouse",
        warehouse_id="abc123",
        permission="CAN_USE",
    ),
)
```

## Step 2: Set Up Your Environment

```bash
cd 06-deploy-with-sdk
cp .env.example .env.local
# Edit .env.local with your DATABRICKS_CONFIG_PROFILE and MLFLOW_EXPERIMENT_ID
```

## Step 3: Run Locally

Local development is the same as all other chapters:

```bash
uv sync
uv run start-app
# Open http://localhost:8000
```

## Step 4: Deploy with the SDK

### Sync your code to the workspace

The SDK doesn't have a built-in file sync. Use the CLI for this step:

```bash
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/agents-workshop-sdk"
```

### Run the deployment script

```bash
uv run deploy-app
```

This will:
1. Create the app with configured resources (MLflow experiment, LLM endpoint)
2. Grant the app's service principal the necessary permissions
3. Deploy the app from the synced workspace path
4. Print the app URL when complete

### Customize the deployment

```bash
# Custom app name
uv run deploy-app --app-name my-custom-agent

# Custom source path
uv run deploy-app --source-code-path /Workspace/Users/me/my-agent

# Different LLM endpoint
uv run deploy-app --llm-endpoint databricks-meta-llama-3-3-70b-instruct
```

## Step 5: Evaluate Your Agent

```bash
uv run agent-evaluate
```

See [Chapter 2](../02-hello-agent/README.md#step-6-evaluate-your-agent) for a detailed explanation of how evaluation works.

## When to Use SDK Deployment

| Scenario | Recommended Method |
|----------|-------------------|
| Exploring for the first time | **Workspace UI** |
| Quick iteration during development | **CLI** |
| Production, multi-environment | **Asset Bundles** |
| CI/CD pipelines with custom logic | **Python SDK** |
| Deploying from Databricks notebooks | **Python SDK** |
| Dynamic app provisioning (multi-tenant) | **Python SDK** |
| Integration testing frameworks | **Python SDK** |

## Key Takeaways

- **`WorkspaceClient().apps`** provides the full app lifecycle: create, deploy, update, stop, delete
- **Resources are Python objects** -- `AppResource`, `AppResourceServingEndpoint`, etc. -- giving you type safety and IDE autocomplete
- **`create_and_wait()` / `deploy_and_wait()`** block until the operation completes, simplifying scripts
- **File sync still uses the CLI** -- the SDK manages app metadata and deployments, not file uploads
- **Same agent code** -- only the deployment mechanism changes; your `agent_server/` code is identical

## Reference

- [Databricks SDK for Python](https://docs.databricks.com/aws/en/dev-tools/sdk-python) -- SDK overview and installation
- [Apps API Reference](https://docs.databricks.com/api/workspace/apps) -- REST API that the SDK wraps
- [Databricks Apps Deploy](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy) -- General deployment docs
- [Author an Agent](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent) -- Getting started guide

## What's Next

You've now seen all four ways to deploy agents on Databricks Apps. Choose the approach that best fits your workflow:

- **UI** for exploration
- **CLI** for development
- **Asset Bundles** for production
- **SDK** for automation and CI/CD
