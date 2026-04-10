# Chapter 4: Deploying with Databricks Asset Bundles

In the previous chapters, we deployed our agent using manual CLI commands (`databricks apps create`, `databricks sync`, `databricks apps deploy`). This works fine for development, but for production you want **declarative, repeatable deployments**. That's what Databricks Asset Bundles (DABs) provide.

In this chapter, you'll define your entire deployment -- the app, its resources, permissions, and environment configuration -- in a single `databricks.yml` file. No separate `app.yaml` needed.

## What are Databricks Asset Bundles?

[Databricks Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/) are a declarative way to define and deploy Databricks resources. Think of them like Terraform or CloudFormation, but specifically for Databricks.

A bundle can declare:
- **Apps** - with their config, resources, and permissions
- **MLflow Experiments** - for tracing and evaluation
- **Jobs** - for scheduled tasks
- **Pipelines** - for data engineering
- **And more**

## What You'll Learn

- How to replace `app.yaml` + manual CLI commands with a single `databricks.yml`
- How to declare app resources (serving endpoints, experiments) and permissions
- How to set up multi-target deployment (dev/staging/prod)
- How to use variables for environment-specific configuration
- How to deploy with `databricks bundle deploy`

## What Changed from Chapter 3

The agent code (`agent_server/`) is **identical** to Chapter 3. The only difference is **how we deploy it**:

| | Chapters 2-3 | Chapter 4 |
|---|---|---|
| Config file | `app.yaml` | `databricks.yml` |
| Create app | `databricks apps create` | `databricks bundle deploy` |
| Sync code | `databricks sync .` | `databricks bundle deploy` |
| Deploy | `databricks apps deploy` | `databricks bundle deploy` |
| Resources | Manual via UI | Declared in YAML |
| Permissions | Manual via UI | Declared in YAML |
| Multi-env | Manual per workspace | Targets in YAML |

One command -- `databricks bundle deploy` -- does everything.

## Step 1: Understand `databricks.yml`

Here's the complete bundle configuration for our agent:

```yaml
bundle:
  name: agents-workshop-mcp

variables:
  resource_name_suffix:
    description: "Suffix for resource names to avoid collisions"

resources:
  # MLflow Experiment for agent tracing
  experiments:
    agent_experiment:
      name: /Users/${workspace.current_user.userName}/agents-workshop-mcp
      permissions:
        - level: CAN_MANAGE
          group_name: users

  # The Databricks App
  apps:
    agent_app:
      name: agents-workshop-${var.resource_name_suffix}
      description: "Workshop agent with custom tools and MCP integration"
      source_code_path: .

      # App runtime configuration (replaces app.yaml)
      config:
        command:
          - uv
          - run
          - start-app
        env:
          - name: MLFLOW_TRACKING_URI
            value: "databricks"
          - name: MLFLOW_REGISTRY_URI
            value: "databricks-uc"
          - name: API_PROXY
            value: "http://localhost:8000/invocations"
          - name: CHAT_APP_PORT
            value: "3000"
          - name: CHAT_PROXY_TIMEOUT_SECONDS
            value: "300"

      # Resources the app needs access to
      resources:
        - name: experiment
          description: "MLflow experiment for tracing"
          experiment:
            experiment_name: ${resources.experiments.agent_experiment.name}
            permission: CAN_MANAGE
        - name: llm
          description: "LLM serving endpoint"
          serving_endpoint:
            name: databricks-claude-sonnet-4
            permission: CAN_QUERY

      # Who can use the app
      permissions:
        - level: CAN_USE
          group_name: users

      # On-behalf-of user scopes (optional)
      # user_api_scopes:
      #   - sql
      #   - serving.serving-endpoints

# Deployment targets
targets:
  dev:
    mode: development
    default: true
    variables:
      resource_name_suffix: dev-${workspace.current_user.domain_friendly_name}

  staging:
    mode: production
    variables:
      resource_name_suffix: "staging"

  prod:
    mode: production
    variables:
      resource_name_suffix: "prod"
```

Let's break this down section by section.

### `bundle:` - Bundle Metadata

```yaml
bundle:
  name: agents-workshop-mcp
```

The bundle name identifies this project. It's used in workspace paths and resource naming.

### `variables:` - Parameterized Config

```yaml
variables:
  resource_name_suffix:
    description: "Suffix for resource names to avoid collisions"
```

Variables let you parameterize the deployment. Different targets can set different values. For example, in `dev` mode, the suffix includes your username so multiple developers can deploy without name conflicts.

### `resources.experiments:` - MLflow Experiment

```yaml
resources:
  experiments:
    agent_experiment:
      name: /Users/${workspace.current_user.userName}/agents-workshop-mcp
      permissions:
        - level: CAN_MANAGE
          group_name: users
```

This creates the MLflow experiment automatically -- no more manual `databricks experiments create-experiment` commands.

### `resources.apps:` - The App Definition

The `config:` block **replaces `app.yaml`** entirely:

```yaml
config:
  command:
    - uv
    - run
    - start-app
  env:
    - name: MLFLOW_TRACKING_URI
      value: "databricks"
```

The `resources:` block declares what Databricks resources the app needs. The bundle automatically grants the app's service principal the specified permissions:

```yaml
resources:
  - name: llm
    serving_endpoint:
      name: databricks-claude-sonnet-4
      permission: CAN_QUERY
```

### `targets:` - Multi-Environment Deployment

```yaml
targets:
  dev:
    mode: development  # Pauses schedules, prefixes names with [dev]
    default: true
  prod:
    mode: production   # Full deployment
```

## Step 2: Deploy

### Deploy to Dev (Default)

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy everything: create app, set permissions, configure resources
databricks bundle deploy
```

That's it. One command creates the app, the MLflow experiment, grants permissions, syncs the code, and deploys.

### Deploy to a Specific Target

```bash
# Deploy to staging
databricks bundle deploy -t staging

# Deploy to production
databricks bundle deploy -t prod
```

### Check Deployment Status

```bash
# See what's deployed
databricks bundle summary

# Destroy all resources (careful!)
databricks bundle destroy
```

## Step 3: Local Development

Local development is the same as Chapter 3. The `databricks.yml` is only used for deployment.

```bash
cp .env.example .env.local
# Edit .env.local with your config

uv sync
uv run start-app
# Open http://localhost:8000
```

## Step 4: Sync Configuration for Clean Deployments

Add a `sync:` section to control which files get uploaded to the workspace:

```yaml
sync:
  include:
    - "agent_server/**"
    - "scripts/**"
    - "pyproject.toml"
    - "uv.lock"
  exclude:
    - ".venv/**"
    - "__pycache__/**"
    - "*.pyc"
    - ".git/**"
    - ".databricks/**"
    - "e2e-chatbot-app-next/**"
    - "backend.log"
    - "frontend.log"
    - ".env.local"
```

This ensures only necessary files are deployed, keeping the workspace clean and deployments fast.

## Adding More Resources

### Genie Space

```yaml
resources:
  - name: my-genie
    description: "Genie space for data exploration"
    genie_space:
      name: My Genie Space
      space_id: 01f05dd06c421ad6b522bf7a517cf6d2
      permission: CAN_RUN
```

### Lakebase Database

```yaml
resources:
  database_instances:
    app_db:
      name: agent-workshop-db
      capacity: CU_1

  apps:
    agent_app:
      resources:
        - name: database
          database:
            database_name: databricks_postgres
            instance_name: ${resources.database_instances.app_db.name}
            permission: CAN_CONNECT_AND_CREATE
```

### On-Behalf-Of User Scopes

For apps that need to access Databricks APIs as the logged-in user:

```yaml
apps:
  agent_app:
    user_api_scopes:
      - sql                           # Query SQL warehouses
      - serving.serving-endpoints     # List/query serving endpoints
      - catalog.catalogs:read         # Read UC catalogs
      - catalog.schemas:read          # Read UC schemas
      - catalog.tables:read           # Read UC tables
      - dashboards.genie              # Access Genie spaces
```

## Key Takeaways

- **`databricks.yml` replaces `app.yaml`** by embedding the app config in the `config:` block
- **One command deploys everything**: `databricks bundle deploy` handles app creation, code sync, resource permissions, and deployment
- **Multi-target support** lets you deploy to dev/staging/prod with different configurations
- **Variables** parameterize your deployment to avoid name collisions and customize per environment
- **Resources are declarative** -- serving endpoints, experiments, genie spaces, and databases are all defined in YAML
- **Permissions are automatic** -- the bundle grants the app's service principal the exact permissions it needs

## Reference

- [Databricks Asset Bundles Docs](https://docs.databricks.com/aws/en/dev-tools/bundles/)
- [Bundle App Resource](https://docs.databricks.com/aws/en/dev-tools/bundles/resources/apps)
- [Bundle Configuration Schema](https://docs.databricks.com/aws/en/dev-tools/bundles/settings)
