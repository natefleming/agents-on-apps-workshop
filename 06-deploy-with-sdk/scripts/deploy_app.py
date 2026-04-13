"""
Deploy an agent app to Databricks using the Python SDK.

This script demonstrates programmatic deployment of a Databricks App,
including creating the app, configuring resources and permissions,
syncing source code, and deploying.

Usage:
    uv run deploy-app                          # Deploy with defaults
    uv run deploy-app --app-name my-agent      # Custom app name
    uv run deploy-app --source-code-path /Workspace/Users/me/agent
"""

import argparse
import sys
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App,
    AppDeployment,
    AppResource,
    AppResourceExperiment,
    AppResourceExperimentExperimentPermission,
    AppResourceServingEndpoint,
    AppResourceServingEndpointServingEndpointPermission,
    EnvVar,
)


def get_current_username(w: WorkspaceClient) -> str:
    """Get the current user's username from the workspace."""
    return w.current_user.me().user_name


def create_or_get_app(
    w: WorkspaceClient,
    app_name: str,
    description: str,
    resources: list[AppResource],
) -> App:
    """Create a new app or return and update the existing one."""
    from databricks.sdk.errors import NotFound

    try:
        existing = w.apps.get(app_name)
        print(f"  App '{app_name}' already exists (status: {existing.app_status.state.value})")
        updated = w.apps.update(
            name=app_name,
            app=App(name=app_name, description=description, resources=resources),
        )
        print(f"  Updated app resources")
        return updated
    except NotFound:
        pass

    print(f"  Creating app '{app_name}'...")
    app = w.apps.create_and_wait(
        App(
            name=app_name,
            description=description,
            resources=resources,
        )
    )
    print(f"  App created (status: {app.app_status.state.value})")
    return app


def deploy_app(
    w: WorkspaceClient,
    app_name: str,
    source_code_path: str,
    env_vars: list[EnvVar] | None = None,
) -> AppDeployment:
    """Deploy (or redeploy) the app from the given source code path."""
    print(f"  Deploying from {source_code_path}...")
    deployment = w.apps.deploy_and_wait(
        app_name=app_name,
        app_deployment=AppDeployment(
            source_code_path=source_code_path,
            env_vars=env_vars or [],
        ),
    )
    print(f"  Deployment {deployment.deployment_id} status: {deployment.status.state.value}")
    return deployment


def main():
    parser = argparse.ArgumentParser(description="Deploy agent app via Databricks SDK")
    parser.add_argument("--app-name", default="agents-workshop-sdk", help="App name")
    parser.add_argument("--source-code-path", default=None, help="Workspace path for source code")
    parser.add_argument("--experiment-name", default=None, help="MLflow experiment name")
    parser.add_argument("--llm-endpoint", default="databricks-claude-sonnet-4-5", help="LLM endpoint")
    args = parser.parse_args()

    w = WorkspaceClient()
    username = get_current_username(w)
    print(f"Authenticated as: {username}")

    # Default paths based on current user
    source_code_path = args.source_code_path or f"/Workspace/Users/{username}/agents-workshop-sdk"
    experiment_name = args.experiment_name or f"/Users/{username}/agents-workshop-sdk"

    # =========================================================================
    # Step 1: Define app resources
    # =========================================================================
    print("\n[1/4] Configuring resources...")

    # Create or get the MLflow experiment to obtain its ID
    import mlflow
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"  Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"  Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

    resources = [
        AppResource(
            name="experiment",
            description="MLflow experiment for tracing agent interactions",
            experiment=AppResourceExperiment(
                experiment_id=experiment_id,
                permission=AppResourceExperimentExperimentPermission.CAN_MANAGE,
            ),
        ),
        AppResource(
            name="llm",
            description="Foundation model endpoint for the agent",
            serving_endpoint=AppResourceServingEndpoint(
                name=args.llm_endpoint,
                permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY,
            ),
        ),
    ]
    for r in resources:
        print(f"  - {r.name}: {r.description}")

    # =========================================================================
    # Step 2: Create the app
    # =========================================================================
    print("\n[2/4] Creating app...")
    app = create_or_get_app(
        w,
        app_name=args.app_name,
        description="Workshop agent deployed via Databricks Python SDK",
        resources=resources,
    )

    # =========================================================================
    # Step 3: Sync source code
    # =========================================================================
    print(f"\n[3/4] Sync source code to {source_code_path}")
    print(f"  Run: databricks sync . \"{source_code_path}\"")
    print("  (The SDK does not have a built-in sync command; use the CLI for this step.)")

    # =========================================================================
    # Step 4: Deploy
    # =========================================================================
    print("\n[4/4] Deploying app...")
    env_vars = [
        EnvVar(name="MLFLOW_TRACKING_URI", value="databricks"),
        EnvVar(name="MLFLOW_REGISTRY_URI", value="databricks-uc"),
    ]
    deployment = deploy_app(w, args.app_name, source_code_path, env_vars)

    # =========================================================================
    # Summary
    # =========================================================================
    app = w.apps.get(args.app_name)
    print(f"\nApp URL: {app.url}")
    print(f"Status:  {app.app_status.state.value}")
    print(f"\nTo query your deployed agent:")
    print(f"  databricks auth token")
    print(f"  curl -X POST {app.url}/invocations \\")
    print(f'    -H "Authorization: Bearer <token>" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"input\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}], \"stream\": true}}'")


if __name__ == "__main__":
    main()
