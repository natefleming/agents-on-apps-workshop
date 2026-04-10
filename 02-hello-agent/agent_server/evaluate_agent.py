import asyncio

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import RelevanceToQuery, Safety
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Load environment variables from .env.local if it exists
load_dotenv(dotenv_path=".env.local", override=True)

# Need to import agent for our @invoke-registered function to be found
from agent_server import agent  # noqa: F401

# Evaluation dataset - test cases for our agent's tools
eval_dataset = [
    {
        "inputs": {
            "request": {
                "input": [{"role": "user", "content": "What time is it right now?"}]
            }
        },
    },
    {
        "inputs": {
            "request": {
                "input": [{"role": "user", "content": "What is 42 multiplied by 17?"}]
            }
        },
        "expected_response": "42 multiplied by 17 is 714.",
    },
    {
        "inputs": {
            "request": {
                "input": [{"role": "user", "content": "Tell me a joke about programming"}]
            }
        },
    },
]

# Get the invoke function registered via @invoke decorator
invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the `@invoke` decorator found. "
    "Ensure you have a function decorated with `@invoke()`."
)

# Wrap async invoke in sync function for evaluation
if asyncio.iscoroutinefunction(invoke_fn):

    def sync_invoke_fn(request: dict) -> ResponsesAgentResponse:
        req = ResponsesAgentRequest(**request)
        return asyncio.run(invoke_fn(req))
else:
    sync_invoke_fn = invoke_fn


def evaluate():
    mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=sync_invoke_fn,
        scorers=[RelevanceToQuery(), Safety()],
    )
