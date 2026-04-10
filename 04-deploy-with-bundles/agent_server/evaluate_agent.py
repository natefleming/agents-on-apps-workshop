import asyncio

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import RelevanceToQuery, Safety
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

load_dotenv(dotenv_path=".env.local", override=True)

from agent_server import agent  # noqa: F401

# Evaluation dataset - includes prompts that exercise both custom and MCP tools
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
                "input": [
                    {
                        "role": "user",
                        "content": "Write Python code to calculate the first 10 Fibonacci numbers and return them as a list",
                    }
                ]
            }
        },
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "Write a Python function to check if 97 is prime",
                    }
                ]
            }
        },
    },
]

invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the `@invoke` decorator found. "
    "Ensure you have a function decorated with `@invoke()`."
)

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
