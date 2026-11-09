"""Check tool argument parsing and a tool-result continuation; execute no tools."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    session = requests.Session()
    tool = {
        "type": "function",
        "function": {
            "name": "get_build_status",
            "description": "Read the status of a build by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"build_id": {"type": "string"}},
                "required": ["build_id"],
                "additionalProperties": False,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Check build-17 using get_build_status. Do not guess the status.",
        }
    ]
    settings = dict(
        model="qwen",
        temperature=0,
        max_tokens=256,
        chat_template_kwargs={"enable_thinking": False},
    )
    first = session.post(
        args.url + "/v1/chat/completions",
        json=dict(settings, messages=messages, tools=[tool], tool_choice="required"),
        timeout=180,
    )
    first.raise_for_status()
    first_data = first.json()
    message = first_data["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    valid = len(calls) == 1 and calls[0]["function"]["name"] == "get_build_status"
    if valid:
        valid = json.loads(calls[0]["function"]["arguments"]) == {
            "build_id": "build-17"
        }
    result = dict(tool_arguments=valid, first_response=first_data)
    if valid:
        messages += [
            message,
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": json.dumps(
                    {"build_id": "build-17", "status": "passed", "failed_tests": 0}
                ),
            },
            {
                "role": "user",
                "content": "Report the build ID, status and number of failed tests in one short sentence.",
            },
        ]
        second = session.post(
            args.url + "/v1/chat/completions",
            json=dict(settings, messages=messages, tools=[tool], tool_choice="none"),
            timeout=180,
        )
        second.raise_for_status()
        result["second_response"] = second.json()
        answer = result["second_response"]["choices"][0]["message"]["content"].lower()
        result["tool_result"] = (
            "build-17" in answer
            and "passed" in answer
            and ("0" in answer or "zero" in answer)
        )
    Path(args.output).write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if not k.endswith("response")})
    if not result.get("tool_arguments") or not result.get("tool_result"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
