#!/usr/bin/env python3
"""Audit reasoning, tool-call, and multimodal behavior on a live SGLang server."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests


ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

SCENE_TOOL = {
    "type": "function",
    "function": {
        "name": "report_scene",
        "description": "Report the important visible facts in an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_activity": {"type": "string"},
                "vehicle_color": {"type": "string"},
                "vehicle_type": {"type": "string"},
                "setting": {"type": "string"},
            },
            "required": [
                "person_activity",
                "vehicle_color",
                "vehicle_type",
                "setting",
            ],
            "additionalProperties": False,
        },
    },
}


def max_run(values: list[str]) -> int:
    best = current = 0
    previous = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def audit_text(label: str, text: str, *, minimum_chars: int = 1) -> dict[str, Any]:
    if not isinstance(text, str):
        raise AssertionError(f"{label}: expected text, got {type(text).__name__}")
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    attached_nines = re.findall(r"[A-Za-z]9(?:\b|(?=[^0-9]))", text)
    report = {
        "chars": len(text),
        "words": len(words),
        "unique_words": len(set(words)),
        "max_character_run": max_run(list(text)),
        "max_word_run": max_run(words),
        "attached_nine_count": len(attached_nines),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    assert len(text) >= minimum_chars, f"{label}: output is too short: {text!r}"
    assert "\ufffd" not in text, f"{label}: replacement character found"
    assert report["max_character_run"] < 32, f"{label}: repeated character run"
    assert report["max_word_run"] < 8, f"{label}: repeated word run"
    assert report["attached_nine_count"] < 5, f"{label}: stray-9 pattern returned"
    if len(words) >= 20:
        assert len(set(words)) / len(words) > 0.15, f"{label}: low word diversity"
    return report


def compact_request(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    for message in value.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            if url.startswith("data:"):
                image_url["url"] = (
                    f"<data-url bytes={len(url)} "
                    f"sha256={hashlib.sha256(url.encode()).hexdigest()}>"
                )
    return value


class Audit:
    def __init__(
        self,
        base_url: str,
        model: str,
        image_path: Path,
        output_path: Path,
        multimodal_mode: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.output_path = output_path
        self.multimodal_mode = multimodal_mode
        image_bytes = image_path.read_bytes()
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        self.image_url = (
            f"data:{mime};base64," + base64.b64encode(image_bytes).decode()
        )
        self.image_info = {
            "path": str(image_path),
            "bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
        self.cases: list[dict[str, Any]] = []

    def post_chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=900,
        )
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except requests.JSONDecodeError:
            body = {"raw_text": response.text}
        if response.status_code != 200:
            raise AssertionError(
                f"HTTP {response.status_code}: {json.dumps(body)[:2000]}"
            )
        return body, elapsed

    def expect_multimodal_unsupported(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=900,
        )
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except requests.JSONDecodeError:
            body = {"raw_text": response.text}
        assert 400 <= response.status_code < 500, (
            "text-only model did not reject image input: "
            f"HTTP {response.status_code}: {json.dumps(body)[:2000]}"
        )
        error_text = json.dumps(body).lower()
        assert any(
            keyword in error_text
            for keyword in ("image", "multimodal", "vision", "media")
        ), f"image rejection was not explicit: {body}"
        return {
            "request": compact_request(payload),
            "response_status": response.status_code,
            "response": body,
            "http_elapsed_s": elapsed,
            "expected_capability": "text-only",
        }

    def run_case(self, name: str, function: Callable[[], dict[str, Any]]) -> None:
        started = time.perf_counter()
        try:
            detail = function()
            self.cases.append(
                {
                    "name": name,
                    "passed": True,
                    "elapsed_s": time.perf_counter() - started,
                    **detail,
                }
            )
            print(json.dumps({"case": name, "passed": True}), flush=True)
        except Exception as exc:
            self.cases.append(
                {
                    "name": name,
                    "passed": False,
                    "elapsed_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                json.dumps({"case": name, "passed": False, "error": str(exc)}),
                flush=True,
            )

    def reasoning_enabled(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A tray has 7 rows of 6 bolts. Five bolts are removed. "
                        "Explain the arithmetic briefly and give the final number."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        }
        body, elapsed = self.post_chat(payload)
        message = body["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        reasoning_audit = audit_text("reasoning_enabled.reasoning", reasoning, minimum_chars=20)
        content_audit = audit_text("reasoning_enabled.content", content, minimum_chars=9)
        assert "37" in content, f"missing correct answer: {content!r}"
        assert "<think>" not in content and "</think>" not in content
        usage = body.get("usage") or {}
        assert usage.get("reasoning_tokens", 0) > 0, f"bad reasoning usage: {usage}"
        return {
            "request": compact_request(payload),
            "response": body,
            "http_elapsed_s": elapsed,
            "reasoning_audit": reasoning_audit,
            "content_audit": content_audit,
        }

    def reasoning_disabled(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 18 + 24? Answer with only the integer.",
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "separate_reasoning": True,
        }
        body, elapsed = self.post_chat(payload)
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        assert re.fullmatch(r"\s*42\s*", content), f"unexpected answer: {content!r}"
        assert not (message.get("reasoning_content") or "")
        usage = body.get("usage") or {}
        assert usage.get("reasoning_tokens", 0) == 0, f"bad reasoning usage: {usage}"
        return {
            "request": compact_request(payload),
            "response": body,
            "http_elapsed_s": elapsed,
            "content_audit": audit_text("reasoning_disabled.content", content),
        }

    def tool_roundtrip(self) -> dict[str, Any]:
        payload1 = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use get_weather to obtain the current weather in "
                        "Brisbane. Do not guess or answer before calling it."
                    ),
                }
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        }
        body1, elapsed1 = self.post_chat(payload1)
        message1 = body1["choices"][0]["message"]
        calls = message1.get("tool_calls") or []
        assert len(calls) == 1, f"expected one tool call: {message1}"
        call = calls[0]
        assert call["function"]["name"] == "get_weather"
        arguments = json.loads(call["function"]["arguments"])
        assert arguments.get("city", "").lower() == "brisbane", arguments
        assert body1["choices"][0]["finish_reason"] == "tool_calls"
        if message1.get("reasoning_content"):
            audit_text(
                "tool_roundtrip.call_reasoning",
                message1["reasoning_content"],
                minimum_chars=5,
            )

        messages = list(payload1["messages"])
        messages.append(message1)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": "get_weather",
                "content": json.dumps(
                    {
                        "city": "Brisbane",
                        "temperature_c": 23,
                        "condition": "sunny",
                    }
                ),
            }
        )
        payload2 = {
            "model": self.model,
            "messages": messages,
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        }
        body2, elapsed2 = self.post_chat(payload2)
        message2 = body2["choices"][0]["message"]
        content = message2.get("content") or ""
        assert not message2.get("tool_calls"), f"unexpected repeated call: {message2}"
        lowered = content.lower()
        assert "23" in content and "sunny" in lowered and "brisbane" in lowered, content
        assert "<think>" not in content and "</think>" not in content
        content_audit = audit_text("tool_roundtrip.final", content, minimum_chars=20)
        if message2.get("reasoning_content"):
            audit_text(
                "tool_roundtrip.final_reasoning",
                message2["reasoning_content"],
                minimum_chars=5,
            )
        return {
            "requests": [compact_request(payload1), compact_request(payload2)],
            "responses": [body1, body2],
            "http_elapsed_s": [elapsed1, elapsed2],
            "parsed_arguments": arguments,
            "content_audit": content_audit,
        }

    def streaming_tool(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the add tool to calculate 19 + 23.",
                }
            ],
            "tools": [ADD_TOOL],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=900,
        )
        assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
        events: list[dict[str, Any]] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reasons: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            event = json.loads(data)
            events.append(event)
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])
                for call in (choice.get("delta") or {}).get("tool_calls") or []:
                    index = int(call.get("index", 0))
                    current = calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    current["id"] += call.get("id") or ""
                    function = call.get("function") or {}
                    current["name"] += function.get("name") or ""
                    current["arguments"] += function.get("arguments") or ""
        elapsed = time.perf_counter() - started
        assert len(calls) == 1, calls
        call = calls[0]
        assert call["name"] == "add", call
        arguments = json.loads(call["arguments"])
        assert arguments == {"a": 19, "b": 23}, arguments
        assert "tool_calls" in finish_reasons, finish_reasons
        return {
            "request": compact_request(payload),
            "events": events,
            "http_elapsed_s": elapsed,
            "assembled_tool_call": call,
            "parsed_arguments": arguments,
            "finish_reasons": finish_reasons,
        }

    def multimodal_description(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {
                            "type": "text",
                            "text": (
                                "Describe the central unusual activity and the "
                                "vehicle in one concise sentence."
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 192,
            "chat_template_kwargs": {"enable_thinking": False},
            "separate_reasoning": True,
        }
        if self.multimodal_mode == "unsupported":
            return self.expect_multimodal_unsupported(payload)
        body, elapsed = self.post_chat(payload)
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        lowered = content.lower()
        assert any(word in lowered for word in ("man", "person")), content
        assert "iron" in lowered, content
        assert "yellow" in lowered, content
        assert any(word in lowered for word in ("taxi", "cab", "vehicle", "suv")), content
        assert not (message.get("reasoning_content") or "")
        return {
            "request": compact_request(payload),
            "response": body,
            "http_elapsed_s": elapsed,
            "content_audit": audit_text(
                "multimodal_description.content", content, minimum_chars=30
            ),
        }

    def multimodal_reasoning(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {
                            "type": "text",
                            "text": (
                                "Decide whether the scene is ordinary or unusual. "
                                "Reason from visible evidence, then explain the "
                                "conclusion concisely."
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        }
        if self.multimodal_mode == "unsupported":
            return self.expect_multimodal_unsupported(payload)
        body, elapsed = self.post_chat(payload)
        message = body["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        lowered = (reasoning + "\n" + content).lower()
        assert "unusual" in content.lower(), content
        assert "iron" in lowered, content
        assert any(word in lowered for word in ("taxi", "cab", "vehicle", "suv")), content
        usage = body.get("usage") or {}
        assert usage.get("reasoning_tokens", 0) > 0, usage
        return {
            "request": compact_request(payload),
            "response": body,
            "http_elapsed_s": elapsed,
            "reasoning_audit": audit_text(
                "multimodal_reasoning.reasoning", reasoning, minimum_chars=20
            ),
            "content_audit": audit_text(
                "multimodal_reasoning.content", content, minimum_chars=20
            ),
        }

    def multimodal_tool(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {
                            "type": "text",
                            "text": (
                                "Inspect the image and call report_scene with the "
                                "visible activity, vehicle color/type, and setting. "
                                "Do not answer in prose."
                            ),
                        },
                    ],
                }
            ],
            "tools": [SCENE_TOOL],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.multimodal_mode == "unsupported":
            return self.expect_multimodal_unsupported(payload)
        body, elapsed = self.post_chat(payload)
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        assert len(calls) == 1, message
        call = calls[0]
        assert call["function"]["name"] == "report_scene", call
        arguments = json.loads(call["function"]["arguments"])
        assert set(arguments) == {
            "person_activity",
            "vehicle_color",
            "vehicle_type",
            "setting",
        }, arguments
        joined = " ".join(str(value) for value in arguments.values()).lower()
        assert "iron" in joined, arguments
        assert "yellow" in joined, arguments
        assert any(word in joined for word in ("taxi", "cab", "vehicle", "suv")), arguments
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        return {
            "request": compact_request(payload),
            "response": body,
            "http_elapsed_s": elapsed,
            "parsed_arguments": arguments,
        }

    def run(self) -> bool:
        server_info_response = requests.get(f"{self.base_url}/server_info", timeout=30)
        server_info_response.raise_for_status()
        server_info = server_info_response.json()

        self.run_case("reasoning_enabled", self.reasoning_enabled)
        self.run_case("reasoning_disabled", self.reasoning_disabled)
        self.run_case("tool_roundtrip", self.tool_roundtrip)
        self.run_case("streaming_tool", self.streaming_tool)
        self.run_case("multimodal_description", self.multimodal_description)
        self.run_case("multimodal_reasoning", self.multimodal_reasoning)
        self.run_case("multimodal_tool", self.multimodal_tool)

        passed = sum(case["passed"] for case in self.cases)
        result = {
            "model": self.model,
            "base_url": self.base_url,
            "multimodal_mode": self.multimodal_mode,
            "image": self.image_info,
            "server_info": server_info,
            "summary": {
                "passed": passed == len(self.cases),
                "passed_cases": passed,
                "total_cases": len(self.cases),
            },
            "cases": self.cases,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summary"]), flush=True)
        return bool(result["summary"]["passed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("examples/assets/example_image.png"),
    )
    parser.add_argument(
        "--multimodal-mode",
        choices=("image", "unsupported"),
        default="image",
        help=(
            "Use 'image' for vision-capable models or 'unsupported' to require "
            "an explicit 4xx rejection from a text-only model."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = Audit(
        args.base_url,
        args.model,
        args.image,
        args.output,
        args.multimodal_mode,
    )
    if not audit.run():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
