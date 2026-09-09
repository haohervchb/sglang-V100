"""Validate image/video grounding against a running Qwen3.8 server.

Uses committed synthetic fixtures with explicit answers, including swapped
images and reversed videos to detect ignored media or stale feature caches.
"""

import argparse
import base64
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import requests


def media(path, kind):
    mime = "image/png" if kind == "image" else "video/mp4"
    url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return {"type": f"{kind}_url", f"{kind}_url": {"url": url}}


def message(text, *items):
    return {
        "role": "user",
        "content": [*items, {"type": "text", "text": text}],
    }


def compact(payload):
    result = copy.deepcopy(payload)
    for msg in result["messages"]:
        if not isinstance(msg["content"], list):
            continue
        for item in msg["content"]:
            for key in ("image_url", "video_url"):
                if key in item:
                    data = item[key]["url"].split(",", 1)[1]
                    item[key]["url"] = {
                        "decoded_sha256": hashlib.sha256(
                            base64.b64decode(data)
                        ).hexdigest()
                    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="Run only these named cases")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.tag}_results.json"
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite {destination}")
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    images = {
        n: media(fixture_dir / f"{n}.png", "image")
        for n in ("scene_a", "scene_b", "ocr", "chart")
    }
    videos = {
        n: media(fixture_dir / f"{n}.mp4", "video") for n in ("forward", "reverse")
    }
    server_info = requests.get(args.base_url + "/get_server_info", timeout=10)
    server_info.raise_for_status()
    (args.output_dir / f"{args.tag}_server_info.json").write_text(
        json.dumps(server_info.json()) + "\n"
    )
    report = {
        "tag": args.tag,
        "fixtures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(fixture_dir.iterdir())
        },
        "cases": [],
    }

    def run(name, messages, expected, *, stream=False, flush=True, min_prompt=0):
        if args.cases and name not in args.cases:
            return ""
        payload = {
            "model": "qwen",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        row = {
            "name": name,
            "request": compact(payload),
            "expected": expected,
            "flushed_cache": flush,
        }
        started = time.perf_counter()
        try:
            if flush:
                requests.post(
                    args.base_url + "/flush_cache",
                    params={"timeout": 30},
                    timeout=60,
                ).raise_for_status()
            started = time.perf_counter()
            response = requests.post(
                args.base_url + "/v1/chat/completions",
                json=payload,
                timeout=600,
                stream=stream,
            )
            row["status_code"] = response.status_code
            if response.status_code != 200:
                row["response"] = response.text
            response.raise_for_status()
            if stream:
                events, pieces, usage, finish = [], [], None, None
                done = False
                for line in response.iter_lines():
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:]
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    events.append(
                        {"elapsed_s": time.perf_counter() - started, "data": event}
                    )
                    assert "error" not in event, event
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        pieces.append(choice.get("delta", {}).get("content") or "")
                        finish = choice.get("finish_reason") or finish
                text = "".join(pieces)
                row["events"] = events
                assert done, "Missing SSE completion marker"
                assert len(events) > 1, "Expected multiple streaming events"
            else:
                body = response.json()
                row["response"] = body
                choice = body["choices"][0]
                text = choice["message"]["content"] or ""
                usage = body["usage"]
                finish = choice["finish_reason"]
            row.update(text=text, usage=usage, finish_reason=finish)
            assert finish == "stop", f"Unexpected finish: {finish}"
            assert usage and usage["completion_tokens"] > 0, usage
            assert usage["prompt_tokens"] > min_prompt, usage
            normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
            if isinstance(expected, list):
                assert all(word in normalized.split() for word in expected), text
            else:
                assert normalized == expected, text
            row["passed"] = True
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        row["elapsed_s"] = time.perf_counter() - started
        report["cases"].append(row)
        report["passed"] = all(r["passed"] for r in report["cases"])
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("name", "passed", "text", "error", "elapsed_s")
                    if k in row
                }
            ),
            flush=True,
        )
        return row.get("text", "")

    circle = "What color is the circle? Reply with only the color name."
    first = message(circle, images["scene_a"])
    answer = run("image_color", [first], "red")
    run(
        "image_swap_without_flush",
        [message(circle, images["scene_b"])],
        "blue",
        flush=False,
    )
    run("image_repeat_without_flush", [first], "red", flush=False)
    run(
        "image_followup",
        [
            first,
            {"role": "assistant", "content": answer},
            message("What shape is on the right? Reply with only the shape name."),
        ],
        "square",
        flush=False,
    )
    run(
        "ocr",
        [message("Transcribe the text exactly. Output only the text.", images["ocr"])],
        "v100 check 4827",
    )
    run(
        "chart",
        [
            message(
                "Which bar is tallest and what is its value? "
                "Reply with only the label and number.",
                images["chart"],
            )
        ],
        "beta 7",
    )
    run(
        "multiple_images",
        [
            message(
                "Give the circle colors in image order, "
                "separated by a comma. Output only the two color names.",
                images["scene_a"],
                images["scene_b"],
            )
        ],
        "red blue",
    )
    run(
        "image_streaming",
        [
            message(
                "Describe both shapes, their colors and "
                "their left/right positions in one sentence.",
                images["scene_a"],
            )
        ],
        ["red", "circle", "left", "blue", "square", "right"],
        stream=True,
    )
    filler = "This is unrelated background context. " * 1600
    run(
        "image_chunked_prefill",
        [message(filler + "\nNow inspect the image. " + circle, images["scene_b"])],
        "blue",
        min_prompt=8192,
    )
    video_prompt = (
        "List the three background colors in chronological order, "
        "separated by commas. Output only the three color names."
    )
    run("video_order", [message(video_prompt, videos["forward"])], "red green blue")
    run(
        "video_reversed_without_flush",
        [message(video_prompt, videos["reverse"])],
        "blue green red",
        flush=False,
    )
    run(
        "text_after_multimodal",
        [message("What is 19 + 23? Output only the number.")],
        "42",
        flush=False,
    )
    health = requests.get(args.base_url + "/health", timeout=10)
    report["health_status"] = health.status_code
    expected_count = len(args.cases) if args.cases else 12
    report["passed"] = (
        report["passed"] and health.ok and len(report["cases"]) == expected_count
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
