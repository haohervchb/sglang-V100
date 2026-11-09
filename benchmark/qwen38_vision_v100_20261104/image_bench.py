"""Cold-cache end-to-end image benchmark; run against an idle test server."""

import argparse, base64, hashlib, io, json, time
from pathlib import Path
import requests
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8082")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--pages", type=int, default=32)
p.add_argument("--width", type=int, default=768)
p.add_argument("--height", type=int, default=1024)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--label", required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
(a.output / "server.json").write_text(
    json.dumps(requests.get(a.url + "/get_server_info", timeout=10).json(), indent=2)
    + "\n"
)
for repeat in range(a.repeats):
    images = []
    hashes = []
    for i in range(a.pages):
        color = (
            (0, 0, 220)
            if i == 0
            else ((220, 0, 0) if i == a.pages - 1 else (20 + i, 150, 50 + i))
        )
        im = Image.new("RGB", (a.width, a.height), color)
        ImageDraw.Draw(im).text(
            (50, 50), f"ROUND {repeat} PAGE {i + 1}", fill="white", font_size=48
        )
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=90)
        raw = b.getvalue()
        hashes.append(hashlib.sha256(raw).hexdigest())
        images.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()
                },
            }
        )
    images.append(
        {
            "type": "text",
            "text": "What are the background colors of the FIRST and LAST images, in that order? Output only the two color names.",
        }
    )
    f = requests.post(a.url + "/flush_cache", params={"timeout": 30}, timeout=120)
    f.raise_for_status()
    try:
        body = f.json()
    except ValueError:
        body = None
    if body is False or isinstance(body, dict) and body.get("success") is False:
        raise RuntimeError("Cache flush refused")
    events = []
    parts = []
    start = time.perf_counter()
    unix = time.time()
    with requests.post(
        a.url + "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": images}],
            "temperature": 0,
            "max_tokens": 32,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        stream=True,
        timeout=(30, 900),
    ) as response:
        response.raise_for_status()
        usage = None
        for line in response.iter_lines():
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            data = json.loads(line[6:])
            usage = data.get("usage") or usage
            if data.get("choices"):
                content = data["choices"][0].get("delta", {}).get("content") or ""
                if content:
                    parts.append(content)
                    events.append(time.perf_counter() - start)
    text = "".join(parts).strip()
    passed = (
        "blue" in text.lower()
        and "red" in text.lower()
        and text.lower().index("blue") < text.lower().index("red")
    )
    row = {
        "label": a.label,
        "repeat": repeat,
        "pages": a.pages,
        "dimensions": [a.width, a.height],
        "start_unix_s": unix,
        "image_sha256": hashes,
        "seconds": time.perf_counter() - start,
        "ttft_s": events[0] if events else None,
        "text": text,
        "passed": passed,
        "usage": usage,
    }
    with (a.output / "results.jsonl").open("a") as out:
        out.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "image_sha256"}), flush=True)
    if not passed:
        raise RuntimeError(row)
