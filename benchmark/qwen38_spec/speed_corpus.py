"""Pin a small SPEED-Bench screen; no claim to the full benchmark score."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REVISION = "fbd6dc64353a3153bf24252a113ebd6d2cff9dec"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    path = hf_hub_download(
        "nvidia/SPEED-Bench",
        "qualitative/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )
    data = pq.read_table(path).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    )
    selected = []
    for category in ["coding", "reasoning", "writing", "rag"]:
        # Select by stable ID before observing model speed/acceptance.
        selected.extend(
            sorted(
                [row for row in data if row["category"] == category],
                key=lambda row: row["question_id"],
            )[:4]
        )
    for subcategory in ["ja_math", "ko_math", "ru-zh", "ar-zh"]:
        selected.append(
            sorted(
                [
                    row
                    for row in data
                    if row["category"] == "multilingual"
                    and row["sub_category"] == subcategory
                ],
                key=lambda row: row["question_id"],
            )[0]
        )
    cases, manifest = [], []
    for row in selected:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["turns"][0]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        identity = dict(
            id="speed_" + row["question_id"],
            kind=row["category"],
            sub_category=row["sub_category"],
            input_sha256=digest,
            input_tokens=len(ids),
        )
        cases.append(dict(identity, input_ids=ids))
        manifest.append(identity)
    meta = dict(
        dataset="nvidia/SPEED-Bench",
        revision=REVISION,
        split="qualitative",
        turn=0,
        enable_thinking=False,
    )
    Path(args.output).write_text(json.dumps(dict(meta, cases=cases)))
    Path(args.manifest).write_text(json.dumps(dict(meta, cases=manifest), indent=2))
    print(
        f"Saved {len(cases)} first-turn cases, {min(len(c['input_ids']) for c in cases)}–{max(len(c['input_ids']) for c in cases)} tokens"
    )


if __name__ == "__main__":
    main()
