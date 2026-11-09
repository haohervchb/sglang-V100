"""Expand a Flash-Next draft map by vocabulary rules, without benchmark fitting."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
CJK = re.compile(
    "[\u1100-\u11ff\u3000-\u318f\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef\U00020000-\U0003134f]"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = set(int(i) for i in torch.load(args.input, weights_only=True))
    if not base or min(base) < 0 or max(base) >= 248320:
        raise ValueError("Base map contains invalid Flash-Next token IDs")
    cjk = set()
    for start in range(0, 248320, 2048):
        texts = tokenizer.batch_decode(
            [[i] for i in range(start, min(248320, start + 2048))],
            skip_special_tokens=False,
        )
        cjk.update(start + i for i, text in enumerate(texts) if CJK.search(text))
    selected = base | cjk
    width = ((len(selected) + 255) // 256) * 256
    # Fill to TP4's 64-row alignment without truncating a required token.
    for token in range(248320):
        if len(selected) == width:
            break
        selected.add(token)
    ids = sorted(selected)
    torch.save(ids, args.output)
    manifest = dict(
        model=MODEL,
        revision=REVISION,
        base_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        base_tokens=len(base),
        cjk_tokens=len(cjk),
        cjk_tokens_missing_from_base=len(cjk - base),
        output_tokens=len(ids),
        token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        file_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        rule=CJK.pattern,
        note="Vocabulary-entry coverage, not frequency-weighted language coverage. No evaluation prompts or outputs were used.",
    )
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
