"""Key-split QSA indexer prefill scorer: bitwise parity with one CTA per row block.

Run on a V100: python test/manual/layers/test_qsa_mqa_prefill_key_split.py [--bench]
"""

import argparse
import sys

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.mqa import tilelang_qsa_mqa_prefill


def _case(rows, keys, prefix, seed=0, heads=4, head_dim=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, heads, head_dim, device="cuda", generator=g).half()
    k = torch.randn(keys, 1, head_dim, device="cuda", generator=g).half()
    # Causal compressed ranges like chunked prefill: row i sees keys [0, (prefix+i)//4).
    ends = ((torch.arange(rows, device="cuda") + prefix + 1) // 4).clamp(max=keys)
    starts = torch.zeros_like(ends)
    return q, k, starts.int(), ends.int()


def _run(split, *args):
    with envs.SGLANG_ENABLE_QSA_MQA_KEY_SPLIT.override(split):
        return tilelang_qsa_mqa_prefill(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    cases = [(37, 50, 200), (2048, 512, 2048), (800, 40000, 160000), (512, 62500, 250000),
             (2048, 6000, 24000), (1024, 32768, 131072)]
    ok = True
    for rows, keys, prefix in cases:
        inputs = _case(rows, keys, prefix)
        ref, new = _run(False, *inputs), _run(True, *inputs)
        same = torch.equal(ref.view(torch.int32), new.view(torch.int32))
        ok &= same
        line = f"rows={rows} keys={keys} bitwise={same}"
        if args.bench:
            times = {}
            for split in (False, True):
                for _ in range(3):
                    _run(split, *inputs)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(10):
                    _run(split, *inputs)
                end.record()
                torch.cuda.synchronize()
                times[split] = start.elapsed_time(end) / 10
            line += f" base={times[False]:.3f}ms split={times[True]:.3f}ms x{times[False] / times[True]:.2f}"
        print(line, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
