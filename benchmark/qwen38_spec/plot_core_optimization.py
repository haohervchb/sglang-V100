"""Render the measured latency changes without mixing them with microbenchmarks."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    order = [
        "prose_1000",
        "prose_8192",
        "prose_25000",
        "prose_70000",
        "prose_160000",
        "code_edit_8192",
    ]
    pairs = {r["case"]: r for r in data["fixed_output"]}
    names = [
        "Prose 1K",
        "Prose 8K",
        "Prose 25K",
        "Prose 70K",
        "Prose 160K",
        "Code edit 8K",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, field, title in zip(
        axes, ["ttft_s", "end_to_end_s"], ["Time to first token", "Complete request"]
    ):
        values = [
            -pairs[name]["metrics"][field]["relative_change_pct"] for name in order
        ]
        ax.barh(
            names, values, color=["#287a64" if x >= 0 else "#c57736" for x in values]
        )
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_xlim(-3.5, 14)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Latency reduction (%)")
        for i, value in enumerate(values):
            ax.text(
                value + (0.25 if value >= 0 else -0.25),
                i,
                f"{value:.1f}%",
                va="center",
                ha="left" if value >= 0 else "right",
            )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Qwen3.8 Flash Next · 4× V100 · fixed 512-token outputs",
        weight="bold",
        fontsize=14,
    )
    fig.text(
        0.02,
        0.025,
        "Two samples per configuration/case; cold prefix cache, warm kernels. Positive bars mean faster.\n"
        "These are end-to-end request measurements, not the larger isolated kernel speedups.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.22, wspace=0.18)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "png"]:
        fig.savefig(args.output_dir / f"latency_changes.{suffix}", dpi=170)
    path = args.output_dir / "latency_changes.svg"
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
