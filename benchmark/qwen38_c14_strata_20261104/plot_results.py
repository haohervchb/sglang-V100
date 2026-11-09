"""Plot whole-request throughput; do not sum per-stream decode rates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
runs = json.loads((root / "summary.json").read_text())["runs"]
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "qwen38-c14"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
variants = [
    ("baseline", "v5.5 MTP-3/4", "#667085"),
    ("compact", "New flags, MTP-3/4", "#00897b"),
    ("wide8", "New flags, MTP-7/8", "#db7a27"),
]
for ax, workload, title in zip(
    axes,
    ("code", "prose"),
    ("Copy/edit: 8K input, 512 output", "Prose: 1K input, 1,024 output"),
):
    for index, (variant, label, color) in enumerate(variants):
        values = [
            runs[f"{variant}_{workload}"][str(c)]["mean_aggregate_e2e_tok_s"]
            for c in (1, 2, 4)
        ]
        bars = ax.bar(
            np.arange(3) + (index - 1) * 0.25, values, 0.24, label=label, color=color
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=3)
    ax.set_xticks(range(3), ("C1", "C2", "C4"))
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(0, 290)
axes[0].set_ylabel("Aggregate output tokens / second (includes prefill)")
fig.suptitle("Wider MTP helps copy-heavy code, hurts prose", fontsize=14, weight="bold")
fig.legend(
    *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False
)
fig.text(
    0.5,
    0.105,
    "4× V100 SXM2 32 GB · same Flash-Next NVFP4 checkpoint · 8 requests, 2 batches per point",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.15, 1, 0.94))
destination = root / "throughput.svg"
fig.savefig(destination, metadata={"Date": None})
destination.write_text(
    "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
)
