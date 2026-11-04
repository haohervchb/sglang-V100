"""Render standalone figures from a completed Qwen3.8 core audit."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = next(
        value
        for name, value in summary["traces"].items()
        if "70000" in name and "TP-0." in name
    )
    prefill = trace["stages"]["prefill"]["semantic_kernel_sum_ms"]
    total = sum(prefill.values())
    labels = [
        "MoE",
        "QSA + projections",
        "Hyperconnections",
        "GDN + projections",
        "Collectives",
        "PLE",
        "Other",
    ]
    keys = [
        "moe",
        "qsa_including_projections",
        "hyperconnections",
        "gdn_including_projections",
        "collectives",
        "ple",
    ]
    values = [100 * prefill[k] / total for k in keys]
    values.append(100 - sum(values))
    cycle = summary["cycles"]["trace_cpu_prose_70000"]["median_ms"]
    phase_labels = [
        "Target verification",
        "MTP drafting",
        "Draft extension",
        "Verify / state / other",
    ]
    phase_values = [
        cycle["target"],
        cycle["draft"],
        cycle["draft_extend"],
        cycle["cycle_gpu_ms"]
        - cycle["target"]
        - cycle["draft"]
        - cycle["draft_extend"],
    ]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.55})
    colors = [
        "#245ea8",
        "#2d8883",
        "#d07b27",
        "#725697",
        "#899644",
        "#b86777",
        "#969696",
    ]
    for ax, names, numbers, palette, title, xlabel in [
        (
            axes[0],
            labels,
            values,
            colors,
            "Prefill: GPU kernel work",
            "Share of summed kernel duration (%)",
        ),
        (
            axes[1],
            phase_labels,
            phase_values,
            ["#245ea8", "#2d8883", "#d07b27", "#969696"],
            "Decode: verification cycle",
            "CUDA-event duration (ms)",
        ),
    ]:
        ax.barh(names, numbers, color=palette)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(numbers) * 1.22)
        ax.tick_params(axis="y", length=0)
        for index, value in enumerate(numbers):
            ax.text(value + max(numbers) * 0.02, index, f"{value:.1f}", va="center")
    fig.suptitle(
        "Qwen3.8 Flash Next on 4× V100 — 70K context", fontsize=15, weight="bold"
    )
    fig.text(
        0.02,
        0.025,
        "FP16 KV · MTP-3/4 · 2,048-token prefill chunks. Traced kernel shares are diagnostic; streams can overlap.\n"
        "Decode phases use the lighter CPU-profiler/event pass. Normal serving measured about 22 ms per cycle.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.22, top=0.84, left=0.18, right=0.97)
    for suffix in ["svg", "png"]:
        fig.savefig(args.output_dir / f"core_breakdown.{suffix}", dpi=170)
    plt.close(fig)

    fields = ["1001", "1002", "1003", "1004", "1005"]
    names = [
        "GPU engine active",
        "SM active",
        "SM occupancy",
        "Tensor pipe active",
        "DRAM active",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, phase, color in [
        (-0.18, "prefill", "#245ea8"),
        (0.18, "decode", "#d07b27"),
    ]:
        values = []
        for field in fields:
            samples = [
                r["mean"][field]
                for r in summary["dcgm"]
                if r["case_id"] == "prose_70000"
                and r["phase"] == phase
                and field in r["mean"]
            ]
            values.append(100 * statistics.mean(samples))
        positions = [i + offset for i in range(len(fields))]
        ax.barh(positions, values, height=0.32, label=phase.capitalize(), color=color)
        for position, value in zip(positions, values):
            ax.text(value + 1, position, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(fields)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Time-averaged activity (%)")
    ax.set_title(
        "Busy GPUs do not imply full arithmetic throughput",
        loc="left",
        weight="bold",
        fontsize=14,
    )
    ax.legend(loc="lower right", frameon=False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.02,
        0.025,
        "DCGM measurements averaged across four V100s at 70K context. SM and Tensor metrics use separate passes.\n"
        "Activity and occupancy are not percentages of theoretical FLOP/s or achieved DRAM bandwidth.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.23, bottom=0.2, top=0.87, right=0.96)
    for suffix in ["svg", "png"]:
        fig.savefig(args.output_dir / f"hardware_activity.{suffix}", dpi=170)
    plt.close(fig)
    for name in ["core_breakdown.svg", "hardware_activity.svg"]:
        path = args.output_dir / name
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )


if __name__ == "__main__":
    main()
