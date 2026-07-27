#!/usr/bin/env python3
"""Render the audited three-model V100 comparison as a dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Callable


MODELS = [
    ("Qwen3.6 27B FP16 + DFlash", "27b", "#2563eb"),
    ("Qwen3.5 122B-A10B INT4 + DFlash", "122b", "#dc2626"),
    ("Laguna S 2.1 118B-A8B INT4", "laguna", "#059669"),
]
CATEGORIES = [
    (1, 1_000),
    (1, 25_000),
    (2, 1_000),
    (2, 25_000),
    (4, 1_000),
    (4, 25_000),
]
PANELS: list[tuple[str, str, Callable[[float], float], str]] = [
    (
        "Median per-request decode rate",
        "median_request_decode_tps_median",
        lambda value: value,
        "tok/s",
    ),
    (
        "Aggregate output throughput",
        "aggregate_output_tps_median",
        lambda value: value,
        "tok/s",
    ),
    (
        "Effective cold input rate",
        "effective_input_tps_median",
        lambda value: value,
        "input tok/s",
    ),
    (
        "Median client TTFT",
        "median_client_ttft_ms_median",
        lambda value: value / 1000,
        "seconds",
    ),
]


def nice_ceiling(value: float, tick_count: int = 5) -> tuple[float, float]:
    if value <= 0:
        return 1.0, 0.2
    rough_step = value / tick_count
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return math.ceil(value / step) * step, step


def format_value(value: float, unit: str) -> str:
    if unit == "input tok/s" and value >= 1000:
        return f"{value / 1000:.1f}k"
    if unit == "seconds":
        return f"{value:.1f}"
    return f"{value:.0f}"


def load_rows(results_dir: Path) -> dict[str, dict[tuple[int, int], dict]]:
    rows_by_model: dict[str, dict[tuple[int, int], dict]] = {}
    for _, model_key, _ in MODELS:
        rows = json.loads((results_dir / model_key / "summary.json").read_text())
        rows_by_model[model_key] = {
            (int(row["concurrency"]), int(row["prompt_len"])): row for row in rows
        }
    return rows_by_model


def render_plot(
    rows_by_model: dict[str, dict[tuple[int, int], dict]],
) -> str:
    width, height = 1600, 1080
    panel_width, panel_height = 690, 330
    panel_positions = [(80, 190), (830, 190), (80, 570), (830, 570)]
    plot_pad_left, plot_pad_right = 75, 20
    plot_pad_top, plot_pad_bottom = 48, 68
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        "<title>Three-model inference comparison on four V100 GPUs</title>",
        (
            "<desc>Audited cold-cache results for Qwen 27B and 122B with "
            "DFlash and target-only Laguna S 2.1.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,'
            'BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}'
            ".title{font-size:31px;font-weight:700}.subtitle{font-size:16px;"
            "fill:#4b5563}.panel-title{font-size:19px;font-weight:650}"
            ".axis{font-size:12px;fill:#4b5563}.value{font-size:10px;"
            "font-weight:600}.unit{font-size:12px;fill:#6b7280}"
            ".footer{font-size:13px;fill:#6b7280}</style>"
        ),
        (
            f'<text class="title" x="{width / 2}" y="48" text-anchor="middle">'
            "V100 inference comparison — short and long context</text>"
        ),
        (
            f'<text class="subtitle" x="{width / 2}" y="78" text-anchor="middle">'
            "4× V100-SXM2-32GB · TP4 · cold unique code prompts · greedy · "
            "up to 256 output tokens/request</text>"
        ),
    ]

    legend_y = 120
    legend_widths = [370, 440, 360]
    legend_x = (width - sum(legend_widths)) / 2
    for (label, _, color), item_width in zip(MODELS, legend_widths):
        parts.extend(
            [
                (
                    f'<rect x="{legend_x:.1f}" y="{legend_y - 12}" width="18" '
                    f'height="18" rx="2" fill="{color}"/>'
                ),
                (
                    f'<text x="{legend_x + 28:.1f}" y="{legend_y + 2}" '
                    f'font-size="14">{html.escape(label)}</text>'
                ),
            ]
        )
        legend_x += item_width

    for panel_index, (title, metric, transform, unit) in enumerate(PANELS):
        px, py = panel_positions[panel_index]
        plot_x = px + plot_pad_left
        plot_y = py + plot_pad_top
        plot_w = panel_width - plot_pad_left - plot_pad_right
        plot_h = panel_height - plot_pad_top - plot_pad_bottom
        values = [
            transform(float(rows_by_model[model_key][category][metric]))
            for _, model_key, _ in MODELS
            for category in CATEGORIES
        ]
        y_max, y_step = nice_ceiling(max(values) * 1.12)

        def sy(value: float) -> float:
            return plot_y + plot_h - value / y_max * plot_h

        parts.extend(
            [
                (
                    f'<rect x="{px}" y="{py}" width="{panel_width}" '
                    f'height="{panel_height}" rx="9" fill="#f9fafb" '
                    'stroke="#d1d5db"/>'
                ),
                (
                    f'<text class="panel-title" x="{px + 18}" y="{py + 29}">'
                    f"{html.escape(title)}</text>"
                ),
                (
                    f'<text class="unit" x="{plot_x}" y="{plot_y - 8}">'
                    f"{html.escape(unit)}</text>"
                ),
            ]
        )
        tick = 0.0
        while tick <= y_max + y_step / 10:
            y = sy(tick)
            parts.extend(
                [
                    (
                        f'<line x1="{plot_x}" y1="{y:.2f}" '
                        f'x2="{plot_x + plot_w}" y2="{y:.2f}" '
                        'stroke="#e5e7eb"/>'
                    ),
                    (
                        f'<text class="axis" x="{plot_x - 10}" y="{y + 4:.2f}" '
                        f'text-anchor="end">{format_value(tick, unit)}</text>'
                    ),
                ]
            )
            tick += y_step

        group_width = plot_w / len(CATEGORIES)
        bar_width = min(23.0, group_width / (len(MODELS) + 1))
        for category_index, category in enumerate(CATEGORIES):
            center = plot_x + group_width * (category_index + 0.5)
            group_start = center - bar_width * len(MODELS) / 2
            concurrency, prompt_len = category
            for model_index, (_, model_key, color) in enumerate(MODELS):
                value = transform(float(rows_by_model[model_key][category][metric]))
                x = group_start + model_index * bar_width
                y = sy(value)
                parts.extend(
                    [
                        (
                            f'<rect x="{x + 1:.2f}" y="{y:.2f}" '
                            f'width="{bar_width - 2:.2f}" '
                            f'height="{plot_y + plot_h - y:.2f}" '
                            f'rx="2" fill="{color}"/>'
                        ),
                        (
                            f'<text class="value" x="{x + bar_width / 2:.2f}" '
                            f'y="{y - 4:.2f}" text-anchor="middle">'
                            f"{format_value(value, unit)}</text>"
                        ),
                    ]
                )
            parts.append(
                (
                    f'<text class="axis" x="{center:.2f}" '
                    f'y="{plot_y + plot_h + 22}" text-anchor="middle">'
                    f"C{concurrency} · {prompt_len // 1000}K</text>"
                )
            )

        parts.append(
            (
                f'<text class="axis" x="{plot_x + plot_w / 2}" '
                f'y="{plot_y + plot_h + 48}" text-anchor="middle">'
                "concurrency · prompt tokens</text>"
            )
        )

    parts.extend(
        [
            (
                f'<text class="footer" x="{width / 2}" y="970" '
                'text-anchor="middle">Decode is median per-request '
                "client-visible rate; aggregate output includes prefill and "
                "all requests.</text>"
            ),
            (
                f'<text class="footer" x="{width / 2}" y="995" '
                'text-anchor="middle">Qwen models use DFlash block size 16; '
                "Laguna is target-only. One audited cold-cache trial per "
                "cell.</text>"
            ),
            (
                f'<text class="footer" x="{width / 2}" y="1020" '
                'text-anchor="middle">SGLang V100 fork · 2026-07-27</text>'
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "plots"
            / "v100_model_comparison.svg"
        ),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_plot(load_rows(args.results_dir)))
    print(args.output)


if __name__ == "__main__":
    main()
