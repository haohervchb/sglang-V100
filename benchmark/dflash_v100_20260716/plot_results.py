#!/usr/bin/env python3
"""Render dependency-free SVG plots for the audited V100 context benchmark."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Callable


MODELS = [
    ("27B FP16", "27b", "#2563eb"),
    ("122B-A10B GPTQ-Int4", "122b", "#dc2626"),
    ("Laguna S 2.1 INT4 (target-only)", "laguna", "#059669"),
]
PANELS: list[tuple[str, str, Callable[[float], float], str]] = [
    (
        "Median per-request decode rate",
        "median_request_decode_tps_median",
        lambda value: value,
        "tok/s",
    ),
    (
        "DFlash acceptance length",
        "weighted_accept_length_median",
        lambda value: value,
        "tokens / verify",
    ),
    (
        "Median client TTFT",
        "median_client_ttft_ms_median",
        lambda value: value / 1000,
        "seconds",
    ),
    (
        "Effective cold input rate",
        "effective_input_tps_median",
        lambda value: value,
        "input tok/s",
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
    ceiling = math.ceil(value / step) * step
    return ceiling, step


def format_tick(value: float, unit: str) -> str:
    if unit == "input tok/s" and value >= 1000:
        return f"{value / 1000:.0f}k"
    if unit == "seconds":
        return f"{value:.0f}" if value >= 10 else f"{value:.1f}"
    if unit == "tokens / verify":
        return f"{value:.1f}"
    return f"{value:.0f}"


def load_rows(results_dir: Path) -> dict[str, list[dict]]:
    return {
        model_key: json.loads((results_dir / model_key / "summary.json").read_text())
        for _, model_key, _ in MODELS
    }


def render_plot(rows_by_model: dict[str, list[dict]], concurrency: int) -> str:
    width, height = 1400, 930
    panel_width, panel_height = 590, 300
    panel_positions = [(90, 150), (745, 150), (90, 520), (745, 520)]
    plot_pad_left, plot_pad_right = 75, 25
    plot_pad_top, plot_pad_bottom = 45, 55
    x_min, x_max = 1000, 25000
    x_ticks = [1000, 5000, 9000, 13000, 17000, 21000, 25000]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>V100 context benchmark at concurrency "
        + str(concurrency)
        + "</title>",
        "<desc>Audited V100 context-scaling results for Qwen3.6 27B FP16 with DFlash, Qwen3.5 122B-A10B GPTQ Int4 with DFlash, and Laguna S 2.1 INT4 target-only.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}.title{font-size:28px;font-weight:700}.subtitle{font-size:15px;fill:#4b5563}.panel-title{font-size:18px;font-weight:650}.axis{font-size:12px;fill:#4b5563}.unit{font-size:12px;fill:#6b7280}.footer{font-size:12px;fill:#6b7280}</style>',
        f'<text class="title" x="{width / 2}" y="48" text-anchor="middle">V100 context scaling — concurrency {concurrency}</text>',
        f'<text class="subtitle" x="{width / 2}" y="76" text-anchor="middle">Qwen DFlash + Laguna target-only · 4× V100-SXM2-32GB · cold unique code prompts · 256 output tokens/request</text>',
    ]

    legend_x = 205
    for index, (label, _, color) in enumerate(MODELS):
        x = legend_x + index * 390
        parts.extend(
            [
                f'<line x1="{x}" y1="108" x2="{x + 34}" y2="108" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{x + 17}" cy="108" r="4" fill="{color}"/>',
                f'<text x="{x + 44}" y="113" font-size="14">{html.escape(label)}</text>',
            ]
        )

    for panel_index, (title, metric, transform, unit) in enumerate(PANELS):
        px, py = panel_positions[panel_index]
        plot_x = px + plot_pad_left
        plot_y = py + plot_pad_top
        plot_w = panel_width - plot_pad_left - plot_pad_right
        plot_h = panel_height - plot_pad_top - plot_pad_bottom
        series: list[tuple[str, str, list[tuple[int, float]]]] = []
        values: list[float] = []
        for label, model_key, color in MODELS:
            points = [
                (int(row["prompt_len"]), transform(float(row[metric])))
                for row in rows_by_model[model_key]
                if int(row["concurrency"]) == concurrency
                and row.get(metric) is not None
            ]
            points.sort()
            series.append((label, color, points))
            values.extend(value for _, value in points)
        y_max, y_step = nice_ceiling(max(values) * 1.04)

        def sx(value: float) -> float:
            return plot_x + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return plot_y + plot_h - value / y_max * plot_h

        parts.extend(
            [
                f'<rect x="{px}" y="{py}" width="{panel_width}" height="{panel_height}" rx="8" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<text class="panel-title" x="{px + 18}" y="{py + 28}">{html.escape(title)}</text>',
                f'<text class="unit" x="{plot_x}" y="{plot_y - 7}">{html.escape(unit)}</text>',
            ]
        )
        tick = 0.0
        while tick <= y_max + y_step / 10:
            y = sy(tick)
            parts.extend(
                [
                    f'<line x1="{plot_x}" y1="{y:.2f}" x2="{plot_x + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                    f'<text class="axis" x="{plot_x - 10}" y="{y + 4:.2f}" text-anchor="end">{format_tick(tick, unit)}</text>',
                ]
            )
            tick += y_step
        for x_tick in x_ticks:
            x = sx(x_tick)
            parts.extend(
                [
                    f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_h}" stroke="#f3f4f6"/>',
                    f'<text class="axis" x="{x:.2f}" y="{plot_y + plot_h + 22}" text-anchor="middle">{x_tick // 1000}K</text>',
                ]
            )
        parts.append(
            f'<text class="axis" x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 44}" text-anchor="middle">prompt tokens</text>'
        )

        for _, color, points in series:
            if not points:
                continue
            path = " ".join(
                ("M" if index == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}"
                for index, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="#fff" stroke="{color}" stroke-width="2"/>'
                )
        if metric == "weighted_accept_length_median":
            parts.append(
                f'<text class="unit" x="{plot_x + plot_w - 4}" y="{plot_y + 17}" text-anchor="end">Laguna target-only: N/A</text>'
            )

    parts.extend(
        [
            f'<text class="footer" x="{width / 2}" y="885" text-anchor="middle">Decode is median per-request client-visible rate, not aggregate batch throughput. Input rate is prompt tokens / last client TTFT.</text>',
            f'<text class="footer" x="{width / 2}" y="907" text-anchor="middle">One audited cold-cache trial per cell · Qwen DFlash acceptance unsmoothed; Laguna target-only N/A · generated text retained · 2026-07-27</text>',
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
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
    )
    args = parser.parse_args()
    rows_by_model = load_rows(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for concurrency in (1, 2, 4):
        output = args.output_dir / f"dflash_concurrency_{concurrency}.svg"
        output.write_text(render_plot(rows_by_model, concurrency))
        print(output)


if __name__ == "__main__":
    main()
