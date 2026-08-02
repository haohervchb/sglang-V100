#!/usr/bin/env python3
"""Render the committed TP2/TP4 context-scaling results as a standalone SVG."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH = 1440
HEIGHT = 560
COLORS = {"TP2": "#2563eb", "TP4": "#ea580c"}


def load_summary(name: str) -> list[dict]:
    rows = json.loads((ROOT / name / "summary.json").read_text())
    return sorted(rows, key=lambda row: row["prompt_len"])


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
    return "<line " + " ".join(
        f'{key}="{esc(value)}"' for key, value in values.items()
    ) + "/>"


def text(x: float, y: float, value: object, **attrs: object) -> str:
    values = {"x": x, "y": y, **attrs}
    return (
        "<text "
        + " ".join(f'{key}="{esc(item)}"' for key, item in values.items())
        + f">{esc(value)}</text>"
    )


def render_panel(
    x0: float,
    title: str,
    metric: str,
    y_max: float,
    y_step: float,
    y_label: str,
    data: dict[str, list[dict]],
) -> list[str]:
    top, plot_height, plot_width = 130, 330, 360
    left = x0 + 62
    bottom = top + plot_height
    out = [
        text(
            x0 + 220,
            104,
            title,
            **{"text-anchor": "middle", "class": "panel-title"},
        )
    ]

    tick = 0.0
    while tick <= y_max + 1e-9:
        y = bottom - plot_height * tick / y_max
        out.append(line(left, y, left + plot_width, y, **{"class": "grid"}))
        label = f"{tick:,.0f}" if y_max > 100 else f"{tick:.0f}"
        out.append(
            text(
                left - 10,
                y + 5,
                label,
                **{"text-anchor": "end", "class": "tick"},
            )
        )
        tick += y_step

    for context in (1_000, 5_000, 9_000, 13_000, 17_000, 21_000, 25_000):
        x = left + plot_width * (context - 1_000) / 24_000
        out.append(line(x, bottom, x, bottom + 6, **{"class": "axis"}))
        out.append(
            text(
                x,
                bottom + 25,
                f"{context // 1000}K",
                **{"text-anchor": "middle", "class": "tick"},
            )
        )

    out.extend(
        [
            line(left, top, left, bottom, **{"class": "axis"}),
            line(left, bottom, left + plot_width, bottom, **{"class": "axis"}),
            text(
                left + plot_width / 2,
                bottom + 53,
                "Input context",
                **{"text-anchor": "middle", "class": "axis-label"},
            ),
            text(
                x0 + 15,
                top + plot_height / 2,
                y_label,
                transform=f"rotate(-90 {x0 + 15} {top + plot_height / 2})",
                **{"text-anchor": "middle", "class": "axis-label"},
            ),
        ]
    )

    for label, rows in data.items():
        points = []
        for row in rows:
            value = float(row[metric])
            x = left + plot_width * (row["prompt_len"] - 1_000) / 24_000
            y = bottom - plot_height * value / y_max
            points.append((x, y, value, row["prompt_len"]))
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in points)
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{COLORS[label]}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y, value, context in points:
            out.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" '
                f'fill="{COLORS[label]}" stroke="#ffffff" stroke-width="1.5">'
                f"<title>{esc(label)} · {context // 1000}K · {value:.2f}</title></circle>"
            )
    return out


def main() -> None:
    data = {"TP2": load_summary("tp2"), "TP4": load_summary("tp4")}
    panels = (
        (
            20,
            "Effective prefill",
            "effective_input_tps_median",
            4_000,
            1_000,
            "Input tok/s",
        ),
        (
            500,
            "Client-visible decode",
            "median_request_decode_tps_median",
            180,
            30,
            "Output tok/s",
        ),
        (
            980,
            "DFlash acceptance",
            "weighted_accept_length_median",
            7,
            1,
            "Accepted tokens / verify",
        ),
    )

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Qwen3.6-27B-FP8 DFlash TP2 versus TP4 context scaling on V100</title>',
        '<desc id="desc">Effective prefill, client-visible decode, and DFlash '
        "accepted length from one to twenty-five thousand input tokens.</desc>",
        """<style>
            text { font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                   sans-serif; fill: #172033; }
            .title { font-size: 24px; font-weight: 700; }
            .subtitle { font-size: 14px; fill: #596579; }
            .panel-title { font-size: 17px; font-weight: 650; }
            .tick { font-size: 12px; fill: #596579; }
            .axis-label { font-size: 13px; font-weight: 600; fill: #334155; }
            .axis { stroke: #64748b; stroke-width: 1.2; }
            .grid { stroke: #dbe2ea; stroke-width: 1; }
            .legend { font-size: 13px; font-weight: 650; }
        </style>""",
        '<rect width="1440" height="560" rx="12" fill="#ffffff"/>',
        text(
            720,
            34,
            "Qwen3.6-27B-FP8 DFlash scaling on V100",
            **{"text-anchor": "middle", "class": "title"},
        ),
        text(
            720,
            58,
            "Single cold trial per point · 256 greedy output tokens · FP16 KV · DFlash block 16",
            **{"text-anchor": "middle", "class": "subtitle"},
        ),
        line(620, 80, 650, 80, stroke=COLORS["TP2"], **{"stroke-width": 3}),
        '<circle cx="635" cy="80" r="4.2" fill="#2563eb"/>',
        text(660, 85, "TP2 (2× V100 32GB)", **{"class": "legend"}),
        line(810, 80, 840, 80, stroke=COLORS["TP4"], **{"stroke-width": 3}),
        '<circle cx="825" cy="80" r="4.2" fill="#ea580c"/>',
        text(850, 85, "TP4 (4× V100 32GB)", **{"class": "legend"}),
    ]
    for panel in panels:
        body.extend(render_panel(*panel, data))
    body.append("</svg>")
    (ROOT / "context_scaling.svg").write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    main()
