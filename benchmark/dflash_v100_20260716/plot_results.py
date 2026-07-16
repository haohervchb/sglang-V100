#!/usr/bin/env python3
"""Render dependency-light PNG plots from the preserved DFlash summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots"
WIDTH = 1800
HEIGHT = 1400

BACKGROUND = "#F7F8FC"
PANEL = "#FFFFFF"
TEXT = "#182230"
MUTED = "#64748B"
GRID = "#D9E0EA"
BLUE = "#2563EB"
ORANGE = "#E4572E"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def load_summary(name: str) -> list[dict]:
    return json.loads((ROOT / name / "summary.json").read_text())


def nice_ceiling(value: float, ticks: int = 5) -> float:
    if value <= 0:
        return 1.0
    rough_step = value / ticks
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
    return math.ceil(value / step) * step


def format_tick(value: float, scale: float) -> str:
    scaled = value * scale
    if abs(scaled) >= 100:
        return f"{scaled:,.0f}"
    if abs(scaled) >= 10:
        return f"{scaled:.0f}"
    if abs(scaled) >= 1:
        return f"{scaled:.1f}".rstrip("0").rstrip(".")
    return f"{scaled:.2f}".rstrip("0").rstrip(".")


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def endpoint_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: float,
    scale: float,
    color: str,
    y_offset: int,
) -> None:
    label = format_tick(value, scale)
    label_font = font(19, bold=True)
    tw, th = text_size(draw, label, label_font)
    right = int(x - 11)
    top = int(y + y_offset - th / 2 - 6)
    draw.rounded_rectangle(
        (right - tw - 14, top, right, top + th + 12),
        radius=8,
        fill="#FFFFFFE8",
        outline=color,
        width=2,
    )
    draw.text((right - tw - 7, top + 5), label, font=label_font, fill=color)


def draw_panel(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    title: str,
    unit: str,
    metric: str,
    scale: float,
    dense: list[dict],
    moe: list[dict],
    concurrency: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=22, fill=PANEL, outline="#E2E8F0", width=2)
    draw.text((left + 28, top + 22), title, font=font(29, bold=True), fill=TEXT)
    draw.text((left + 28, top + 59), unit, font=font(19), fill=MUTED)

    chart_left = left + 94
    chart_right = right - 34
    chart_top = top + 105
    chart_bottom = bottom - 68

    series = []
    for label, color, rows in (
        ("27B FP16", BLUE, dense),
        ("122B GPTQ Int4", ORANGE, moe),
    ):
        selected = sorted(
            (row for row in rows if row["concurrency"] == concurrency),
            key=lambda row: row["prompt_len"],
        )
        values = [float(row[f"{metric}_median"]) for row in selected]
        lows = [float(row[f"{metric}_min"]) for row in selected]
        highs = [float(row[f"{metric}_max"]) for row in selected]
        series.append((label, color, selected, values, lows, highs))

    observed_max = max(max(item[5]) for item in series)
    y_max = nice_ceiling(observed_max * 1.08)
    if metric == "weighted_accept_length":
        y_max = 20.0

    x_values = [row["prompt_len"] / 1000 for row in series[0][2]]

    def x_px(value: float) -> float:
        return chart_left + (value - 1) / 24 * (chart_right - chart_left)

    def y_px(value: float) -> float:
        return chart_bottom - value / y_max * (chart_bottom - chart_top)

    tick_count = 5
    tick_font = font(18)
    for idx in range(tick_count + 1):
        raw_value = y_max * idx / tick_count
        y = y_px(raw_value)
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=2)
        label = format_tick(raw_value, scale)
        tw, th = text_size(draw, label, tick_font)
        draw.text((chart_left - tw - 13, y - th / 2), label, font=tick_font, fill=MUTED)

    for value in x_values:
        x = x_px(value)
        draw.line((x, chart_top, x, chart_bottom), fill="#EDF1F6", width=1)
        label = f"{int(value)}"
        tw, _ = text_size(draw, label, tick_font)
        draw.text((x - tw / 2, chart_bottom + 13), label, font=tick_font, fill=MUTED)

    for _, color, _, values, lows, highs in series:
        upper = [(x_px(x), y_px(y)) for x, y in zip(x_values, highs)]
        lower = [(x_px(x), y_px(y)) for x, y in zip(x_values, lows)]
        draw.polygon(upper + list(reversed(lower)), fill=color + "22")
        points = [(x_px(x), y_px(y)) for x, y in zip(x_values, values)]
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#FFFFFF", outline=color, width=4)

    endpoint_label(
        draw,
        x_px(x_values[-1]),
        y_px(series[0][3][-1]),
        series[0][3][-1],
        scale,
        BLUE,
        -17,
    )
    endpoint_label(
        draw,
        x_px(x_values[-1]),
        y_px(series[1][3][-1]),
        series[1][3][-1],
        scale,
        ORANGE,
        18,
    )

    x_title = "Prompt context (K tokens)"
    tw, _ = text_size(draw, x_title, font(18))
    draw.text(
        ((chart_left + chart_right - tw) / 2, bottom - 31),
        x_title,
        font=font(18),
        fill=MUTED,
    )


def render(concurrency: int, dense: list[dict], moe: list[dict]) -> Path:
    image = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")

    draw.text(
        (70, 44),
        f"DFlash on 4× V100 — concurrency {concurrency}",
        font=font(48, bold=True),
        fill=TEXT,
    )
    draw.text(
        (72, 104),
        "27B dense FP16 vs 122B-A10B GPTQ Int4 · medians of 3 cold-cache trials",
        font=font(24),
        fill=MUTED,
    )

    legend_y = 100
    for x, color, label in (
        (1260, BLUE, "Qwen3.6-27B FP16"),
        (1260, ORANGE, "Qwen3.5-122B GPTQ Int4"),
    ):
        current_y = legend_y
        draw.line((x, current_y, x + 48, current_y), fill=color, width=6)
        draw.ellipse((x + 18, current_y - 7, x + 32, current_y + 7), fill="#FFFFFF", outline=color, width=4)
        draw.text((x + 62, current_y - 14), label, font=font(20, bold=True), fill=TEXT)
        legend_y += 38

    outer_left = 70
    outer_right = 70
    top = 184
    bottom = 112
    gap_x = 38
    gap_y = 38
    panel_width = (WIDTH - outer_left - outer_right - gap_x) // 2
    panel_height = (HEIGHT - top - bottom - gap_y) // 2

    panels = [
        ("Aggregate decode", "post-first-token throughput", "aggregate_decode_tps", 1.0),
        ("Effective prefill", "client-observed, includes scheduling · kTok/s", "effective_prefill_tps", 0.001),
        ("Median TTFT", "client-observed localhost latency · seconds", "median_ttft_ms", 0.001),
        ("DFlash acceptance", "weighted accepted tokens per verification", "weighted_accept_length", 1.0),
    ]
    for idx, (title, unit, metric, scale) in enumerate(panels):
        col = idx % 2
        row = idx // 2
        left = outer_left + col * (panel_width + gap_x)
        panel_top = top + row * (panel_height + gap_y)
        draw_panel(
            image,
            (left, panel_top, left + panel_width, panel_top + panel_height),
            title,
            unit,
            metric,
            scale,
            dense,
            moe,
            concurrency,
        )

    footer = (
        "Shading shows observed min–max. Cache flushed before every trial. "
        "Aggregate decode excludes each request’s first streamed token."
    )
    draw.text((72, HEIGHT - 78), footer, font=font(20), fill=MUTED)
    draw.text(
        (WIDTH - 72, HEIGHT - 38),
        "sglang-V100 · 4c8434780 · 2026-07-16",
        font=font(19),
        fill=MUTED,
        anchor="ra",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"concurrency_{concurrency}.png"
    image.convert("RGB").save(output, format="PNG", optimize=False, dpi=(150, 150))
    return output


def main() -> None:
    dense = load_summary("27b")
    moe = load_summary("122b")
    for concurrency in (1, 2, 4):
        output = render(concurrency, dense, moe)
        print(output)


if __name__ == "__main__":
    main()
