"""Generate the controlled image and video fixtures used by validate.py."""

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "fixtures"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def main():
    ROOT.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT, 48)
    for name, circle, square in [
        ("scene_a", "red", "blue"),
        ("scene_b", "blue", "red"),
    ]:
        canvas = Image.new("RGB", (768, 512), "white")
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((64, 120, 304, 360), fill=circle)
        draw.rectangle((464, 120, 704, 360), fill=square)
        canvas.save(ROOT / f"{name}.png")
    canvas = Image.new("RGB", (768, 384), "white")
    ImageDraw.Draw(canvas).text((70, 100), "V100 CHECK 4827", font=font, fill="black")
    canvas.save(ROOT / "ocr.png")
    canvas = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, value) in enumerate([("ALPHA", 3), ("BETA", 7), ("GAMMA", 5)]):
        x = 60 + 245 * i
        draw.rectangle(
            (x, 420 - value * 45, x + 160, 420), fill=["red", "blue", "green"][i]
        )
        draw.text((x + 55, 425 - value * 45), str(value), font=font, fill="white")
        draw.text((x, 440), label, font=ImageFont.truetype(FONT, 36), fill="black")
    canvas.save(ROOT / "chart.png")
    for name, colors in [
        ("forward", ["red", "green", "blue"]),
        ("reverse", ["blue", "green", "red"]),
    ]:
        with av.open(str(ROOT / f"{name}.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=8)
            stream.width, stream.height, stream.pix_fmt = 384, 256, "yuv420p"
            for color in colors:
                pixels = np.asarray(Image.new("RGB", (384, 256), color))
                for _ in range(16):
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)


if __name__ == "__main__":
    main()
