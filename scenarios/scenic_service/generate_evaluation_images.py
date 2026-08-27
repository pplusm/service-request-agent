"""生成公开、无隐私风险的图文评测 PNG 夹具。

这些图片是简单的示意图，不来自真实景区，也不包含人物、车牌或地理信息。
评测运行器会把它们当作输入传输样本；当前的确定性视觉观察仍由案例声明提供，
不会声称从这些 PNG 中识别出了真实现场内容。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


IMAGE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "scenic_service" / "evaluation_images"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """按 PNG 规范生成一个带 CRC 校验的区块。"""

    body = chunk_type + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _write_png(path: Path, *, lamp_on: bool, blurred: bool) -> None:
    """绘制一张抽象路灯示意图，不使用任何第三方图像库。"""

    pixels: list[bytes] = []
    for y in range(IMAGE_HEIGHT):
        row = bytearray()
        for x in range(IMAGE_WIDTH):
            # 深色背景代表夜间示意场景，颜色不是现实照片或地点信息。
            red, green, blue = 32, 48, 72
            pole = 155 <= x <= 163 and 35 <= y <= 158
            head = 143 <= x <= 176 and 30 <= y <= 42
            glow = lamp_on and ((x - 160) ** 2 + (y - 52) ** 2 < 34 ** 2)
            if pole or head:
                red, green, blue = 150, 160, 170
            if glow:
                red, green, blue = 235, 196, 80
            if blurred and (x + y) % 7 < 3:
                # 用规则化噪点表达“模糊”，不包含任何真实图像内容。
                red = min(255, red + 18)
                green = min(255, green + 18)
                blue = min(255, blue + 18)
            row.extend((red, green, blue))
        pixels.append(b"\x00" + bytes(row))

    raw_pixels = b"".join(pixels)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", IMAGE_WIDTH, IMAGE_HEIGHT, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw_pixels, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    """创建三张可重复生成的公开评测图片。"""

    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_png(
        IMAGE_DIRECTORY / "lighting_west_gate_off.png",
        lamp_on=False,
        blurred=False,
    )
    _write_png(
        IMAGE_DIRECTORY / "lighting_west_gate_on.png",
        lamp_on=True,
        blurred=False,
    )
    _write_png(
        IMAGE_DIRECTORY / "unclear_scene.png",
        lamp_on=False,
        blurred=True,
    )
    print(f"已生成评测图片：{IMAGE_DIRECTORY}")


if __name__ == "__main__":
    main()
