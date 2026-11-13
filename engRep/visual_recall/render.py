from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)

LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def find_font(explicit: Path | None = None, *, require_cjk: bool = False) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"font not found: {explicit}")
        return explicit
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    if require_cjk:
        raise RuntimeError(
            "Chinese text requires a CJK font; install fonts-noto-cjk or pass "
            "--font-path /path/to/NotoSansCJK-Regular.ttc"
        )
    for candidate in LATIN_FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def load_font(path: Path | None, size: int) -> ImageFont.ImageFont:
    if path is None:
        return ImageFont.load_default(size=max(10, size))
    return ImageFont.truetype(str(path), size=size)


def _tokens(text: str) -> list[str]:
    # Keep English words together while allowing line breaks between CJK chars.
    return re.findall(r"[A-Za-z0-9$€£¥%+./:_-]+\s*|\s+|.", text, flags=re.DOTALL)


def wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for paragraph_index, paragraph in enumerate(text.splitlines() or [""]):
        for token in _tokens(paragraph.strip()):
            candidate = current + token
            width = draw.textbbox((0, 0), candidate.rstrip(), font=font)[2]
            if current and width > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
            current = ""
        if paragraph_index < len(text.splitlines()) - 1 and not paragraph.strip():
            lines.append("")
    return lines or [""]


def _text_height(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont, spacing: int) -> int:
    sample_box = draw.textbbox((0, 0), "Ag国", font=font)
    line_height = sample_box[3] - sample_box[1]
    return len(lines) * line_height + max(0, len(lines) - 1) * spacing


def render_labeled_image(
    source_path: Path,
    output_path: Path,
    *,
    title: str,
    source_footnote: str,
    font_path: Path | None = None,
    jpeg_quality: int = 94,
) -> Path:
    with Image.open(source_path) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
    width, height = source.size
    # Keep labels readable for narrow crops. Upscaling is only for presentation;
    # the untouched source remains available separately.
    if width < 640:
        scale = 640 / width
        source = source.resize((640, max(1, round(height * scale))), Image.Resampling.LANCZOS)
        width, height = source.size

    selected_font = find_font(font_path, require_cjk=contains_cjk(title + source_footnote))
    title_size = max(22, min(40, round(width / 25)))
    footer_size = max(15, min(24, round(width / 42)))
    title_font = load_font(selected_font, title_size)
    footer_font = load_font(selected_font, footer_size)
    probe = Image.new("RGB", (width, 10), "white")
    draw = ImageDraw.Draw(probe)
    horizontal_padding = max(18, round(width * 0.025))
    text_width = width - 2 * horizontal_padding
    title_lines = wrap_text(title, draw, title_font, text_width)
    footer_lines = wrap_text(source_footnote, draw, footer_font, text_width) if source_footnote else []
    title_spacing = max(5, title_size // 5)
    footer_spacing = max(3, footer_size // 5)
    title_height = _text_height(draw, title_lines, title_font, title_spacing) + 2 * horizontal_padding
    footer_height = (
        _text_height(draw, footer_lines, footer_font, footer_spacing) + 2 * horizontal_padding if footer_lines else 0
    )
    canvas = Image.new("RGB", (width, title_height + height + footer_height), "white")
    canvas.paste(source, (0, title_height))
    out_draw = ImageDraw.Draw(canvas)
    accent = (170, 25, 35)
    out_draw.line((0, 0, width, 0), fill=accent, width=max(2, width // 500))
    y = horizontal_padding
    sample_box = out_draw.textbbox((0, 0), "Ag国", font=title_font)
    title_line_height = sample_box[3] - sample_box[1]
    for line in title_lines:
        out_draw.text((horizontal_padding, y), line, fill=(18, 18, 18), font=title_font)
        y += title_line_height + title_spacing
    out_draw.line((0, title_height - 1, width, title_height - 1), fill=accent, width=max(1, width // 700))
    if footer_lines:
        footer_top = title_height + height
        out_draw.line((0, footer_top, width, footer_top), fill=accent, width=max(1, width // 700))
        y = footer_top + horizontal_padding
        sample_box = out_draw.textbbox((0, 0), "Ag国", font=footer_font)
        footer_line_height = sample_box[3] - sample_box[1]
        for line in footer_lines:
            out_draw.text((horizontal_padding, y), line, fill=(50, 50, 50), font=footer_font)
            y += footer_line_height + footer_spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
    return output_path
