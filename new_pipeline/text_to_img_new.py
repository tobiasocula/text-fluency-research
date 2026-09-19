"""Render text to a PIL image with reliable word wrapping.

Fixes over the naive version:
  * wraps by pixel width OR character count
  * breaks words that are longer than the line limit instead of overflowing
  * preserves explicit newlines and blank lines
  * uses font metrics for line height, so spacing is uniform regardless of
    whether a line happens to contain descenders
  * accounts for negative left bearing (italics, some scripts) so glyphs
    are never clipped
  * refuses to build absurdly large images instead of dying on memory
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# One throwaway canvas reused for all measurements.
_MEASURE = ImageDraw.Draw(Image.new("L", (1, 1)))

# Safety net: refuse to allocate anything bigger than this (in pixels).
MAX_PIXELS = 80_000_000


def _width(text: str, font: ImageFont.FreeTypeFont) -> float:
    """Advance width of a single line of text."""
    if not text:
        return 0.0
    return _MEASURE.textlength(text, font=font)


def _break_long_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
) -> list[str]:
    """Split a single word that cannot fit on one line into chunks."""
    chunks: list[str] = []
    current = ""

    for ch in word:
        candidate = current + ch
        if current and _width(candidate, font) > max_width:
            chunks.append(current)
            current = ch
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks or [word]


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Wrap `text` into a list of lines.

    Control the line length with either:
      max_width  -- maximum line width in pixels
      max_chars  -- maximum number of characters per line

    If both are given, whichever limit is hit first applies. If neither is
    given, only explicit newlines in the input break lines.
    """
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be positive")

    lines: list[str] = []

    # Honour the user's own line breaks; wrap within each paragraph.
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")  # preserve blank lines
            continue

        current = ""

        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"

            too_wide = max_width is not None and _width(candidate, font) > max_width
            too_long = max_chars is not None and len(candidate) > max_chars

            if not (too_wide or too_long):
                current = candidate
                continue

            if current:
                lines.append(current)
                current = ""

            # The word alone may still not fit -- hard-break it.
            word_too_wide = max_width is not None and _width(word, font) > max_width
            word_too_long = max_chars is not None and len(word) > max_chars

            if word_too_wide or word_too_long:
                pieces = (
                    _break_long_word(word, font, max_width)
                    if word_too_wide
                    else [word[i:i + max_chars] for i in range(0, len(word), max_chars)]
                )
                lines.extend(pieces[:-1])
                current = pieces[-1]
            else:
                current = word

        if current:
            lines.append(current)

    return lines or [""]


def text_to_img(
    text: str,
    font_path: str | Path,
    font_size: int,
    out_path: str | Path | None = None,
    padding: int = 20,
    max_width: float | None = 800,
    max_chars: int | None = None,
    line_spacing: int = 8,
    mode: str = "L",
    bg=255,
    fg=0,
) -> Image.Image:
    """Render `text` to an image, wrapping as needed.

    max_width : pixel budget for the text area (excluding padding). None to
                disable pixel-based wrapping.
    max_chars : hard character-count budget per line. None to disable.
    """
    font = ImageFont.truetype(str(font_path), font_size)
    lines = wrap_text(text, font, max_width=max_width, max_chars=max_chars)

    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    step = line_height + line_spacing

    # Measure the actual ink box of each line, anchored at the baseline-top.
    left = 0.0
    right = 0.0
    for line in lines:
        if not line:
            continue
        x0, _, x1, _ = _MEASURE.textbbox((0, 0), line, font=font, anchor="la")
        left = min(left, x0)
        right = max(right, x1)

    x_shift = -min(0.0, left)  # push right if a glyph overhangs to the left

    text_w = int(round(right + x_shift))
    text_h = line_height * len(lines) + line_spacing * (len(lines) - 1)

    img_w = max(1, text_w + 2 * padding)
    img_h = max(1, text_h + 2 * padding)

    if img_w * img_h > MAX_PIXELS:
        raise ValueError(
            f"Refusing to render a {img_w}x{img_h} image "
            f"({img_w * img_h:,} px). Lower font_size, or set a smaller "
            f"max_width / max_chars."
        )

    img = Image.new(mode, (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        if line:
            draw.text((padding + x_shift, y), line, font=font, fill=fg, anchor="la")
        y += step

    if out_path is not None:
        img.save(out_path)

    return img





def text_to_imgs(
    text, font_path, font_size,
    max_width=800, max_chars=None,
    lines_per_page=60,
    padding=20, line_spacing=8,
    mode="L", bg=255, fg=0,
    out_pattern=None,   # e.g. "page_{:03d}.png"
):
    """Render text across as many images as needed."""
    font = ImageFont.truetype(str(font_path), font_size)
    lines = wrap_text(text, font, max_width=max_width, max_chars=max_chars)

    pages = []
    for i in range(0, len(lines), lines_per_page):
        chunk = "\n".join(lines[i:i + lines_per_page])
        out = out_pattern.format(len(pages)) if out_pattern else None
        pages.append(
            text_to_img(
                chunk, font_path, font_size,
                out_path=out,
                padding=padding,
                max_width=None,       # already wrapped
                max_chars=None,
                line_spacing=line_spacing,
                mode=mode, bg=bg, fg=fg,
            )
        )
    return pages