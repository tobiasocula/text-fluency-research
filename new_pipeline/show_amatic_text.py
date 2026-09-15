"""
debug_amatic.py

Renders a line of text using the Amatic SC font and shows the result,
plus prints min/max/unique-ish pixel stats so you can see whether the
rendered glyphs ever actually hit value 0 (pure black).

Run with:  python3 debug_amatic.py
"""

from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# --- Config -----------------------------------------------------------------

#FONT_PATH = str(Path.cwd() / "amatic" / "AmaticSC-Regular.ttf")
#FONT_PATH = str(Path.cwd() / "arial" / "ARIAL.TTF")
#FONT_PATH = str(Path.cwd() / "times-new-roman" / "times.ttf")
FONT_PATH = str(Path.cwd() / "Edengarth-Font" / "Edengarth" / "OTF" / "Edengarth.otf")

TEXT = "Aazertyuiopmlkjhgfdsqwxcvbn"
FONT_SIZE = 20
LINE_SPACING = 10

# Background colour and text colour (PIL uses RGB tuples)
BG_COLOR = (255, 255, 255)
FG_COLOR = (0, 0, 0)

IMG_WIDTH = 600
IMG_HEIGHT = 120
MARGIN = 10


# --- Render -----------------------------------------------------------------

def text_to_img(text, font_path, font_size, line_spacing,
                img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
                bg_color=BG_COLOR, fg_color=FG_COLOR):
    """Minimal version of your text_to_img, just for this debug script."""
    font = ImageFont.truetype(font_path, font_size)

    img = Image.new("L", (img_width, img_height), color=bg_color[0])
    draw = ImageDraw.Draw(img)

    # Simple single-line draw (no wrapping). Enough to see pixel values.
    draw.text((MARGIN, MARGIN), text, font=font, fill=fg_color[0])

    return img


def main():
    print(f"Looking for font at: {FONT_PATH}")
    if not Path(FONT_PATH).exists():
        print("  -> FONT NOT FOUND. Adjust FONT_PATH.")
        return
    print("  -> font file exists.")

    try:
        f = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"  -> loaded font: {f.getname()}")
    except Exception as e:
        print(f"  -> FAILED to load font: {e}")
        return

    img = text_to_img(TEXT, FONT_PATH, FONT_SIZE, LINE_SPACING)
    arr = np.array(img)

    print("\nImage stats:")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  min   : {arr.min()}")
    print(f"  max   : {arr.max()}")
    print(f"  mean  : {arr.mean():.2f}")
    print(f"  pixels == 0   : {int(np.sum(arr == 0))}")
    print(f"  pixels <  128 : {int(np.sum(arr < 128))}")
    print(f"  pixels <  200 : {int(np.sum(arr < 200))}")
    print(f"  pixels <  250 : {int(np.sum(arr < 250))}")

    if np.sum(arr == 0) == 0:
        print("\n  NOTE: NO pure-black (0) pixels! This is why your")
        print("        layer detector with `img[y,:] == 0` finds 0 layers.")
        print("        Amatic SC at this size anti-aliases to grey values.")
        print("        Either relax the threshold (e.g. `< 128`) or use")
        print("        a heavier font / larger font size.")

    # Save to disk so you can eyeball it even without a GUI
    out_path = Path.cwd() / "amatic_debug.png"
    cv2.imwrite(str(out_path), arr)
    print(f"\nSaved render to: {out_path}")

    # Show it
    cv2.imshow("Amatic render (press any key to close)", arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()