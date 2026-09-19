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


text = "qlmskdjfmlqskdjfmlkqsjdmflkqsjmdlfkjqsmldfkjqsmldkfj"
text *= 100
from text_to_img_new import text_to_img
img = text_to_img(text, FONT_PATH, 20, max_width=5000)
cv2.imshow("img", np.array(img))
cv2.waitKey(0)