"""
MANUAL DRAWING
"""

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from scipy.signal import find_peaks
import sys
from pathlib import Path

font = ImageFont.truetype(str(Path.cwd() / "arial" / "ARIAL.ttf"), 200)

letters_dir = Path.cwd() / "new_models" / "arial" / "letters"
bars_dir = Path.cwd() / "new_models" / "arial" / "bars"

img = Image.new("L", (256, 256), 255)
draw = ImageDraw.Draw(img)

letter = "Z"

draw.text((50, 10), letter, font=font, fill=0)

img = np.array(img)      # Convert the PIL Image, not the Draw object


# remove white margin space
_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(thresh)
x, y, w, h = cv2.boundingRect(coords)

cropped_letter = img[y:y+h, x:x+w]
pad = 5
# add padding
cropped_letter = cv2.copyMakeBorder(
    cropped_letter,
    pad, pad, pad, pad,
    cv2.BORDER_CONSTANT,
    value=255
)

bars = np.zeros_like(cropped_letter)
bars[:, 20 : 35] = 255
bars[:, 95 : 110] = 255
#bars[:, 165 : 180] = 255
letter_bgr = cv2.cvtColor(cropped_letter, cv2.COLOR_GRAY2BGR)
letter_bgr[bars == 255] = (0, 0, 255)  

cv2.imshow("draw", letter_bgr)
cv2.waitKey(0)

to_dir = Path.cwd() / "new_vertical" / "letters_w_bars"
to_dir_normal = Path.cwd() / "new_vertical" / "letters"
cv2.imwrite(to_dir / f"{letter}_cap.png", letter_bgr)
cv2.imwrite(to_dir_normal / f"{letter}_cap.png", cropped_letter)