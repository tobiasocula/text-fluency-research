from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path

font = ImageFont.truetype(str(Path.cwd() / "times-new-roman" / "times.ttf"), 200)

img = Image.new("L", (256,256), 255)
draw = ImageDraw.Draw(img)

draw.text((30,20), "O", font=font, fill=0)

img = np.array(img)

_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,25))
vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('vertical', vertical)
cv2.waitKey(0)
