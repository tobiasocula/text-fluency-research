import numpy as np
import cv2
from pathlib import Path
from extract_letters import extract_letters
import sys
from match_letter import match_letter

trainingdir = Path.cwd() / "data" / "cropped_characters"
bardir = Path.cwd() / "data" / "cropped_bars"

newdir_imgs = Path.cwd() / "data" / "cropped_chars_resized"
newdir_bars = Path.cwd() / "data" / "cropped_bars_resized"

images = [cv2.imread(str(x)) for x in list(trainingdir.glob("*.png"))]
bars = [cv2.imread(str(x)) for x in list(bardir.glob("*.png"))]
resized_images = [cv2.resize(x, (32, 32)) for x in images]
resized_bars = [cv2.resize(x, (32, 32)) for x in bars]
for idx, (i, b) in enumerate(zip(resized_images, resized_bars)):

    cv2.imwrite(str(newdir_imgs / f"img_{idx}.png"), i)
    cv2.imwrite(str(newdir_bars / f"bar_{idx}.png"), b)
    