from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path
from extract_letters import extract_letters
from match_letter import match_letter
import sys

"""
MAIN FILE FOR DRAWING BARS ON IMAGES
"""

letters_dir = Path.cwd() / "new_models" / "arial" / "letters"
bars_dir = Path.cwd() / "new_models" / "arial" / "bars"
imgpath = Path.cwd() / "arial_alphabet.png"
#imgpath = Path.cwd() / "amatic_text.png"
#imgpath = Path.cwd() / "amatic_alphabet.png"

img = cv2.imread(str(imgpath))

letters, _ = extract_letters(str(imgpath))
result_img = img.copy()

for letterdata in letters:
    letter = letterdata[0]
    x, y, w, h = letterdata[1]

    best_match, matches, scores, corresponding_bar = match_letter(
        letter=letter,
        training_data_dir=letters_dir,
        bars_dir=bars_dir
    )



#     print('new')
#     cv2.imshow("letter", letter)
#     cv2.waitKey(0)

# #    shape of corresp_bar and best_match: (256, 256)
#     cv2.imshow("best match", best_match)
#     cv2.waitKey(0)
#     cv2.imshow("corresp bar", corresponding_bar)
#     cv2.waitKey(0)

    rescaled_bar = cv2.resize(corresponding_bar, (w, h))
    # cv2.imshow("b", rescaled_bar)
    # cv2.waitKey(0)
    result_img[y:y+h, x:x+w][rescaled_bar == 255] = (0, 0, 255)

cv2.imwrite(str(Path.cwd() / "new_models" / "arial" / "output_images" / "alphabet.png"), result_img)

cv2.imshow("r", result_img)
cv2.waitKey(0)