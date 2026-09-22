from funcs import get_valid_texts, extract_letters, text_to_image, match_letter, eval_img, remove_diacritics, extract_letters_template
from pathlib import Path
import cv2
import numpy as np


text = "mjééä"

ctext = remove_diacritics(text)

image = text_to_image(text,
                        font_path=str(Path.cwd() / "arial" / "ARIAL.TTF"),
                        font_size=20)
cimage = text_to_image(ctext,
                        font_path=str(Path.cwd() / "arial" / "ARIAL.TTF"),
                        font_size=20)

FONT_PATH = str(Path.cwd() / "arial" / "ARIAL.TTF")

#letters, _ = extract_letters_template(image, text, FONT_PATH)
cletters, _ = extract_letters_template(cimage, ctext, FONT_PATH, fontsize=20)
letters = cletters[:]

for l,c in zip(letters, cletters):
    print('here')
    a,b = l[0], c[0]
    cv2.imshow("a", b)
    cv2.waitKey(0)



"""
python3 new_vertical/test_remove_diacritics.py
"""