import pandas as pd
from pathlib import Path
from new_pipeline.funcs import draw_bar, extract_letters, text_to_img
import cv2
import numpy as np
import sys

df = pd.read_csv(Path.cwd() / "wmf" / "wmf.csv")
hu = df[df["language"] == "hu"]["text"].sample(10, random_state=0) # is pd.Series
#hu.to_csv("hungarian_fragments.txt", index=False, header=False)
hu = hu.tolist()[0]
print('HU:'); print(hu)

text_len = 20
text = hu[:text_len] # smaller section of text

img = text_to_img(
    text=text,
    font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
    font_size=50
)
img = np.array(img)
cv2.imshow("img", img)
cv2.waitKey(0)


letters = extract_letters(img)


print('extracted letters, len:'); print(len(letters))
for l in letters:
    letter,coords = l
    #cv2.imshow("bar", np.array(draw_bar(letter, 0.5, 0.5, 10)))
    #cv2.imshow("bar", np.array(letter))
    cv2.imshow("bar", np.array(draw_bar(np.array(letter), 0.5, 0.5, 10)))
    cv2.waitKey(0)
    