import pandas as pd
from pathlib import Path
from new_pipeline.funcs import *
import cv2
import numpy as np
import sys
from scipy.signal import savgol_filter

# df = pd.read_csv(Path.cwd() / "wmf" / "wmf.csv")
# hu = df[df["language"] == "hu"]["text"].sample(10, random_state=0) # is pd.Series
# #hu.to_csv("hungarian_fragments.txt", index=False, header=False)
# hu = hu.tolist()[0]
# print('HU:'); print(hu)

# text_len = 20
# text = hu[:text_len] # smaller section of text

# img = text_to_img(
#     text=text,
#     font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
#     font_size=50
# )
# img2 = np.array(img)

def crop_text_region(img, thresh=250):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    cols = np.where(bw.sum(axis=0) > 0)[0]
    rows = np.where(bw.sum(axis=1) > 0)[0]

    if len(cols) == 0 or len(rows) == 0:
        return None

    x1, x2 = cols[0], cols[-1]
    y1, y2 = rows[0], rows[-1]
    return img[y1:y2+1, x1:x2+1]


# img = cv2.imread(Path.cwd() / "languages_horizontal" / "text_English_01.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print('shape img:', img.shape)
# print(type(img))
# cut_whitespace(img)
# scale = 0.8
# h,w = img.shape
# img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

df = pd.read_csv(Path.cwd() / "wmf" / "wmf.csv")
hu = df[df["language"] == "hu"]["text"].sample(10, random_state=0) # is pd.Series
#hu.to_csv("hungarian_fragments.txt", index=False, header=False)
hu = hu.tolist()[0]
length = 500
print('HU:'); print(hu)
hu = hu[:length]
img = text_to_img(
    text=hu,
    font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
    font_size=50
)
img = np.array(img)
#cv2.imshow("img", img)
#cv2.waitKey(0)

compute_scores(img)



#cv2.imshow("img", img)
#cv2.waitKey(0)