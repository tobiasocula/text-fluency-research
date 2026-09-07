import json
from letter_analysis import *
from pathlib import Path
import numpy as np

num_texts = 100

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]

json_file = np.random.choice(jsonl_files)
from new_vertical.funcs import *

max_len_text = 1000

text = get_valid_texts([json_file], 1)[0][0][:max_len_text]
assert isinstance(text, str), AssertionError(f"type: {type(text)}")

img = text_to_img(
    text=text,
    font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
    font_size=20
)
img = np.array(img)
cv2.imshow("img", img)
cv2.waitKey(0)
print('LOADED IMAGE')

compute_scores_att_2(img, show_contours=True, show_boundaries=True, showbar=True)
