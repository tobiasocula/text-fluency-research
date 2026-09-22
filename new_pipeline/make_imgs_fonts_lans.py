import json
from pathlib import Path
import numpy as np
from funcs import *


num_texts = 1
charlim = 1000
fonts = [Path.cwd()/"arial"/"ARIAL.TTF",
         Path.cwd()/"chunkfive"/"ChunkFive-Regular.otf",
         Path.cwd()/"Cormorant"/"Cormorant-VariableFont_wght.ttf",
         Path.cwd()/"times-new-roman"/"times.ttf"]
fontnames = ["arial","chunkfive","cormorant","times"]

def filter_text(text, filter_chars):
    return "".join(c for c in text if c not in filter_chars)

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]
lan_labels = ["french", "german", "hun", "fin"]

all_texts = get_valid_texts(jsonl_files, num_texts, charlim)
all_texts = [
    [filter_text(t, "0123456789,?!;/:+-*=") for t in texts_per_lan]
    for texts_per_lan in all_texts
]

for label, texts in zip(lan_labels, all_texts):
    text = texts[0]
    for fontpath, fn in zip(fonts, fontnames):
        font = ImageFont.truetype(fontpath, 20)
        lines = wrap_lines(text, max_chars=100)
        img = lines_to_img(lines, font)
        res = compute_scores_att_3(np.array(img), call_idx=0, debug=True, return_img=True)
        path = Path.cwd()/"new_pipeline"/"func_3_sample_outputs"/f"{label}_{fn}.png"
        cv2.imwrite(path, res)


