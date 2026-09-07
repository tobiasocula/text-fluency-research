
import cv2
import numpy as np
from pathlib import Path
import sys
from new_vertical.funcs import get_valid_texts, extract_letters, text_to_image, match_letter, eval_img

letters_dir = Path.cwd() / "new_models" / "arial" / "letters"
bars_dir = Path.cwd() / "new_models" / "arial" / "bars"

num_texts = 200

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
    Path.cwd() / "spa.jsonl"
]
lan_labels = ["french", "german", "hun", "fin", "spa"]

def filter_text(text, filter_chars):
    res = ""
    for char in text:
        if char not in filter_chars:
            res += char
    return res

charlim = 1000
all_texts = get_valid_texts(jsonl_files, num_texts, charlim)

all_texts = [
    [filter_text(t, "0123456789,?!;/:+-*=") for t in texts_per_lan]
    for texts_per_lan in all_texts
]

assert len(all_texts) == len(lan_labels), AssertionError(f"lengths: {len(all_texts)}")
res_stats = {key: [] for key in lan_labels}

per_lan_dists = []

count = 0
failures = 0

for i,(label,texts) in enumerate(zip(lan_labels, all_texts)):
    print('in language', label)
    per_text_dists = []

    for j,text in enumerate(texts):
        print('TEXT LENGTH:', len(text))
        print('in text', j)

        image = text_to_image(text,
                            font_path=str(Path.cwd() / "arial" / "arial.ttf")
        )
        letters, _ = extract_letters(image)
        result_img = image.copy()
        for letterdata in letters:
            letter = letterdata[0]
            x, y, w, h = letterdata[1]

            best_match, matches, scores, corresponding_bar = match_letter(
                    letter=letter,
                    training_data_dir=Path.cwd() / "new_vertical" / "letters",
                    bars_dir=Path.cwd() / "new_vertical" / "bars"
                )

            _, text_mask = cv2.threshold(letter, 150, 255, cv2.THRESH_BINARY)
            
            # Resize the bar to match the letter
            bar_correct_size = cv2.resize(corresponding_bar, (w, h))
        
            # print('len:', len(bar_correct_size.shape))
            # cv2.imshow("bar", bar_correct_size)
            # cv2.waitKey(0)
        
            bar_mask = bar_correct_size > 0
            #print('bar mask:'); print(bar_correct_size)
        
            letter_region = result_img[y:y+h, x:x+w]
            letter_region[bar_mask] = (0, 0, 255)
            result_img[y:y+h, x:x+w] = letter_region

        hors = eval_img(result_img)
        for x in hors:
            per_text_dists.append(x)

    res_stats[label] = {
        "std": np.std(per_text_dists),
        "mean": np.mean(per_text_dists)
    }
import json

print('FINAL STATS:'); print(res_stats)
with open(Path.cwd() / "new_vertical" / "results.json", "w") as f:
    json.dump(res_stats, f)