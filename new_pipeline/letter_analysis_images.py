import json
from letter_analysis import *
from pathlib import Path
from new_vertical.funcs import *

num_texts = 1000

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]
lan_labels = ["french", "german", "hun", "fin"]

all_texts = get_valid_texts(jsonl_files, num_texts)

assert len(all_texts) == lan_labels, AssertionError()

per_language_stats = [] # [std lower, std upper, med lower, med upper] per lan

for label, language_texts in zip(lan_labels, all_texts):
    print('LANGUAGE:', label)
    current_stats = np.empty((len(language_texts), 4))
    for i,text in enumerate(language_texts):
        img = text_to_img(
            text=text,
            font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
            font_size=50
        )
        img = np.array(img)
        print('LOADED IMAGE')
        current_stats[i,:] = compute_scores_att_2(img, show_contours=False, show_boundaries=False, showbar=False)

    per_language_stats.append(np.mean(current_stats, axis=0))

print('FINAL STATS')
for label, i in zip(lan_labels, range(len(per_language_stats))):
    print('language:', label)
    print('std lower:', per_language_stats[i,0])
    print('std upper:', per_language_stats[i,1])
    print('med lower:', per_language_stats[i,2])
    print('med upper:', per_language_stats[i,3])
    print()

        
        
        

