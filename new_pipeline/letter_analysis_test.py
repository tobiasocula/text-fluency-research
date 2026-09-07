import json
from letter_analysis import *
from pathlib import Path

num_texts = 100

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]
lan_labels = ["french", "german", "hun", "fin"]

lan_stats = [] # list per stats: each is entry [std_upper, std_lower, med_upper, med_lower]

num_texts = 100

from new_vertical.funcs import get_valid_texts
lan_texts = get_valid_texts(jsonl_files, num_texts)


for i,texts in enumerate(lan_texts):
    print('in language', i, 'out of', len(lan_texts))

    lan_stats_std_upper = []
    lan_stats_std_lower = []
    lan_stats_med_upper = []
    lan_stats_med_lower = []
    l = len(texts)

    for j,text in enumerate(texts):
        print('in text', j, 'out of', l)
        std_upper, std_lower, med_upper, med_lower, mean_upper, mean_lower = analyze_text(text)
        lan_stats_std_lower.append(std_lower)
        lan_stats_std_upper.append(std_upper)
        lan_stats_med_lower.append(med_lower)
        lan_stats_med_upper.append(med_upper)

    lan_stats.append([
        np.mean([x for x in lan_stats_std_upper if x is not None]),
        np.mean([x for x in lan_stats_std_lower if x is not None]),
        np.mean([x for x in lan_stats_med_upper if x is not None]),
        np.mean([x for x in lan_stats_med_lower if x is not None]),
    ])


for label, stats in zip(lan_labels, lan_stats):
    print('language:', label)
    print('stats:'); print(stats)
    print()

        
