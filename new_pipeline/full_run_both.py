# import json
# from letter_analysis import *
# from pathlib import Path
# from funcs import *
# import sys

# num_texts = 1000

# jsonl_files = [
#     Path.cwd() / "french_texts_random_subset.jsonl",
#     Path.cwd() / "german_texts_subset.jsonl",
#     Path.cwd() / "hun_texts_subset.jsonl",
#     Path.cwd() / "fin_Latn" / "10_1.jsonl",
#     #Path.cwd() / "spa.jsonl"
# ]
# #lan_labels = ["french", "german", "hun", "fin", "spa"]
# lan_labels = ["french", "german", "hun", "fin"]

# def filter_text(text, filter_chars):
#     res = ""
#     for char in text:
#         if char not in filter_chars:
#             res += char
#     return res

# all_texts = get_valid_texts(jsonl_files, num_texts, None)
# print('amount of texts per lan before filter:', [len(x) for x in all_texts])
# all_texts = [
#     [filter_text(t, "0123456789,?!;/:+-*=") for t in texts_per_lan]
#     for texts_per_lan in all_texts
# ]

# assert len(all_texts) == len(lan_labels), AssertionError(f"lengths: {len(all_texts)}")

# res_stats = {key: [] for key in lan_labels}
# acc_lower_dists_analytical = []
# acc_upper_dists_analytical = []
# acc_lower_dists_graph = []
# acc_upper_dists_graph = []

# count = 0
# failures = 0

# for i,(label,texts) in enumerate(zip(lan_labels, all_texts)):
#     print('in language', label)

#     l = len(texts)

#     for j,text in enumerate(texts):
#         print('TEXT LENGTH:', len(text))
#         print('in text', j)
#         uppers, lowers = analyze_text(text)
#         for l in lowers:
#             acc_lower_dists_analytical.append(l)
#         for u in uppers:
#             acc_upper_dists_analytical.append(u)

#         img = text_to_img(text,
#                           #font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
#                           #font_path=str(Path.cwd() / "arial" / "ARIAL.TTF"),
#                           font_path=str(Path.cwd() / "chunkfive" / "Chunk Five Print.otf"),
#                         #font_path=str(Path.cwd() / "amatic" / "AmaticSC-Regular.ttf"),
#                           font_size=20,
#                           line_spacing=10
#         )
#         res = compute_scores_att_2(np.array(img), call_idx=count, debug=False)
#         if res is None:
#             failures += 1
#             continue

#         uppers, lowers = res
        
#         for u in uppers:
#             acc_upper_dists_graph.append(u)
#         for l in lowers:
#             acc_lower_dists_graph.append(l)

#         count += 1

        

#     res_stats[label] = {
#         "analytical": {
#             "std_upper": np.std(acc_upper_dists_analytical),
#             "std_lower": np.std(acc_lower_dists_analytical),
#             "med_upper": np.mean(acc_upper_dists_analytical),
#             "med_lower": np.mean(acc_lower_dists_analytical),
#         },
#         "graphical": {
#             "std_upper": np.std(acc_upper_dists_graph),
#             "std_lower": np.std(acc_lower_dists_graph),
#             "med_upper": np.mean(acc_upper_dists_graph),
#             "med_lower": np.mean(acc_lower_dists_graph),
#         }
#     }

# print('FINAL STATS:'); print(res_stats)
# with open(Path.cwd() / "new_pipeline" / "outputs" / "results_cf.json", "w") as f:
#     json.dump(res_stats, f)

# print('num failures:', failures)

# """

# FINAL STATS:
# {'french':
# {'analytical': {'std_upper': np.float64(15.038595842517928), 'std_lower': np.float64(40.932500769591144), 'med_upper': np.float64(3.6041083334169652), 'med_lower': np.float64(23.42762838166208)}, 'graphical': {'std_upper': np.float64(0.6112681142575472), 'std_lower': np.float64(0.601493668926042), 'med_upper': np.float64(2.067460774600834), 'med_lower': np.float64(2.1065512758165523)}}, 'german': {'analytical': {'std_upper': np.float64(11.319860688927829), 'std_lower': np.float64(39.36587305702989), 'med_upper': np.float64(3.3752960668185987), 'med_lower': np.float64(26.97965098193733)}, 'graphical': {'std_upper': np.float64(0.661772695837032), 'std_lower': np.float64(0.8589359602491553), 'med_upper': np.float64(2.0800457907713983), 'med_lower': np.float64(2.2328105612627014)}}, 'hun': {'analytical': {'std_upper': np.float64(5.148891549205859), 'std_lower': np.float64(21.48823675035574), 'med_upper': np.float64(2.174868762756746), 'med_lower': np.float64(16.423086900808684)}, 'graphical': {'std_upper': np.float64(0.7579048813107487), 'std_lower': np.float64(0.6538593878349904), 'med_upper': np.float64(2.098928130504631), 'med_lower': np.float64(2.1356058968483853)}}, 'fin': {'analytical': {'std_upper': np.float64(4.226999346088061), 'std_lower': np.float64(21.899261919151694), 'med_upper': np.float64(2.1042583594190107), 'med_lower': np.float64(18.234556503378045)}, 'graphical': {'std_upper': np.float64(0.9926301622889305), 'std_lower': np.float64(0.6488635543080212), 'med_upper': np.float64(2.18603916890633), 'med_lower': np.float64(2.1411990502635407)}}, 'spa': {'analytical': {'std_upper': np.float64(4.263883144299232), 'std_lower': np.float64(21.958217229709877), 'med_upper': np.float64(2.144536662472639), 'med_lower': np.float64(18.365363327206033)}, 'graphical': {'std_upper': np.float64(0.9982638219076667), 'std_lower': np.float64(0.648325845248634), 'med_upper': np.float64(2.188319765000826), 'med_lower': np.float64(2.1401648679845136)}}}
# """


# """
# python3 new_pipeline/full_run_both.py
# """


import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

from letter_analysis import *
from funcs import *

num_texts = 1000

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]
lan_labels = ["french", "german", "hun", "fin"]

#FONT_PATH = str(Path.cwd() / "times-new-roman" / "times.ttf")
#FONT_PATH = str(Path.cwd() / "Cormorant" / "Cormorant-VariableFont_wght.ttf")
#FONT_PATH = str(Path.cwd() / "chunkfive" / "ChunkFive-Regular.otf")
FONT_PATH = str(Path.cwd() / "arial" / "ARIAL.TTF")



def filter_text(text, filter_chars):
    return "".join(c for c in text if c not in filter_chars)


def process_one(job):
    """
    Runs in a worker process. Takes one (label, idx, text) job,
    returns a small dict of results (or None on failure) instead of
    mutating any shared state.
    """
    label, idx, text = job

    uppers_a, lowers_a = analyze_text(text)

    img = text_to_img(
        text,
        font_path=FONT_PATH,
        font_size=20,
        line_spacing=10,
    )
    res = compute_scores_att_3(np.array(img), call_idx=idx, debug=True)
    if res is None:
        return {"label": label, "failed": True}

    uppers_g, lowers_g = res
    return {
        "label": label,
        "failed": False,
        "uppers_a": uppers_a,
        "lowers_a": lowers_a,
        "uppers_g": uppers_g,
        "lowers_g": lowers_g,
    }


def main():
    all_texts = get_valid_texts(jsonl_files, num_texts, None)
    print('amount of texts per lan before filter:', [len(x) for x in all_texts])
    all_texts = [
        [filter_text(t, "0123456789,?!;/:+-*=") for t in texts_per_lan]
        for texts_per_lan in all_texts
    ]

    assert len(all_texts) == len(lan_labels), AssertionError(f"lengths: {len(all_texts)}")

    # Flatten into one job list so the pool can balance work across all
    # languages/texts, not just within one language at a time.
    jobs = []
    idx = 0
    for label, texts in zip(lan_labels, all_texts):
        for text in texts:
            jobs.append((label, idx, text))
            idx += 1

    n_workers = cpu_count()  # or a fixed number if you want to leave cores free
    print(f"Dispatching {len(jobs)} jobs across {n_workers} workers")

    with Pool(processes=n_workers) as pool:
        # imap keeps input order; use imap_unordered if you don't care
        # about order and want results as soon as they're ready.
        results = list(pool.imap(process_one, jobs, chunksize=4))

    # --- Aggregate back in the main process ---
    per_lang = {label: {
        "upper_a": [], "lower_a": [], "upper_g": [], "lower_g": []
    } for label in lan_labels}

    failures = 0
    for r in results:
        if r["failed"]:
            failures += 1
            continue
        pl = per_lang[r["label"]]
        pl["upper_a"].extend(r["uppers_a"])
        pl["lower_a"].extend(r["lowers_a"])
        pl["upper_g"].extend(r["uppers_g"])
        pl["lower_g"].extend(r["lowers_g"])

    res_stats = {}
    for label in lan_labels:
        pl = per_lang[label]
        res_stats[label] = {
            "analytical": {
                "std_upper": np.std(pl["upper_a"]),
                "std_lower": np.std(pl["lower_a"]),
                "med_upper": np.mean(pl["upper_a"]),
                "med_lower": np.mean(pl["lower_a"]),
            },
            "graphical": {
                "std_upper": np.std(pl["upper_g"]),
                "std_lower": np.std(pl["lower_g"]),
                "med_upper": np.mean(pl["upper_g"]),
                "med_lower": np.mean(pl["lower_g"]),
            },
        }

    print('FINAL STATS:'); print(res_stats)
    with open(Path.cwd() / "new_pipeline" / "outputs" / "results_tnr.json", "w") as f:
        json.dump(res_stats, f)

    print('num failures:', failures)


if __name__ == "__main__":
    main()


"""
python3 new_pipeline/full_run_both.py
"""