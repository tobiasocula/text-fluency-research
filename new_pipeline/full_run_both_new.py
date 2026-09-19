"""
with new text-to-img generation
"""
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
FONT_PATH = str(Path.cwd() / "Cormorant" / "Cormorant-VariableFont_wght.ttf")
#FONT_PATH = str(Path.cwd() / "chunkfive" / "ChunkFive-Regular.otf")
#FONT_PATH = str(Path.cwd() / "arial" / "ARIAL.TTF")



def filter_text(text, filter_chars):
    return "".join(c for c in text if c not in filter_chars)

def process_one(job):
    """
    Runs in a worker process. Takes one (label, idx, text) job and
    returns a small dict of results instead of mutating shared state.
    """
    label, idx, text = job

    font = ImageFont.truetype(FONT_PATH, 20)
    lines = wrap_lines(text, font, max_width=5000)

    # analytical: per wrapped line
    uppers_a, lowers_a = [], []
    for line in lines:
        u, l = analyze_text(line)
        uppers_a.extend(u)
        lowers_a.extend(l)

    # graphical: same lines rendered to an image
    img = lines_to_img(lines, font)
    res = compute_scores_att_3(np.array(img), call_idx=idx, debug=False)
    if res is None:
        return {"label": label, "failed": True}

    uppers_g_all, lowers_g_all = res

    return {
        "label": label,
        "failed": False,
        "uppers_a": uppers_a,
        "lowers_a": lowers_a,
        "uppers_g": uppers_g_all,
        "lowers_g": lowers_g_all,
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
    with open(Path.cwd() / "new_pipeline" / "outputs_new" / "results_co.json", "w") as f:
        json.dump(res_stats, f)

    print('num failures:', failures)


if __name__ == "__main__":
    main()


"""
python3 new_pipeline/full_run_both_new.py
"""
