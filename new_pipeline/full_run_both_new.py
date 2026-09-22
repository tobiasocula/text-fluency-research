"""
with new text-to-img generation
usage: python3 new_pipeline/full_run_both_new.py --font PATH_TO_FONT [--name LABEL]
"""
import argparse
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


def filter_text(text, filter_chars):
    return "".join(c for c in text if c not in filter_chars)


def process_one(job):
    """
    Runs in a worker process. Takes one (font_path, label, idx, text) job and
    returns a small dict of results instead of mutating shared state.
    """
    font_path, label, idx, text = job   # font path now travels with the job

    font = ImageFont.truetype(font_path, 20)
    lines = wrap_lines(text, max_chars=500)

    uppers_a, lowers_a = [], []
    for line in lines:
        u, l = analyze_text(line)
        uppers_a.extend(u)
        lowers_a.extend(l)

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--font", required=True, help="path to a .ttf/.otf font file")
    p.add_argument("--name", default=None,
                   help="label used in the output filename (default: font file name)")
    p.add_argument("--out-dir", default=str(Path.cwd() / "new_pipeline" / "outputs_new"))
    return p.parse_args()


def main():
    args = parse_args()
    font_path = str(Path(args.font).resolve())
    if not Path(font_path).is_file():
        raise SystemExit(f"Font not found: {font_path}")
    font_name = args.name or Path(font_path).stem
    print(f"Using font: {font_path} (label: {font_name})")

    all_texts = get_valid_texts(jsonl_files, num_texts, None)
    print('amount of texts per lan before filter:', [len(x) for x in all_texts])
    all_texts = [
        [filter_text(t, "0123456789,?!;/:+-*=") for t in texts_per_lan]
        for texts_per_lan in all_texts
    ]

    assert len(all_texts) == len(lan_labels), f"lengths: {len(all_texts)}"

    jobs = []
    idx = 0
    for label, texts in zip(lan_labels, all_texts):
        for text in texts:
            jobs.append((font_path, label, idx, text))
            idx += 1

    n_workers = cpu_count()
    print(f"Dispatching {len(jobs)} jobs across {n_workers} workers")

    with Pool(processes=n_workers) as pool:
        results = list(pool.imap(process_one, jobs, chunksize=4))

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
                "std_upper": float(np.std(pl["upper_a"])),
                "std_lower": float(np.std(pl["lower_a"])),
                "med_upper": float(np.mean(pl["upper_a"])),
                "med_lower": float(np.mean(pl["lower_a"])),
            },
            "graphical": {
                "std_upper": float(np.std(pl["upper_g"])),
                "std_lower": float(np.std(pl["lower_g"])),
                "med_upper": float(np.mean(pl["upper_g"])),
                "med_lower": float(np.mean(pl["lower_g"])),
            },
        }

    print('FINAL STATS:'); print(res_stats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{font_name}.json"   # one file per font
    with open(out_path, "w") as f:
        json.dump(res_stats, f, indent=2)
    print('saved to', out_path)
    print('num failures:', failures)


if __name__ == "__main__":
    main()