import json, random
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

def sample_wikipedia_gb(lang_code, out_path, budget_gb, dump="20231101", seed=42):
    fs = HfFileSystem()
    repo_dir = f"datasets/wikimedia/wikipedia/{dump}.{lang_code}"
    shards = sorted(f["name"] for f in fs.ls(repo_dir, detail=True)
                    if f["name"].endswith(".parquet"))

    # 1) Read only the parquet footers (tiny) to list every row group and its size
    groups = []  # (shard_path, row_group_index, compressed_bytes)
    for path in shards:
        with fs.open(path, "rb") as f:
            md = pq.ParquetFile(f).metadata
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                size = sum(rg.column(c).total_compressed_size
                           for c in range(rg.num_columns))
                groups.append((path, i, size))

    # 2) Randomly pick row groups until the byte budget is reached
    random.Random(seed).shuffle(groups)
    budget = budget_gb * 1024**3
    chosen, total = [], 0
    for g in groups:
        if total >= budget:
            break
        chosen.append(g)
        total += g[2]
    print(f"Selected {len(chosen)} of {len(groups)} row groups (~{total/1024**3:.2f} GB)")

    # 3) Download just those row groups and write JSONL
    chosen.sort(key=lambda g: (g[0], g[1]))  # group by file so each file opens once
    rng = random.Random(seed)
    with open(out_path, "w", encoding="utf-8") as out:
        current_path, pf, fh = None, None, None
        for path, idx, _ in chosen:
            if path != current_path:
                if fh: fh.close()
                fh = fs.open(path, "rb")
                pf = pq.ParquetFile(fh)
                current_path = path
            table = pf.read_row_group(idx, columns=["title", "text"])
            rows = table.to_pylist()
            rng.shuffle(rows)
            for r in rows:
                out.write(json.dumps({"text": r["text"], "title": r["title"]},
                                     ensure_ascii=False) + "\n")
        if fh: fh.close()

sample_wikipedia_gb("no", "wiki_no.jsonl", budget_gb=5.0)

"""
python3 download_languages.py
"""