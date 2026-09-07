import zstandard as zstd
from pathlib import Path
import random

jsonl_files = [
    #Path.cwd() / "french_texts_random_subset.jsonl",
    #Path.cwd() / "german_texts_subset.jsonl",
    #Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
]

from new_vertical.funcs import get_valid_texts

lan_texts = get_valid_texts(jsonl_files)
print('lengths:')

print([len(x) for x in lan_texts])