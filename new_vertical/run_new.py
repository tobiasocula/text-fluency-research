
import cv2
import numpy as np
from pathlib import Path
import sys
from funcs import *

FONT_PATH = str(Path.cwd() / "arial" / "ARIAL.TTF")
FONT_SIZE = 20
num_texts = 200

jsonl_files = [
    Path.cwd() / "french_texts_random_subset.jsonl",
    Path.cwd() / "german_texts_subset.jsonl",
    Path.cwd() / "hun_texts_subset.jsonl",
    Path.cwd() / "fin_Latn" / "10_1.jsonl",
    #Path.cwd() / "spa.jsonl"
    Path.cwd() / "wiki_nl.jsonl",
    Path.cwd() / "wiki_no.jsonl",
]
lan_labels = ["french", "german", "hun", "fin", "nl", "no"]


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

converted_texts = [
    [remove_diacritics(t) for t in texts_per_lan]
    for texts_per_lan in all_texts]

TEMPLATE_FONT_SIZE = 200

def process_one(job):
    lan,text = job
    img = text_to_image(text, FONT_PATH, FONT_SIZE)
    letters, _ = extract_letters_template(
        img, text,
        letters_dir=Path.cwd() / "new_vertical" / "letters",
        template_font_size=TEMPLATE_FONT_SIZE,
        target_font_size=FONT_SIZE,
    )
    