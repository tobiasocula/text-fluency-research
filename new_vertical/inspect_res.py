import cv2
import numpy as np
from pathlib import Path
import sys
import json

output = Path.cwd() / "new_vertical"

with open(output / "results.json", "r") as f:
    fdata = json.load(f)

for lan,content in fdata.items():
    print(f"---------{lan}---------")
    for field,val in content.items():
        print(field, val)

"""
python3 new_vertical/inspect_res.py
"""