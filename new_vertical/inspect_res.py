import cv2
import numpy as np
from pathlib import Path
import sys
import json

output = Path.cwd() / "new_vertical"

with open(output / "results.json", "r") as f:
    fdata = json.load(f)

for k,v in fdata.items():
    print(k)
    print(v)
    print()

"""
french
{'std': 1.52542242682049, 'mean': 4.7555555555555555}

german
{'std': 1.6777274756201983, 'mean': 4.332926829268293}

hun
{'std': 1.2030342781067314, 'mean': 5.008771929824562}

fin
{'std': 1.223057991622277, 'mean': 4.978102189781022}

spa
{'std': 2.3446927171527445, 'mean': 4.713780918727915}
"""