from pathlib import Path

p = Path.cwd()/"new_vertical"/"bars"
for dir in p.iterdir():
    s = dir.name.split(".")
    if len(s) == 3:
        dir.rename(s[0] + "." + s[1])
        