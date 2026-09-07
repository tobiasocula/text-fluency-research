
import json
from letter_analysis import *
from pathlib import Path
from new_vertical.funcs import *
import sys
import cv2

lan_labels = ["french", "german", "hun", "fin", "spa"]
texts = [
    "J’ai vu un poney et un singe dans le jardin",  # French: j, p, y, g
    "Der Junge pflückt eine Qualle am Strand",      # German: j, p, q, l
    "A pácban jártam és láttam egy gémet",           # Hungarian: p, j, g
    "Pojalla on kypärä ja hän juoksee",             # Finnish: p, j, y
    "El niño juega con un yoyó y un globo"          # Spanish: j, y, g, b
]
imgs = [
    text_to_img(text,
    font_path=str(Path.cwd() / "times-new-roman" / "times.ttf"),
    font_size=50,
    line_spacing=10
    ) for text in texts
]

results = np.empty((2, len(texts), 4))
uppers_an = []
uppers_gr = []
lowers_an = []
lowers_gr = []
for i,(img,text) in enumerate(zip(imgs, texts)):
    print('LANGUAGE:', lan_labels[i])

    u, l = compute_scores_att_2(np.array(img), call_idx=i, debug=True)
    print('u,l:', u, l)
    uppers_gr.append(u)
    lowers_gr.append(l)
    u2, l2 = analyze_text(text)
    print('u,l:', u2, l2)
    print('compare lengths:', len(u), 'and', len(l), 'vs', len(u2), 'and', len(l2))
    uppers_an.append(u2)
    lowers_an.append(l2)

stats = np.empty((2, len(texts), 4))
for i in range(len(texts)):
    stats[0,i,:] = [np.mean(uppers_an[i]), np.mean(lowers_an[i]), np.std(uppers_an[i]), np.std(lowers_an[i])]
    stats[1,i,:] = [np.mean(uppers_gr[i]), np.mean(lowers_gr[i]), np.std(uppers_gr[i]), np.std(lowers_gr[i])]

mean_stats = np.mean(stats, axis=1) # (2,4)
std_stats = np.std(stats, axis=1) # (2,4)
print('std:', std_stats)
print('mean:', mean_stats)

z = np.zeros_like(stats)
z[0,:,:] = (stats[0] - mean_stats[0]) / std_stats[0]
z[1,:,:] = (stats[1] - mean_stats[1]) / std_stats[1]
print('start')
for i in range(2):
    print()
    print(z[i,:,:])