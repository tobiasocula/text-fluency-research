import json
from letter_analysis import *
from pathlib import Path
from new_vertical.funcs import *

output = Path.cwd() / "new_pipeline" / "outputs"

a = 3
b = 5
with open(output / f"lastresults_{a}.json", "r") as f:
    fdata_a = json.load(f)
with open(output / f"lastresults_{b}.json", "r") as f:
    fdata_b = json.load(f)


def construct_arr(fdata):

    stats_an = np.empty((len(fdata), 4)) # std upper, std lower, etc (sums)
    stats_gr = np.empty((len(fdata), 4)) # std upper, std lower, etc (sums)
    for i,(lan,data) in enumerate(fdata.items()):
        for mode,modedata in data.items():
            print(modedata.values())
            if mode == "analytical":
                stats_an[i,:] = list(modedata.values())
            else:
                stats_gr[i,:] = list(modedata.values())

    mean_an = np.mean(stats_an, axis=0)
    mean_gr = np.mean(stats_gr, axis=0)
    std_an = np.std(stats_an, axis=0)
    std_gr = np.std(stats_gr, axis=0)

    lan_labels = ["french", "german", "hun", "fin", "spa"]
    table = np.empty((2,4,len(lan_labels)))
    for i in range(2):
        for j in range(4):
            for l in range(len(lan_labels)):
                if i == 0:
                    table[i,j,l] = (stats_an[l, j] - mean_an[j]) / std_an[j]
                else:
                    table[i,j,l] = (stats_gr[l, j] - mean_gr[j]) / std_gr[j]

    return table

lan_labels = ["french", "german", "hun", "fin", "spa"]
table_a, table_b = construct_arr(fdata_a), construct_arr(fdata_b)
err =  (table_a - table_b)**2
for i,label in enumerate(lan_labels):
    print(label, ':')
    print(err[:,:,i])