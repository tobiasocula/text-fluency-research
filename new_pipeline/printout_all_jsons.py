from pathlib import Path
import json
import numpy as np

fonts = ["arial","chunkfive","cormorant","times"]
fontpaths = [Path.cwd()/"new_pipeline"/"outputs_new"/f"results_{x}.json"
             for x in fonts]
a = np.zeros((4,4,4)) # lan,font,stat
lans = ["french","german","hun","fin"]
stats = ["std_upper","std_lower","med_upper","med_lower"]
for i,fp in enumerate(fontpaths):
    with open(fp, "r") as f:
        file = json.load(f)
    for j,lan in enumerate(lans):
        for k,stat in enumerate(stats):
            a[j,i,k] = file[lan]["graphical"][stat]

for i,lan in enumerate(lans):
    print(lan)
    print(a[i,:,:])
    print()

"""
H: font
V: stat

french
[[ 28.79518179 174.23242426  27.43688781 172.58343421]
 [ 32.14809648 160.75886187  22.40485079  92.32921462]
 [ 22.60734977 113.61689232  18.35555851  51.19272228]
 [ 24.71670214 154.46926951  23.08275955 124.7218261 ]]

german
[[ 22.37161271 241.66408928  21.10684703 208.66489018]
 [ 30.05059672 260.90390767  23.40747308 232.50173952]
 [ 17.56135289 145.7075918   15.27747755  58.23143523]
 [ 18.73470036 200.40131687  17.09730738 123.36721359]]

hun
[[ 21.83449846 140.07234055  18.6076653  119.76635729]
 [ 31.42431023  96.99123734  15.12536981  32.19267658]
 [ 13.87751064  53.61255308   7.43356135  11.03162233]
 [ 17.85295329 121.1653543   15.3701621   79.11220745]]

fin
[[ 21.49868332 180.61670216  17.40922151 163.67231989]
 [ 36.88206867 193.49394599  25.1427079  170.21357882]
 [ 15.93107772 136.56871558  13.01112468  51.93948428]
 [ 17.2347044  155.87404586  14.50064502 108.08894417]]
 """