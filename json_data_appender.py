"""
Data appender
I just used this to append into the .json params
Holds no function.
"""

import os
import pandas as pd
import json


LOADPATH = f"{os.getcwd()}/PlayerParams/A_parameters.json"
parameters = None

LOADPATH_CSV = f"{os.getcwd()}/dfThrowing.csv"

with open(LOADPATH) as g:
    parameters = json.load(g)

rawdata = pd.read_csv(LOADPATH_CSV).fillna(0).to_numpy()

newdata = []
for i in range(4):
    newdata.append(rawdata[:, i].tolist())

for i in range(len(newdata)):
    for j in range(len(newdata[i])):
        newdata[i][j] = int(newdata[i][j])

counter = 0
for i in ["A", "K", "M", "T"]:
    print(f"{os.getcwd()}/PlayerParams/{i}_parameters.json")
    load = f"{os.getcwd()}/PlayerParams/{i}_parameters.json"
    with open(load) as f:
        olddata = json.load(f)

        olddata.append([newdata[counter]])

    with open(load, "w+") as g:
        json.dump(olddata, g)
        counter += 1
