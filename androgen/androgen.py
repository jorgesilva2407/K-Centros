# https://archive.ics.uci.edu/dataset/509/qsar+androgen+receptor

import numpy as np

data = []

with open('androgen/qsar_androgen_receptor.csv', 'r') as f:
    lines = f.readlines()

    for line in lines:
        data.append(line.strip().split(';'))

data = np.array(data)
print(data)