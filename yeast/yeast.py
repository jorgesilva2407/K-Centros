# https://archive.ics.uci.edu/dataset/110/yeast

import numpy as np

data = []
with open('yeast/yeast.data', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.strip().split()[1:])

data = np.array(data)
print(data)