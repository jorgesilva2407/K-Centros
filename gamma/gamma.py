# https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

import numpy as np

data = []

with open('gamma/magic04.data', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.strip().split(','))

data = np.array(data)
print(data)