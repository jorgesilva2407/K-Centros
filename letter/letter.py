# https://archive.ics.uci.edu/dataset/59/letter+recognition

import numpy as np

data = []

with open('letter/letter-recognition.data', 'r') as f:
    lines = f.readlines()

    for line in lines:
        data.append(line.strip().split(','))

data = np.array(data)
print(data)