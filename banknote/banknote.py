# https://archive.ics.uci.edu/dataset/267/banknote+authentication

import numpy as np

data = np.loadtxt('banknote/data_banknote_authentication.txt', delimiter=',')

print(data)