# https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

import numpy as np

class Gamma():
    name = 'gamma'
    n_centers = 2

    @staticmethod
    def read():
        data = []
        with open('gamma/magic04.data', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip().split(','))
        data = np.array(data)
        labels = np.vectorize(lambda x: 1 if x == 'g' else 0)(data[:,-1])
        data = np.vectorize(float)(data[:,:-1])
        return data, labels