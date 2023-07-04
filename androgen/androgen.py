# https://archive.ics.uci.edu/dataset/509/qsar+androgen+receptor

import numpy as np

class Androgen():
    name = 'androgen'
    n_centers = 2
    
    @staticmethod
    def read():
        data = []
        with open('androgen/qsar_androgen_receptor.csv', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip().split(';'))
        data = np.array(data)
        labels = np.vectorize(lambda x: 1 if x == 'positivo' else 0)(data[:,-1])
        data = np.vectorize(float)(data[:,:-1])
        return data, labels