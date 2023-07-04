# https://archive.ics.uci.edu/dataset/59/letter+recognition

import numpy as np

class Letter():
    name = 'letter'
    n_centers = 26
    
    @staticmethod
    def read():
        data = []
        with open('letter/letter-recognition.data', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip().split(','))
        data = np.array(data)
        labels = np.vectorize(lambda x: ord(x) - ord('A'))(data[:,0])
        data = np.vectorize(float)(data[:,1:])
        return data, labels