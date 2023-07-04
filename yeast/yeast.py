# https://archive.ics.uci.edu/dataset/110/yeast

import numpy as np

class Yeast():
    name = 'yeast'
    classes = ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL']
    n_centers = 10
    
    @staticmethod
    def read():
        data = []
        with open('yeast/yeast.data', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip().split()[1:])
        data = np.array(data)
        labels = np.vectorize(Yeast.sub)(data[:,-1])
        data = np.vectorize(float)(data[:,1:-1])
        print(data)
        print(labels)
    
    @staticmethod
    def sub(val: str) -> int:
        for i in range(len(Yeast.classes)):
            if val == Yeast.classes[i]:
                return i