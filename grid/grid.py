# https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data

import pandas as pd
import numpy as np

class Grid():
    name = 'grid'
    n_centers = 2

    @staticmethod
    def read():
        data = pd.read_csv('grid/Data_for_UCI_named.csv')
        data = data.to_numpy()
        labels = np.vectorize(lambda x: 1 if x == 'stable' else 0)(data[:,-1])
        data = data[:,:-1]
        return data, labels