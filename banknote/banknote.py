# https://archive.ics.uci.edu/dataset/267/banknote+authentication

import numpy as np

class Banknote():
    name = 'banknote'
    n_centers = 2

    @staticmethod
    def read():
        data = np.loadtxt('banknote/data_banknote_authentication.txt', delimiter=',')
        labels = data[:,-1]
        data = data[:,:-1]
        return data, labels