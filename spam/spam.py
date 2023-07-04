# https://archive.ics.uci.edu/dataset/94/spambase

import numpy as np

class Spam():
    name = 'spam'
    n_centers = 2

    @staticmethod
    def read():
        data = np.loadtxt('spam/spambase.data', delimiter=',')
        labels = data[:,-1].astype(np.int64) # separa as labels das features
        data = data[:,:-1].astype(np.float64) # mantém apeans as features para o cálculo de distancias
        return data, labels