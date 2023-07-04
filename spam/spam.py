# https://archive.ics.uci.edu/dataset/94/spambase

from k_centros import z_normalize
import numpy as np

class Spam():
    def __init__(self) -> None:
        name = 'spam'

    @staticmethod
    def read():
        data = np.loadtxt('spam/spambase.data', delimiter=',')
        labels = data[:,-1].astype(np.int64) # separa as labels das features
        data = data[:,:-1].astype(np.float64) # mantém apeans as features para o cálculo de distancias
        data = z_normalize(data)
        print(data)
        return data, labels