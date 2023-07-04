# https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset

import pandas as pd

class Churn():
    name = 'churn'
    n_centers = 2

    @staticmethod
    def read():
        data = pd.read_csv('churn/Customer Churn.csv')
        data = data.to_numpy()
        labels = data[:,-1]
        data = data[:,:-1]
        print(labels)
        return data, labels