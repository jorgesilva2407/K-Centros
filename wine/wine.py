# https://archive.ics.uci.edu/dataset/186/wine+quality

import pandas as pd

class Wine():
    name = 'wine'
    n_centers = 11

    @staticmethod
    def read():
        df1 = pd.read_csv('wine/winequality-red.csv', sep=';')
        df2 = pd.read_csv('wine/winequality-white.csv', sep=';')
        df = pd.concat([df1,df2], axis=0)
        df = df.drop_duplicates()
        df = df.dropna()
        labels = df['quality']
        data = df.loc[:, df.columns != 'quality']
        return data.to_numpy(), labels.to_numpy()