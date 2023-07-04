import pandas as pd

class Wine():
    def __init__(self) -> None:
        name = 'spam'

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