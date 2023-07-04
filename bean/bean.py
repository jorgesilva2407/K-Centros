# https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

import pandas as pd
import numpy as np

class Bean():
    classes = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']
    name = 'bean'
    n_centers = 7
    
    @staticmethod
    def read():
        data = pd.read_excel('bean/Dry_Bean_Dataset.xlsx')
        data = data.to_numpy()
        labels = np.vectorize(Bean.sub)(data[:,-1])
        data = data[:,:-1]
        return data, labels
    
    @staticmethod
    def sub(val: str) -> int:
        for i in range(len(Bean.classes)):
            if val == Bean.classes[i]:
                return i