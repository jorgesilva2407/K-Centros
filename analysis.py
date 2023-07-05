import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import os

tests = ['gamma', 'spam', 'androgen', 'bean', 'letter', 'wine', 'banknote', 'yeast', 'churn', 'grid']

df = pd.DataFrame()

array = np.zeros((10,6))
for i in range(10):
    results = np.loadtxt(f'{tests[i]}/{tests[i]}_results_kmeans.txt', delimiter=',')
    array[i, 0] = np.mean(results[:,0])
    array[i, 1] = np.std(results[:,0])
    array[i, 2] = np.mean(results[:,1])
    array[i, 3] = np.std(results[:,1])
    array[i, 4] = np.mean(results[:,2])
    array[i, 5] = np.std(results[:,2])

cols = ['radius_mean', 'radius_std', 'silhouette_mean', 'silhouette_std', 'rand_mean', 'rand_std']
df = pd.DataFrame(array, columns=cols, index=tests)

dfi.export(df, 'results_kmeans.png')