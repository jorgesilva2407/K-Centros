from k_centros import z_normalize, calc_dists, fit_k_centers, predict_k_centers
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
from spam.spam import Spam
from wine.wine import Wine

# tests = [Spam(), Wine()]
tests = [Wine()]

for test in tests:
    data, labels = test.read()
    counter = 0
    manhattan_stats = []
    manhattan_dists = calc_dists(data, 1)
    while counter < 30:
        try:
            radius, centers_indexes = fit_k_centers(manhattan_dists, 2)
            centers = data[centers_indexes]
            pred_labels = predict_k_centers(data, centers, 1)
            silhouette = silhouette_score(data, pred_labels, metric='manhattan')
            rand = adjusted_rand_score(labels, pred_labels)
            manhattan_stats.append([radius, silhouette, rand])
            counter += 1
        except:
            continue

    np.savetxt(f'{test.name}/results_{test.name}1.txt', np.array(manhattan_stats), delimiter=',')

    counter = 0
    euclidean_stats = []
    euclidean_dists = calc_dists(data, 1)
    while counter < 30:
        try:
            radius, centers_indexes = fit_k_centers(euclidean_dists, 2)
            centers = data[centers_indexes]
            pred_labels = predict_k_centers(data, centers, 2)
            silhouette = silhouette_score(data, pred_labels, metric='euclidean')
            rand = adjusted_rand_score(labels, pred_labels)
            euclidean_stats.append([radius, silhouette, rand])
            counter += 1
        except:
            continue

    np.savetxt(f'{test.name}/results_{test.name}2.txt', np.array(euclidean_stats), delimiter=',')