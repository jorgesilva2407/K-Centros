import multiprocessing
from k_centros import z_normalize, calc_dists, fit_k_centers, predict_k_centers
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import numpy as np
from spam.spam import Spam
from wine.wine import Wine
from androgen.androgen import Androgen
from banknote.banknote import Banknote
from bean.bean import Bean
from churn.churn import Churn
from gamma.gamma import Gamma
from grid.grid import Grid
from letter.letter import Letter
from yeast.yeast import Yeast

def test(item):
    print(f'Iniciando: {item.name}')
    data, labels = item.read()
    data = z_normalize(data)
    counter = 0
    manhattan_stats = []
    try:
        manhattan_dists = np.loadtxt(f'{item.name}/{item.name}_manhattan.txt', delimiter=',')
    except:
        manhattan_dists = calc_dists(data, 1)
        np.savetxt(f'{item.name}/{item.name}_manhattan.txt', manhattan_dists, delimiter=',')
    while counter < 30:
        try:
            radius, centers_indexes = fit_k_centers(manhattan_dists, item.n_centers)
            centers = data[centers_indexes]
            pred_labels = predict_k_centers(data, centers, 1)
            silhouette = silhouette_score(data, pred_labels, metric='manhattan')
            rand = adjusted_rand_score(labels, pred_labels)
            manhattan_stats.append([radius, silhouette, rand])
            counter += 1
        except:
            continue
    np.savetxt(f'{item.name}/results_{item.name}1.txt', np.array(manhattan_stats), delimiter=',')

    counter = 0
    euclidean_stats = []
    try:
        euclidean_dists = np.loadtxt(f'{item.name}/{item.name}_euclidean.txt', delimiter=',')
    except:
        euclidean_dists = calc_dists(data, 1)
        np.savetxt(f'{item.name}/{item.name}_euclidean.txt', euclidean_dists, delimiter=',')
    while counter < 30:
        try:
            radius, centers_indexes = fit_k_centers(euclidean_dists, item.n_centers)
            centers = data[centers_indexes]
            pred_labels = predict_k_centers(data, centers, 2)
            silhouette = silhouette_score(data, pred_labels, metric='euclidean')
            rand = adjusted_rand_score(labels, pred_labels)
            euclidean_stats.append([radius, silhouette, rand])
            counter += 1
        except:
            continue
    np.savetxt(f'{item.name}/{item.name}_results_2.txt', np.array(euclidean_stats), delimiter=',')

    print(f'Finalizado: {item.name}')

if __name__ == '__main__':
    tests = [Androgen(), Banknote(), Bean(), Churn(), Gamma(), Grid(), Letter(), Spam(), Wine(), Yeast()]
    pool = multiprocessing.Pool(2)
    pool.map(test, tests)
