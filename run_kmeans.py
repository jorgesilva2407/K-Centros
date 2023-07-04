import multiprocessing
from k_centros import Minkowski_dist
from sklearn.cluster import KMeans
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

def get_radius(data, centers, labels):
    radius = 0
    for i in range(data.shape[0]):
        dist = Minkowski_dist(data[i], centers[labels[i]], 2)
        if dist > radius:
            radius = dist
    return radius

def test(item):
    print(f'Iniciando: {item.name}')

    data, labels = item.read()
    data = data.astype(np.float64)
    labels = labels.astype(np.int64)
    counter = 0
    manhattan_stats = []
    while counter < 30:
        try:
            kmeans = KMeans(n_clusters=item.n_centers, n_init='auto').fit(data)
            silhouette = silhouette_score(data, kmeans.labels_)
            rand = adjusted_rand_score(labels, kmeans.labels_)
            radius = get_radius(data, kmeans.cluster_centers_, kmeans.labels_)
            manhattan_stats.append([radius, silhouette, rand])
            counter += 1
        except:
            continue
    np.savetxt(f'{item.name}/{item.name}_results_kmeans.txt', np.array(manhattan_stats), delimiter=',')

    print(f'Finalizado: {item.name}')

if __name__ == '__main__':
    tests = [Androgen(), Banknote(), Bean(), Churn(), Gamma(), Grid(), Letter(), Spam(), Wine(), Yeast()]
    pool = multiprocessing.Pool(4)
    pool.map(test, tests)