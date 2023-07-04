import numpy as np

def z_normalize(data : np.array) -> np.array:
    """Para cada coluna do problema, subtrai desta sua média e em seguida divide o resultado pelo desvio padrao

    Args:
        data (np.array): O problema em questao

    Returns:
        np.array: A matriz z-normalizada
    """
    new_data = data.copy()
    for i in range(data.shape[1]):
        new_data[:,i] = (new_data[:,i] - np.mean(new_data[:,i]))/np.std(new_data[:,i])
    return new_data

def Minkowski_dist(x : np.array, y : np.array, p : int) -> float:
    """Calcula a distância de Minkowski das amostras

    Args:
        x (np.array): ponto
        y (np.array): ponto
        p (int): ordem da distância a ser calculada

    Raises:
        AssertionError: ordem menor do que zero

    Returns:
        float: distancia entre os pontos x e y
    """
    if p < 1:
        raise AssertionError
    diff = x - y
    abs_diff = np.abs(diff)
    p_diff = np.power(abs_diff, p)
    sum_diff = sum(p_diff)
    return np.power(sum_diff, 1/p)

def calc_dists(points : np.array, p : int = 2) -> np.array:
    """Calcula a matriz de distancias do problema

    Args:
        points (np.array): conjunto de pontos a serem analisados
        p (int): ordem da distância a ser calculada

    Returns:
        np.array: matriz de distancias para o problema
    """
    n = points.shape[0] # número de pontos no conjunto
    dists = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if i == j:
                dists[i,i] = 0
            else:
                dists[i,j] = Minkowski_dist(points[i,:], points[j,:], p)
                dists[j,i] = dists[i,j]
    return dists

def k_centers(original_dists : np.array, radius : float, k : int) -> list:
    """Calcula os k-centros que vão agrupar o conjunto em questão

    Args:
        original_dists (np.array): matriz de distâncias dos pontos
        radius (float): raio das esferas que vão partir dos centros
        k (int): número de centros que se deseja encontrar

    Returns:
        list: lista com os centros do conjunto
    """
    dists = original_dists.copy()
    indexes = np.array(list(range(dists.shape[0])))
    centers = set()
    while dists.size > 0:
        if len(centers) >= k:
            return None
        center = np.random.choice(range(dists.shape[0])) # escolhe um centro arbitrário
        centers.add(indexes[center])
        distant = np.where(dists[center, :] > radius)[0].tolist() # encontra os pontos distantes do centro
        dists = dists[distant, :][:, distant] # mantém apenas pontos distantes do centro
        indexes = indexes[distant]
    return centers
    
def fit_k_centers(dists : np.array, k : int) -> tuple:
    """Tenta encontrar o menor raio possível que ainda encontra k-centros

    Args:
        dists (np.array): matriz de distancias do problema em questao
        k (int): numero de centros a serem encontrados

    Returns:
        tuple: o raio encontrado e os centros escolhidos
    """
    min_radius = 0
    max_radius = dists.max()
    centers = k_centers(dists, dists.max(), k)
    while max_radius - min_radius > 0.00001: # realiza uma busca binaria no intervalo [0, max_radius] para tentar encontar o melhor raio
        radius = min_radius + (max_radius - min_radius)/2
        new_centers = k_centers(dists, radius, k)
        if new_centers == None:
            min_radius = radius
        else:
            max_radius = radius
            centers = new_centers
    return max_radius, np.array(list(centers))

def predict_k_centers(instances: np.array, centers: np.array, p: int = 2) -> np.array:
    """Calcula de qual centro uma determinada instância está mais próxima

    Args:
        instances (np.array): Pontos a serem classificados
        centers (np.array): Coordenadas dos centros em questão
        p (int, optional): Ordem da distância a ser calculada. Defaults to 2.

    Returns:
        np.array: Vetor com o índice do centro ao qual cada ponto está associado
    """
    distances = np.zeros((instances.shape[0], centers.shape[0]), dtype=np.float64)
    results = np.zeros(instances.shape[0], dtype=np.int64)
    for i in range(instances.shape[0]):
        for j in range(centers.shape[0]):
            distances[i,j] = Minkowski_dist(instances[i], centers[j], p)
        results[i] = np.argmin(distances[i,:])

    return results