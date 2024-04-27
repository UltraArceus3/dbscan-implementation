import numpy as np

def region_query(X: np.ndarray, P, eps):
    '''
    region_query: Find all points in X that are within a distance of eps from point P

    Inputs:
    X: np.ndarray: The dataset
    P: np.ndarray: The point to find neighbors for
    eps: float: The radius of the neighborhood

    Returns:
    neighbors: list: A list of indices of the points in X that are in the neighborhood of P
    '''

    neighbors = []
    for i in range(X.shape[0]):
        if np.linalg.norm(X[i] - P) < eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(X: np.ndarray, P, cl_id, eps, min_pts, cluster_info = None, region_query = region_query):
    if cluster_info is None:
        cluster_info = np.zeros(X.shape[0])

    seeds = region_query(X, X[P], eps) # All points within eps distance from P
    if len(seeds) < min_pts: # not a core point
        return False
    else: # all points in seeds are density reachable from P
        cluster_info[seeds] = cl_id
        #print(seeds, P)
        seeds.remove(P)
        while len(seeds) > 0:
            current_point = seeds[0]
            result = region_query(X, X[current_point], eps)
            
            if len(result) >= min_pts:
                for i in range(len(result)):
                    result_point = result[i]
                    if cluster_info[result_point] in (0, -1): # 0 is undefined, -1 is noise
                        if cluster_info[result_point] == 0:
                            seeds.append(result_point)
                        cluster_info[result_point] = cl_id
            seeds.remove(current_point)
        return True
    

def dbscan(X: np.ndarray, eps, min_pts, cluster_info = None, region_query = region_query):
    if cluster_info is None:
        cluster_info = np.zeros(X.shape[0])
        
    NOISE = -1
    UNCLASSIFIED = 0

    cluster_id = UNCLASSIFIED + 1

    for i in range(X.shape[0]):
        point = X[i]
        if cluster_info[i] == UNCLASSIFIED:
            cluster = expand_cluster(X, i, cluster_id, eps, min_pts, cluster_info, region_query)
            if cluster:
                cluster_id += 2 if cluster_id == NOISE else 1
    return cluster_info

    



if __name__ == "__main__":
    X = np.array([[1, 2], [1, 3], [2, 2], [8, 7], [8, 8], [6,9]])
    eps = 3
    min_pts = 2
    print(dbscan(X, eps, min_pts))