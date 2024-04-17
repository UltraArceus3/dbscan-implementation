import numpy as np

def region_query(X: np.ndarray, P, eps):
    neighbors = []
    for i in range(X.shape[0]):
        if np.linalg.norm(X[i] - P) < eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(X: np.ndarray, P, cl_id, eps, min_pts, cluster_info = None):
    if cluster_info is None:
        cluster_info = np.zeros(X.shape[0])

    seeds = region_query(X, P, eps)
    if len(seeds) < min_pts: # not a core point
        return False
    else: # all points in seeds are density reachable from P
        cluster_info[seeds] = cl_id
        seeds.remove(P)
        while len(seeds) > 0:
            current_point = seeds[0]
            result = region_query(X, current_point, eps)
            
            if len(result) >= min_pts:
                for i in range(len(result)):
                    result_point = result[i]
                    if cluster_info[result_point] in (0, -1): # 0 is undefined, -1 is noise
                        if cluster_info[result_point] == 0:
                            seeds.append(result_point)
                        cluster_info[result_point] = cl_id
            seeds.remove(current_point)
        return True
    

def dbscan(X: np.ndarray, eps, min_pts, cluster_info = None):
    if cluster_info is None:
        cluster_info = np.zeros(X.shape[0])
        
    clusters = []
    noise = []
    for point in range(X.shape[0]):
        if point not in noise:
            cluster = expand_cluster(X, point, [], eps, min_pts)
            if cluster:
                clusters.append(cluster)
            else:
                noise.append(point)
    return clusters, noise

    



if __name__ == "__main__":
    X = np.array([[1, 2], [1, 3], [2, 2], [8, 7], [8, 8], [6,9]])
    eps = 3
    min_pts = 2
    dbscan(X, eps, min_pts)