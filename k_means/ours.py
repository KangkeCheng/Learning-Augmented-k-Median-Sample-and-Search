import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances

def Ours(data, oracle_labels, k, alpha, epsilon, repeat, size_subsets, sample_size_subsets):
    # Handle with the sample parameters (这部分逻辑不变)
    if repeat == 0:
        repeat = int(np.ceil(np.log2(k)))
        
    if  sample_size_subsets == 0: # 修正原始代码中可能存在的变量未定义问题
        sample_size = int(np.ceil(1/((1-alpha)*epsilon)))

    if size_subsets == 0:
        size_subsets = int(np.ceil(1/epsilon))
    C = []
    for i in range(k):
        X_i_indices = np.where(oracle_labels == i)[0]
        m_i = X_i_indices.shape[0]
        if m_i == 0:
            continue
        data_i = data[X_i_indices]
        effective_size = min(size_subsets, m_i)
        random_indices_list = [np.random.choice(m_i, effective_size, replace=False) for _ in range(sample_size_subsets)]
        C_prime = np.array([np.mean(data_i[indices], axis=0) for indices in random_indices_list])
        if C_prime.shape[0] == 0:
            C.append(data_i[np.random.choice(m_i)])
            continue
        num_inliers = max(1, int((1 - alpha) * m_i))
        dists = euclidean_distances(data_i, C_prime, squared=True)
        partitioned_dists = np.partition(dists, num_inliers - 1, axis=0)
        nearest_distances = partitioned_dists[:num_inliers, :]
        costs = np.sum(nearest_distances, axis=0)
        best_candidate_index = np.argmin(costs)
        distances_to_best_candidate = dists[:, best_candidate_index]
        inlier_indices = np.argsort(distances_to_best_candidate)[:num_inliers]
        inlier_points = data_i[inlier_indices]
        best_c = np.mean(inlier_points, axis=0)
        C.append(best_c)
        
    return np.array(C)