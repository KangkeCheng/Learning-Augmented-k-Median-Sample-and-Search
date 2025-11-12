import numpy as np
import math
import time
import itertools
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs
from geom_median.numpy import compute_geometric_median
from utils import *




def fast_sampling_kmedian_literal_grid(points, oracle_labels,sample_size,num_sampled_levels,k, alpha, epsilon=0.1,max_step=0):
    n, d = points.shape
    centers = np.zeros((k, d))
    
    for i in range(k):
        points_i_indices = np.where(oracle_labels == i)[0]
        points_i = points[points_i_indices]
        m_i = points_i.shape[0]
        if(sample_size==0):
            sample_size = int(min(m_i, max(10, 5 * (math.log(k*d if k*d>1 else 2) / (1-2*alpha if 1-2*alpha>0.01 else 0.01)))))
        U_i = points_i[np.random.choice(m_i, size=min(sample_size,m_i), replace=False)]
        candidate_centers_list = [u for u in U_i]

        min_coords, max_coords = points_i.min(axis=0), points_i.max(axis=0)
        diameter = euclidean_distances([min_coords], [max_coords])[0, 0]
        diameter = max(diameter, 1e-6)
        
        max_levels = math.ceil(math.log2(m_i * diameter) if m_i * diameter > 1 else 1)
        if num_sampled_levels==0:
           num_sampled_levels=max_levels
        actual_num_to_sample = min(num_sampled_levels, max_levels)
        
        if max_levels > 0:
    
            sampled_levels = np.random.choice(max_levels, size=actual_num_to_sample, replace=False)
        else:
            sampled_levels = []

        for q in sampled_levels:
            current_scale = diameter / (2**q)
            

            new_candidates = []
      
            step_size = (1 - alpha) * (epsilon / 8.0) * current_scale / math.sqrt(d)
            step_size = (1 - alpha) * (epsilon / 8.0) * current_scale / np.sqrt(d)
            if step_size < 1e-9:
                return np.array([]) 

            num_initial_points = U_i.shape[0]
            repeated_U = np.repeat(U_i, d * max_step * 2, axis=0)
            steps = np.arange(1, max_step + 1)
            identity_matrix = np.eye(d)
            base_shifts = identity_matrix * step_size
            step_shifts = np.outer(steps, base_shifts).reshape(max_step * d, d)
            signed_shifts = np.repeat(step_shifts, 2, axis=0)
            signed_shifts[1::2] *= -1 
            final_displacements = np.tile(signed_shifts, (num_initial_points, 1))
            new_candidates = repeated_U + final_displacements
            candidate_centers_list.extend(new_candidates)
        U_prime_i = np.unique(np.array(candidate_centers_list), axis=0)


  
        min_cost = float('inf')
        best_center = U_prime_i[0]
        num_inliers = max(1, int((1 - alpha) * m_i))
        m_i = points_i.shape[0]
        if len(U_prime_i) == 0:
            return None
            
        num_inliers = max(1, int((1 - alpha) * m_i))
        dists = euclidean_distances(points_i, U_prime_i)
        partitioned_dists = np.partition(dists, num_inliers - 1, axis=0)
        nearest_distances = partitioned_dists[:num_inliers, :]
        costs = np.sum(nearest_distances, axis=0)
        best_candidate_index = np.argmin(costs)
        distances_to_best_candidate = dists[:, best_candidate_index]
        inlier_indices = np.argsort(distances_to_best_candidate)[:num_inliers]
        inlier_points = points_i[inlier_indices]
        if len(inlier_points) == 0:
            best_center = U_prime_i[best_candidate_index]
        else:
            best_center = compute_geometric_median(inlier_points).median
        centers[i] = best_center
    return centers

def huang(X,y_noisy,k,alpha,epsilon,sample_size,num_sampled_levels,max_step):
    
    start_time = time.time()
    # 调用新的、实现完整网格的函数
    calculated_centers = fast_sampling_kmedian_literal_grid(X,y_noisy,sample_size,num_sampled_levels,k,alpha,epsilon,max_step)
    end_time = time.time()
    
    labels, algorithm_cost =k_medians_cost(X, calculated_centers)
    return labels,algorithm_cost,(end_time-start_time)
    

