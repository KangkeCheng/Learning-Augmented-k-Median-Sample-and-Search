import numpy as np
from sklearn.cluster import kmeans_plusplus as kpp
from utils import * # Assumes utils.py contains helper functions like euclidean_distances, compute_geometric_median, k_medians_cost
import argparse
from itertools import combinations
import time
import tqdm

def median_candidate(D, t, alpha, epsilon, max_step=2, step_rate=1):
    """
    Generates a set of candidate median points by exploring the space around a given set of points D.

    Args:
        D (np.array): A subset of points, typically from a single cluster.
        t (float): A scaling factor for the search radius.
        alpha (float): The noise level.
        epsilon (float): An error parameter.
        max_step (int): The number of steps to take in each direction.
        step_rate (int): A multiplier for the step size.

    Returns:
        np.array: An array of candidate points, including the original points and the new generated points.
    """
    # Get the dimension of the data points.
    dim = D.shape[1]

    # Calculate the step size 'l' for moving along direction vectors.
    # This determines how far to move from an original point to generate a new candidate.
    l = step_rate * alpha * epsilon * t / (10 * D.shape[0])

    # If max_step is 0, no new candidates are generated, so return the original points.
    if max_step == 0:
        return D

    # Normalize the points in D to get unit direction vectors.
    # This handles potential division by zero if a point is the origin.
    l2_norms = np.linalg.norm(D, ord=2, axis=1, keepdims=True)
    D_unit = np.where(l2_norms > 0, D / l2_norms, 0)

    # --- Vectorized generation of candidate points ---
    num_points, dim = D.shape
    num_directions = D_unit.shape[0]

    # Repeat the original points matrix to align with the generated displacements.
    repeated_D = np.repeat(D, num_directions * max_step, axis=0)

    # Create a block of direction vectors and tile it for all original points.
    directions_block = np.repeat(D_unit, max_step, axis=0)
    repeated_D_unit = np.tile(directions_block, (num_points, 1))

    # Create an array of step multipliers (1, 2, ..., max_step).
    steps = np.arange(1, max_step + 1)
    tiled_steps = np.tile(steps, num_points * num_directions)

    # Calculate the displacement vectors.
    displacements = tiled_steps.reshape(-1, 1) * (repeated_D_unit * l)

    # Generate new points by adding displacements to the original points.
    displaced_points = repeated_D + displacements

    # Combine the original points with the newly generated displaced points.
    result = np.concatenate((D, displaced_points), axis=0)
    return result


def Ours(points, oracle_labels, k, alpha, epsilon=0.1, max_step=0, repeat=0, sample_size_p=100, sample_size_r=0, sample_size_subsets=10, step_rate=1):
    """
    A robust clustering algorithm to find k centers from a dataset with noisy labels.
    For each cluster identified by the noisy 'oracle_labels', it finds a robust center
    by generating candidates and selecting the one that minimizes the k-medians cost for inliers.

    Args:
        points (np.array): The full dataset of points.
        oracle_labels (np.array): The noisy cluster labels for each point.
        k (int): The number of clusters.
        alpha (float): The estimated noise level (fraction of outliers).
        epsilon (float): Error tolerance parameter.
        max_step (int): Max steps for candidate generation in median_candidate.
        repeat (int): Number of repetitions for the main loop for each cluster.
        sample_size_p (int): Sample size for estimating the search range.
        sample_size_r (int): Sample size of points from a cluster to generate candidates.
        sample_size_subsets (int): Number of subsets to sample for range estimation.
        step_rate (int): Step rate multiplier for candidate generation.

    Returns:
        tuple: A tuple containing:
               - The final cluster labels for all points.
               - The final k-medians cost.
               - The total execution time.
    """
    # --- Parameter Initialization ---
    # If 'repeat' is not set, default it based on k.
    if repeat == 0:
        repeat = int(np.ceil(np.log2(k)))

    # If 'sample_size_r' is not set, default it based on alpha and epsilon.
    if sample_size_r == 0:
        sample_size_r = int(np.ceil(np.log2(1 / (alpha * epsilon)) / ((1 - alpha) * (alpha * epsilon))**3))

    size_subsets = int(np.ceil(1 / epsilon))
    start_time = time.time()

    C = [] # This list will store the final k centers.

    # --- Main Loop: Find one center for each cluster ---
    for i in range(k):
        C_prime_list = [] # List to store candidate centers for the current cluster.
        X_i = np.where(oracle_labels == i)[0] # Get indices of points in the current cluster i.
        points_i = points[X_i] # Get the actual points for cluster i.

        # Inner loop for repeated trials to ensure robustness.
        for j in range(repeat):
            # Randomly sample a subset R of points from the current cluster.
            R_indices = np.random.choice(X_i.shape[0], size=min([sample_size_r, X_i.shape[0]]), replace=False)
            points_R = points_i[R_indices]
            
            # If max_step is 0, candidate generation is skipped; use the sample R directly.
            if max_step == 0:
                C_prime_list.append(points_R)
            else:
                # --- Estimate the search range [a_, b_] for the parameter 't' ---
                a_ = float('inf')
                b_ = 0
                for p_prime_ind in range(sample_size_subsets):
                    # Sample points to estimate the range of costs.
                    p_indices = np.random.choice(X_i.shape[0], size=sample_size_p, replace=True)
                    P_prime_indices = np.random.choice(X_i.shape[0], min([size_subsets, X_i.shape[0]]), replace=False)
                    
                    # Calculate sum of distances to estimate cost variance.
                    dist = euclidean_distances(points_i[P_prime_indices], points_i[p_indices])
                    v = np.sum(dist, axis=0)
                    a = np.min(v) * epsilon * epsilon / (P_prime_indices.shape[0])
                    b = np.max(v) / epsilon

                    # Update the tightest bounds for a and b.
                    if a < a_: a_ = a
                    if b > b_: b_ = b
                
                # --- Generate Candidates using the estimated range ---
                # Iterate through powers of 2 for 't' within the estimated range.
                for u in range(int(np.floor(np.log2(a_))), int(np.ceil(np.log2(b_)))):
                    t = 2 ** u
                    # Generate candidate points for this value of t and add to the list.
                    S = median_candidate(points_R, t, alpha, epsilon, max_step, step_rate)
                    C_prime_list.append(S)

        # --- Select the Best Center from Candidates ---
        # Consolidate all generated candidates for cluster i into a single array.
        C_prime = np.concatenate(C_prime_list, axis=0) if C_prime_list else np.array([])

        if C_prime is None or len(C_prime) == 0:
            # If no candidates were generated, skip to the next cluster.
            # (This might happen if a cluster is empty).
            continue

        # Calculate the number of expected inliers in the cluster.
        m_i = X_i.shape[0]
        num_inliers = max(1, int(np.floor((1 - alpha) * m_i)))

        # Calculate distances from all points in cluster i to all candidate centers.
        dists = euclidean_distances(points_i, C_prime)

        # For each candidate, find the sum of distances to its 'num_inliers' closest points.
        # This is the cost of the candidate.
        partitioned_dists = np.partition(dists, num_inliers - 1, axis=0)
        nearest_distances = partitioned_dists[:num_inliers, :]
        costs = np.sum(nearest_distances, axis=0)

        # Find the candidate center with the minimum cost.
        best_candidate_index = np.argmin(costs)

        # --- Refine the Best Center ---
        # Identify the inlier points corresponding to the best candidate.
        distances_to_best_candidate = dists[:, best_candidate_index]
        inlier_indices = np.argsort(distances_to_best_candidate)[:num_inliers]
        inlier_points = points_i[inlier_indices]
        if len(inlier_points) == 0:
            best_c = C_prime[best_candidate_index]
        else:
            best_c = compute_geometric_median(inlier_points).median

        C.append(best_c)

    end_time = time.time()
    
    # After finding all k centers, assign all points to the nearest center and calculate the final cost.
    label, Cost = k_medians_cost(points, np.array(C))
    
    return label, Cost, (end_time - start_time)
