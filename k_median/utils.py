
import numpy as np
import math
import time
import itertools
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs
from geom_median.numpy import compute_geometric_median
import utils



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def label_cost(points,labels,k):
    _, d = points.shape
    center = np.zeros((k, d))
    good_indices = []
    for i in range(k):
        to_index = np.where(labels == i)[0]
        if len(to_index) > 0:
            center[i,:] = compute_geometric_median(points[to_index]).median
            good_indices.append(i)
    if not good_indices: return None, float('inf')
    return k_medians_cost_label(points, center[good_indices,:])

def k_medians_cost_label(points, centers):
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    distance = euclidean_distances(points, centers)
    labels = np.argmin(distance, axis=1)
    return labels, np.min(distance, axis=1).sum()
def k_medians_cost_nonlabel(points, centers):
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    distance = euclidean_distances(points, centers)
    return np.min(distance, axis=1).sum()
def k_medians_cost(points,centers):
    k=centers.shape[0]
    _, d = points.shape
    center = np.zeros((k, d))
    good_indices = []
    labels,_=k_medians_cost_label(points, centers)
    for i in range(k):
        to_index = np.where(labels == i)[0]
        if len(to_index) > 0:
            center[i,:] = compute_geometric_median(points[to_index]).median
            good_indices.append(i)
    if not good_indices: return None, float('inf')
    return k_medians_cost_label(points, center[good_indices,:])


def hard_noisy_oracle_median(data,label,k,alpha):
    new_labels = np.copy(label)
    for i in range(k):
        cluster_indices = np.where(label == i)[0]
        if len(cluster_indices) == 0:
            continue
        num_to_corrupt = int(alpha * len(cluster_indices))
        
        if num_to_corrupt == 0:
            continue

        indices_to_corrupt = np.random.choice(
            cluster_indices, 
            size=num_to_corrupt, 
            replace=False  
        )
        possible_new_labels = list(range(k))
        new_random_labels = np.random.choice(
            possible_new_labels, 
            size=num_to_corrupt
        )
        new_labels[indices_to_corrupt] = new_random_labels
        
    return new_labels
    
    




        