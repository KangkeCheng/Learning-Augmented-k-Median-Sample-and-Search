import numpy as np
import math
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs

from utils import *
from geom_median.numpy import compute_geometric_median

def algo2Medians(points, oracle_labels, k, alpha, sample_size=20):

    n,d = points.shape
    centers = np.zeros((k, d))
    # loop over each label
    for i in range(k):
        points_with_labels = np.where(oracle_labels == i)[0]
        m_i = len(points_with_labels)
        best = float('inf')
        for j in range(sample_size):
            randPoint = np.random.choice(points_with_labels)
            randDist = euclidean_distances(points[[randPoint]], points[points_with_labels])
            
            index_to_keep = np.argsort(randDist)[0,: int((1-alpha) * m_i)]
            #print(np.argsort(randDist))
            #print(len(index_to_keep), len(points_with_labels))
            randSubset = points[points_with_labels[index_to_keep]]
            gm = compute_geometric_median(randSubset).median
            #print(np.average(randSubset, axis = 0))
            if sample_size > 1:
                
                cost = k_medians_cost_nonlabel(points[points_with_labels], gm)
                if cost < best:
                    best = cost
                    centers[i] = gm
            else:
                centers[i] = gm

    return centers
    
def algo1Medians(points, oracle_labels, k, eps, iterN = 1):
    n,d = points.shape
    centers = np.zeros((k, d))
    sampleN = int(1/eps**4 * (np.log(k/ eps))**2)

    # loop over each label
    for i in range(k):
        points_with_labels = np.where(oracle_labels == i)[0]
        best = float('inf')
        for j in range(iterN):
            randSubset = points[np.random.choice(points_with_labels, min(sampleN,len(points_with_labels)))]
            gm = compute_geometric_median(randSubset).median
            if iterN > 1:
                cost = k_medians_cost_nonlabel(points[points_with_labels], gm)
                if cost < best:
                    best = cost
                    centers[i] = gm
            else:
                centers[i] = gm
                    
    return centers




def ngu(X,y_noisy,k,alpha,sample_size):

    start_time = time.time()

    calculated_centers = algo2Medians(X, y_noisy, k, alpha,sample_size)
    end_time = time.time()
    labels,algorithm_cost = k_medians_cost(X, calculated_centers)
    return labels,algorithm_cost,end_time-start_time

def det(X,y_noisy,k,eps,iterN =20):

    start_time = time.time()

    calculated_centers = algo1Medians(X, y_noisy, k, eps, iterN)
    
    end_time = time.time()
    labels,algorithm_cost = k_medians_cost(X, calculated_centers)
    return labels,algorithm_cost,end_time-start_time
