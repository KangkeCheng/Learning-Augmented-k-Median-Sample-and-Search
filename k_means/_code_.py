import sklearn

import sys 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import random
import time


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# return k means cost given centers
def k_means_cost(points, centers):
    distance = euclidean_distances(points, centers)
    distance = distance**2
    labels = np.argmin(distance, axis=1)
    return labels, np.min(distance, axis = 1).sum()

def k_means_labels(points, centers):
    distance = euclidean_distances(points, centers)
    distance = distance**2
    labels = np.argmin(distance, axis=1)
    return labels

def kmeans_cost_label(points, labels, num_labels):
    _ , d = np.shape(points)
    centers = np.zeros((num_labels, d))
    good_indices = []
    for i in range(num_labels):
        to_index = np.where(labels == i)[0]
        if len(to_index) > 0:
            curr_points = points[to_index]
            centers[i,:] = np.average(curr_points, axis = 0)
            good_indices.append(i)
        else:
            pass
    centers = centers[good_indices,:]
        
    return k_means_cost(points, centers)


    

def hard_noisy_oracle(data,label,k,alpha):
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


def algo2(points, eps):
    n = len(points)
    
    to_return = 0.0
    if n < 10:
        return sum(points)/n

    for i in range(25):
        # randomly partition points into two groups of equal size
        points = np.random.permutation(points)
        X1 = points[:n//2]
        X2 = points[n//2:]
        X1 = np.sort(X1)

        # find interval of X1 with (1-eps) fraction of points
        # call this interval [a,b]
        counter = int((1-5*eps)*(n//2))
        curr_len = float('inf')
        a = 0
        b = 0
        for i in range(n//2-counter+1):
            curr_int_left = X1[i]
            curr_int_right = X1[i+counter-1]
            if curr_int_right -  curr_int_left < curr_len:
                a = curr_int_left
                b = curr_int_right
                curr_len = b - a
        X2_filtered = [x for x in X2 if a <= x <= b]


        # return average of points in X2 that are in [a,b]
        if len(X2_filtered) == 0:
            to_return += 0.0
        else:
            to_return += sum(X2_filtered)/len(X2_filtered)
    return to_return/25.0


def algo1(points, oracle_labels, k, eps):
    n,d = points.shape
    centers = np.zeros((k, d))
    labels_so_far = []

    # loop over each label
    for i in range(k):

        # get labels that haven't been processed so far
        good_indices = np.where(~np.isin(oracle_labels, labels_so_far))[0]
        curr_labels = oracle_labels[good_indices]
        
        if len(curr_labels) > 0:

            # get most common label
            label_counts = np.bincount(curr_labels)
            most_common_label = np.argmax(label_counts)
            points_with_labels = points[np.where(oracle_labels == most_common_label)[0]]


            # for most common label, loop over each dimension and run alg 2
            for j in range(d):
                curr_dim_points = points_with_labels[:,j]
                curr_dim_center = algo2new(curr_dim_points, eps)
                centers[most_common_label, j] = curr_dim_center
            
            labels_so_far.append(most_common_label)
        else:
            pass
    return centers






def detAlg(points, oracle_labels, k, eps):
    n,d = points.shape
    centers = np.zeros((k, d))
    for i in range(k):
        points_with_labels = points[np.where(oracle_labels == i)[0]]
        for j in range(d):
            curr_dim_points = points_with_labels[:,j]
            curr_dim_center = smallCluster(curr_dim_points, eps, j)
            centers[i, j] = curr_dim_center
    return centers


def smallCluster(L, eps, j):
    K = int(np.floor(len(L)*(1-eps)))
    K = max(K, 1)
    L = np.sort(L)
    S = np.sum(L[:K])
    S_square = np.sum((L**2)[:K])    
    best_mean = S / K
    best_cost = S_square - S**2 / K
    costList = []
    costList.append(best_cost)
    for i in range(K, len(L)):
        S_square += L[i]**2 - L[i-K]**2 
        S += L[i] - L[i-K] 
        cost_all = S_square - S**2 / (K)
        costList.append(cost_all)
        if cost_all < best_cost:
            best_mean = S/K
            best_cost = cost_all
    return best_mean
def algo2new(points, eps):
    
    n = len(points)
    
    if n <= 10:
        return points.mean()
    
    to_return = 0.0
    for i in range(1):
        points = np.random.permutation(points)
        X1 = points[:n//2]
        X2 = points[n//2:]
        X1 = np.sort(X1)

        counter = int((1-5*eps)*(n//2))

        
        if counter == 1:
            to_return += X2.mean()
        else:
            X1_left = X1[:-counter+1]
            X1_right = X1[counter-1:]

            good_indx = np.argmin(X1_right-X1_left)
            a = X1_left[good_indx]
            b = X1_right[good_indx]
            to_index = np.where((a <= X2) & (X2 <= b))[0]
            if len(to_index) == 0:
                to_return += 0.0
            else:
                to_return += X2[to_index].mean()

    return to_return/1
    