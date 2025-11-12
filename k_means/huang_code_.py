import numpy as np
from _code_ import kmeans_cost_label, algo1, k_means_cost, detAlg, hard_noisy_oracle, unpickle
from sklearn.cluster import KMeans
import random
# import csv
import math
from sklearn.datasets import load_digits
from sklearn.cluster import kmeans_plusplus as kpp
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import BallTree

import warnings
from ours import *
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import pandas as pd
import time

import os







def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.array(linear_assignment(np.max(w) - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def find_minimum(dim_j_points, sample_id, n_neibor):
    minimum = 1E20
    best_point = 0.0 # 提供一个默认返回值
    dim_j_points = np.sort(dim_j_points)
    sum_l1 = dim_j_points ** 2
    for j1 in range(0, len(sample_id)):
        l = sample_id[j1]
        points_l = dim_j_points[l:l + n_neibor]
        
        # --- 核心修复：添加安全检查 ---
        if points_l.size > 0:
            sum_l = np.sum(points_l)
            cost = np.sum(sum_l1[l:l + n_neibor]) - sum_l ** 2 / points_l.shape[0]
            if (cost < minimum):
                minimum = cost
                best_point = sum_l / n_neibor
        # 如果 points_l 为空，则不进行任何操作，使用上一次的 best_point 或默认值
        
    return best_point


def find_center(dim_j_points, sample_id, n_neibor):
    minimum = 1E20
    best_point = 0.0 # 提供一个默认返回值
    sum_l1 = dim_j_points ** 2
    for j1 in range(0, len(sample_id)):
        dis = np.abs(dim_j_points - dim_j_points[sample_id[j1]])
        
        # 确保 n_neibor 不大于数组长度
        k_neighbors = min(n_neibor, len(dis) -1)
        if k_neighbors <= 0: continue

        nearest_id = np.argpartition(dis, k_neighbors)[0:k_neighbors]
        nearest_points = dim_j_points[nearest_id]
        
        # --- 核心修复：添加安全检查 ---
        if nearest_points.size > 0:
            sum_l = np.sum(nearest_points)
            cost = np.sum(sum_l1[nearest_id]) - sum_l ** 2 / n_neibor
            if (cost < minimum):
                minimum = cost
                best_point = sum_l / n_neibor

    return best_point


def find_minimum1(dim_j_points, omega_j, sample_id, outliers, n_neibor_1):
    minimum = 1E20
    best_point = None
    
    for j1 in range(0, len(sample_id)):
        dis = np.sum((omega_j - dim_j_points[sample_id[j1]]) ** 2, axis=1)
        
        safe_outliers = min(outliers, omega_j.shape[0] - 1)
        num_inliers = omega_j.shape[0] - safe_outliers
        if num_inliers <= 0: continue
            
        nearest = np.argpartition(dis, num_inliers - 1)[:num_inliers]
        cost_j1 = (dis[nearest]).sum()
        
        if (cost_j1 < minimum):
            minimum = cost_j1
            best_point = dim_j_points[sample_id[j1]]

    if best_point is None:
        best_point = dim_j_points[0] # 如果没有找到，提供一个默认值
        
    best_point = best_point.reshape(1, -1)
    
    dis_j1 = np.sum((dim_j_points - best_point) ** 2, axis=1)
    
    k_neighbors = min(n_neibor_1, len(dis_j1) - 1)
    if k_neighbors <= 0: return np.mean(dim_j_points, axis=0) # 返回簇的整体均值作为备用
        
    n_id = np.argpartition(dis_j1, k_neighbors)[0:k_neighbors]
    nearest_points = dim_j_points[n_id]
    
    # --- 核心修复：添加安全检查 ---
    if nearest_points.size > 0:
        center = np.mean(nearest_points, axis=0)
    else:
        # 如果还是空，返回最佳候选点本身或整体均值
        center = best_point.flatten()
        
    return center


def find_minimum2(dim_j_points, omega_j, center_candidates, outliers, n_neibor_1):
    minimum = 1E20
    best_point = center_candidates[0] # 提供一个默认值
    
    for j1 in range(0, len(center_candidates)):
        dis = np.abs(omega_j - center_candidates[j1])
        
        safe_outliers = min(outliers, omega_j.shape[0] - 1)
        num_inliers = omega_j.shape[0] - safe_outliers
        if num_inliers <= 0: continue
            
        nearest = np.argpartition(dis, num_inliers - 1)[:num_inliers]
        cost_j1 = (dis[nearest]).sum()
        if (cost_j1 < minimum):
            minimum = cost_j1
            best_point = center_candidates[j1]
            
    dis_j1 = np.abs(dim_j_points - best_point)
    
    k_neighbors = min(n_neibor_1, len(dis_j1) - 1)
    if k_neighbors <= 0: return np.mean(dim_j_points) # 返回整体均值
        
    n_id = np.argpartition(dis_j1, k_neighbors)[0:k_neighbors]
    nearest_points = dim_j_points[n_id]

    # --- 核心修复：添加安全检查 ---
    if nearest_points.size > 0:
        center = np.mean(nearest_points)
    else:
        # 如果还是空，返回最佳候选点
        center = best_point
        
    return center



def huang(points, oracle_labels, k, p_huang):
    n, d = points.shape

    # print("Method", k*d, n/100)

    centers = np.zeros((k, d))
    # sample_range = [i for i in range(0, n)]
    for i in range(0, k):
        R = 2
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor = math.floor((1 - p_huang) * points_i.shape[0])
        if(points_i.shape[1] * k > points_i.shape[0]/25):
            sample_id1 = random.sample(range(0, points_i.shape[0] - n_neibor), min(R, points_i.shape[0] - n_neibor))
        else:
            sample_id1 = random.sample(range(0, points_i.shape[0]), min(R, points_i.shape[0]))
        sample_id1 = np.array(sample_id1)
        for j in range(0, d):
            dim_j_points = points_i[:, j]
            if (points_i.shape[1] * k > dim_j_points.shape[0]/25):
                best_point = find_minimum(dim_j_points, sample_id1, n_neibor)
            else:
                best_point = find_center(dim_j_points, sample_id1, n_neibor)

            centers[i][j] = best_point

    return centers


    


def huang1(points, oracle_labels, k, p_huang):
    # print("Check", p_huang)
    n, d = points.shape
    centers = np.zeros((k, d))
    epsilon = 0.2
    for i in range(0, k):
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor_1 = math.floor((1 - p_huang) * points_i.shape[0])
        R = 10
        epsilon = 1
        sample_size = math.log10(
            (points_i.shape[0] ** 3) * d * (math.log10(n * 1E4 / (epsilon ** 2))) ** 3) * math.log10(
            points_i.shape[0] * 1E4) / (epsilon ** 4)
        sample_size = min(int(sample_size), int(points_i.shape[0] / 20))
        sample_size = max(sample_size, 2)

        outliers = math.floor((p_huang * 1.3 * math.ceil(sample_size)))
        outliers = max(outliers, 1)

        dim_j_points = points_i
        omega_j = random.sample(range(0, points_i.shape[0]), sample_size)
        omega_j = dim_j_points[omega_j]
        sample_id = random.sample(range(0, dim_j_points.shape[0]), min(dim_j_points.shape[0], R))
        sample_id = np.array(sample_id)

        best_center = find_minimum1(dim_j_points.copy(), omega_j, sample_id, outliers, n_neibor_1)
        centers[i] = best_center

    return centers


def generate_center_candidates(dim_j_points, sample_id, p_huang):
    lower = 1e-2
    upper = dim_j_points.shape[0] ** 2
    q = lower

    # Estimate a maximum size for the candidate array
    max_candidates = len(sample_id) * 10 * int(math.log2(upper / lower))
    center_candidates = np.zeros(max_candidates)
    count = 0  # Tracks the number of populated rows in center_candidates

    # Initial population of center_candidate with original sample points
    for j1 in range(len(sample_id)):
        if count < max_candidates:
            center_candidates[count] = dim_j_points[sample_id[j1]]
            count += 1

    # Iterative loop to populate center_candidate with shifted points
    while q < upper:
        lij = math.sqrt(q / ((1 - p_huang) * dim_j_points.shape[0]))
        shifts = np.array([-2 * lij, -lij, lij, 2 * lij])
        for j1 in range(len(sample_id)):
            base_point = dim_j_points[sample_id[j1]]
            for shift in shifts:
                if count < max_candidates:
                    center_candidates[count] = base_point + shift
                    count += 1

        q *= 10  # Double q in each iteration
    center_candidates = center_candidates[:count]
    center_candidates = np.sort(center_candidates)
    id_new = np.zeros(len(center_candidates), dtype=np.int64)
    now = center_candidates[0]
    id_now = 1
    for i in range(1, len(center_candidates)):
        if (center_candidates[i] - now < 1E-2):
            continue
        else:
            id_new[id_now] = i
            id_now += 1
            now = center_candidates[i]
    id_new = id_new[:id_now]
    center_candidates = center_candidates[id_new]
    return center_candidates[:count]





def huang2(points, oracle_labels, k, p_huang):
    n, d = points.shape
    centers = np.zeros((k, d))
    epsilon = 0.2
    for i in range(0, k):
        points_i = points[np.where(oracle_labels == i)[0]]
        n_neibor_1 = math.floor((1 - 2 * p_huang) * points_i.shape[0])
        R = 5
        epsilon = 1
        sample_size = math.log10(
            (points_i.shape[0] ** 3) * d * (math.log10(n * 1E4 / (epsilon ** 2))) ** 3) * math.log10(
            points_i.shape[0] * 1E4) / (epsilon ** 4)
        sample_size = min(int(sample_size), int(points_i.shape[0] / 20))
        sample_size = max(sample_size, 2)
        weights = np.ones(sample_size) * points_i.shape[0] / sample_size
        outliers = math.floor((p_huang * 1.3 * math.ceil(sample_size)))
        outliers = max(outliers, 1)

        for j in range(0, d):
            dim_j_points = points_i[:, j]
            omega_j = random.sample(range(0, points_i.shape[0]), sample_size)
            omega_j = dim_j_points[omega_j]
            sample_id = random.sample(range(0, dim_j_points.shape[0]), min(dim_j_points.shape[0], R))

            center_candidates = generate_center_candidates(dim_j_points, sample_id, p_huang)
            centers[i][j] = find_minimum2(dim_j_points.copy(), omega_j, center_candidates, outliers, n_neibor_1)
            stop = 1

    return centers
