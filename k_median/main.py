import numpy as np
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.datasets import load_digits
from sklearn_extra.cluster import KMedoids
from huang import huang
from ours import Ours
from ngu import ngu, det
from utils import unpickle, hard_noisy_oracle_median, label_cost

def run_single_test_iteration(data, k, alpha, ground_truth_labels, n_trials):
    """
    Runs a single iteration of the hyperparameter search for all algorithms.

    Args:
        data (np.array): The input data.
        k (int): The number of clusters.
        alpha (float): The noise level.
        ground_truth_labels (np.array): The true cluster labels for comparison.
        n_trials (int): The number of hyperparameter values to test.

    Returns:
        dict: A dictionary containing the best results (cost, time, labels) for each method.
    """
    # Generate noisy labels for this iteration
    noisy_labels = hard_noisy_oracle_median(data, ground_truth_labels, k, alpha)

    # Define hyperparameter search spaces for the algorithms
    pvals_ngu = np.linspace(0.01, 0.5, n_trials)
    pvals_det = np.linspace(0.01, 0.5, n_trials)
    pvals_huang = np.linspace(0.01, 0.5, n_trials)
    pvals_ours = np.linspace(0.01, 0.5, n_trials)

    best_results = {
        'det': {'cost': float('inf'), 'time': 0, 'labels': None},
        'huang': {'cost': float('inf'), 'time': 0, 'labels': None},
        'ngu': {'cost': float('inf'), 'time': 0, 'labels': None},
        'ours': {'cost': float('inf'), 'time': 0, 'labels': None}
    }

    # Iterate through hyperparameter options to find the best one for each algorithm
    for p_ngu, p_det, p_huang, p_ours in zip(pvals_ngu, pvals_det, pvals_huang, pvals_ours):
        # Run DET algorithm
        labels_d, cost_d, time_d = det(X=data, y_noisy=noisy_labels, k=k, eps=0.1)
        if cost_d < best_results['det']['cost']:
            best_results['det'] = {'cost': cost_d, 'time': time_d, 'labels': labels_d}

        # Run Huang's algorithm
        labels_h, cost_h, time_h = huang(X=data, y_noisy=noisy_labels, k=k, alpha=p_huang, epsilon=0.1, sample_size=5, num_sampled_levels=1, max_step=1)
        if cost_h < best_results['huang']['cost']:
            best_results['huang'] = {'cost': cost_h, 'time': time_h, 'labels': labels_h}

        # Run Ngu's algorithm
        labels_n, cost_n, time_n = ngu(X=data, y_noisy=noisy_labels, k=k, alpha=p_ngu, sample_size=20)
        if cost_n < best_results['ngu']['cost']:
            best_results['ngu'] = {'cost': cost_n, 'time': time_n, 'labels': labels_n}

        # Run our proposed algorithm
        labels_o, cost_o, time_o = Ours(data, noisy_labels, k, p_ours, epsilon=0.1, max_step=2, repeat=1, sample_size_p=100, sample_size_r=10, sample_size_subsets=1000, step_rate=5)
        if cost_o < best_results['ours']['cost']:
            best_results['ours'] = {'cost': cost_o, 'time': time_o, 'labels': labels_o}

    return best_results, noisy_labels


def run_experiment():
    """
    Main function to run the clustering algorithm comparison experiments.
    This script runs two main experiments:
    1. Fixed Alpha, Varying K: Compares algorithms with a fixed noise level across different numbers of clusters.
    2. Fixed K, Varying Alpha: Compares algorithms with a fixed number of clusters across different noise levels.
    """
    # --- Experiment Configuration ---
    np.random.seed(42)  # for reproducibility

    # Parameters for the experiments
    k_values = [5, 10, 15, 20]            # List of k values to test
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5] # List of alpha values to test
    fixed_alpha_for_k_test = 0.3         # The fixed alpha for the first experiment
    fixed_k_for_alpha_test = 10          # The fixed k for the second experiment

    # General settings
    dataset_name = 'mnist'             # Options: 'cifar10', 'mnist', 'phy', 'fashion_mnist'
    data_portion = 0.1                   # Use a fraction of the dataset for faster testing
    n_iterations = 2                   # Number of times to repeat each test for stable statistics
    n_trials = 2                     # Number of hyperparameter trials for each algorithm

    # --- File Naming and Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    detailed_csv_filename = f"detailed_runs_{dataset_name}_{timestamp}.csv"
    summary_output_filename = f"summary_report_{timestamp}.txt"
    all_results = [] # List to hold all DataFrames before concatenating

    # --- Data Loading ---
    print(f"Loading dataset: {dataset_name}...")
    if dataset_name == 'cifar10':
        full_data = unpickle('dataset/cifar-10-batches-py/test_batch')[b'data']
    elif dataset_name == 'mnist':
        full_data = load_digits().data
    elif dataset_name == 'phy':
        full_data = np.loadtxt("dataset/phy.dat")
    elif dataset_name == 'fashion_mnist':
        full_data = pd.read_csv('dataset/fashion-mnist.csv').drop('label', axis=1).values
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    np.random.shuffle(full_data)
    num_points = int(len(full_data) * data_portion)
    data = full_data[:num_points, :]
    print(f"Dataset loaded. Shape: {data.shape}\n")

    # === Experiment 1: Fixed Alpha, Varying K ===
    print("\n" + "="*80)
    print(f"=== Starting Experiment 1: Fixed Alpha = {fixed_alpha_for_k_test}, Varying K ===")
    print("="*80)
    for k in k_values:
        print(f"\n--- Testing k = {k} ---")
        # Generate ground truth labels for this k
        kmedoids_model = KMedoids(n_clusters=k, method='pam', init='k-medoids++', random_state=42).fit(data)
        ground_truth_labels = kmedoids_model.labels_
        _, optimal_cost = label_cost(data, ground_truth_labels, k)

        current_run_results = []
        for i in range(n_iterations):
            print(f"  - Iteration {i+1}/{n_iterations}")
            best_results, noisy_labels = run_single_test_iteration(data, k, fixed_alpha_for_k_test, ground_truth_labels, n_trials)
            _, baseline_cost = label_cost(data, noisy_labels, k)

            # Record optimal and baseline
            current_run_results.append({'experiment': 'fixed_alpha', 'k': k, 'alpha': fixed_alpha_for_k_test, 'iteration': i+1, 'method': 'optimal', 'cost': optimal_cost, 'time': 0, 'nmi': 1.0, 'ari': 1.0})
            current_run_results.append({'experiment': 'fixed_alpha', 'k': k, 'alpha': fixed_alpha_for_k_test, 'iteration': i+1, 'method': 'baseline', 'cost': baseline_cost, 'time': 0, 'nmi': normalized_mutual_info_score(ground_truth_labels, noisy_labels), 'ari': adjusted_rand_score(ground_truth_labels, noisy_labels)})

            # Record results for each algorithm
            for method, res in best_results.items():
                current_run_results.append({
                    'experiment': 'fixed_alpha', 'k': k, 'alpha': fixed_alpha_for_k_test, 'iteration': i + 1, 'method': method,
                    'cost': res['cost'], 'time': res['time'],
                    'nmi': normalized_mutual_info_score(ground_truth_labels, res['labels']),
                    'ari': adjusted_rand_score(ground_truth_labels, res['labels'])
                })
        all_results.append(pd.DataFrame(current_run_results))

    # === Experiment 2: Fixed K, Varying Alpha ===
    print("\n" + "="*80)
    print(f"=== Starting Experiment 2: Fixed K = {fixed_k_for_alpha_test}, Varying Alpha ===")
    print("="*80)
    # Generate ground truth labels once for the fixed k
    kmedoids_model = KMedoids(n_clusters=fixed_k_for_alpha_test, method='pam', init='k-medoids++', random_state=42).fit(data)
    ground_truth_labels = kmedoids_model.labels_
    _, optimal_cost = label_cost(data, ground_truth_labels, fixed_k_for_alpha_test)

    for alpha in alpha_values:
        print(f"\n--- Testing alpha = {alpha} ---")
        current_run_results = []
        for i in range(n_iterations):
            print(f"  - Iteration {i+1}/{n_iterations}")
            best_results, noisy_labels = run_single_test_iteration(data, fixed_k_for_alpha_test, alpha, ground_truth_labels, n_trials)
            _, baseline_cost = label_cost(data, noisy_labels, fixed_k_for_alpha_test)

            # Record optimal and baseline
            current_run_results.append({'experiment': 'fixed_k', 'k': fixed_k_for_alpha_test, 'alpha': alpha, 'iteration': i+1, 'method': 'optimal', 'cost': optimal_cost, 'time': 0, 'nmi': 1.0, 'ari': 1.0})
            current_run_results.append({'experiment': 'fixed_k', 'k': fixed_k_for_alpha_test, 'alpha': alpha, 'iteration': i+1, 'method': 'baseline', 'cost': baseline_cost, 'time': 0, 'nmi': normalized_mutual_info_score(ground_truth_labels, noisy_labels), 'ari': adjusted_rand_score(ground_truth_labels, noisy_labels)})

            # Record results for each algorithm
            for method, res in best_results.items():
                current_run_results.append({
                    'experiment': 'fixed_k', 'k': fixed_k_for_alpha_test, 'alpha': alpha, 'iteration': i + 1, 'method': method,
                    'cost': res['cost'], 'time': res['time'],
                    'nmi': normalized_mutual_info_score(ground_truth_labels, res['labels']),
                    'ari': adjusted_rand_score(ground_truth_labels, res['labels'])
                })
        all_results.append(pd.DataFrame(current_run_results))

    # --- Final Processing and Saving ---
    print("\n\n--- All experiments are complete! Processing and saving results... ---")
    final_df = pd.concat(all_results, ignore_index=True)

    # Save detailed CSV
    final_df.to_csv(detailed_csv_filename, index=False, float_format='%.4f')
    print(f"Detailed logs for all experiments saved in: {detailed_csv_filename}")

    # Calculate and save summary statistics
    summary_stats = final_df.groupby(['experiment', 'k', 'alpha', 'method']).agg(
        avg_cost=('cost', 'mean'), std_cost=('cost', 'std'),
        avg_time=('time', 'mean'), std_time=('time', 'std'),
        avg_nmi=('nmi', 'mean'), std_nmi=('nmi', 'std'),
        avg_ari=('ari', 'mean'), std_ari=('ari', 'std')
    ).reset_index()

    with open(summary_output_filename, 'w') as f:
        f.write(f"Experiment Summary Report ({timestamp})\n")
        f.write(f"Dataset: {dataset_name}, Data Portion: {data_portion}, Iterations: {n_iterations}\n\n")

        # Summary for Experiment 1
        f.write("="*80 + "\n")
        f.write(f"EXPERIMENT 1: FIXED ALPHA = {fixed_alpha_for_k_test}, VARYING K\n")
        f.write("="*80 + "\n\n")
        exp1_summary = summary_stats[summary_stats['experiment'] == 'fixed_alpha']
        f.write(exp1_summary.to_string(index=False))
        f.write("\n\n")

        # Summary for Experiment 2
        f.write("="*80 + "\n")
        f.write(f"EXPERIMENT 2: FIXED K = {fixed_k_for_alpha_test}, VARYING ALPHA\n")
        f.write("="*80 + "\n\n")
        exp2_summary = summary_stats[summary_stats['experiment'] == 'fixed_k']
        f.write(exp2_summary.to_string(index=False))

    print(f"Full summary report saved in: {summary_output_filename}")


if __name__ == '__main__':
    run_experiment()
