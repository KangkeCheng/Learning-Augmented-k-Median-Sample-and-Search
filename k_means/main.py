import os
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Import scikit-learn and other ML libraries
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

# --- Module Imports & Dummy Functions ---
# This block attempts to import your custom modules.
# If they are not found, it creates placeholder (dummy) functions
# so the script can still run for demonstration purposes.

from _code_ import kmeans_cost_label, algo1, k_means_cost, detAlg, hard_noisy_oracle, unpickle
from ours import Ours
from huang_code_ import huang, huang1, huang2

   




def run_single_experiment(k, alpha, args, test_data, true_labels):
    """
    Runs all algorithms and their trials sequentially for a given k and alpha.

    Args:
        k (int): The number of clusters.
        alpha (float): The noise level to test.
        args (Namespace): Command-line arguments.
        test_data (np.array): The dataset.
        true_labels (np.array): The ground truth labels for the data.

    Returns:
        list: A list of dictionaries, where each dictionary is a raw result from one run.
    """
    # Generate noisy labels based on the current alpha
    noisy_labels = hard_noisy_oracle(test_data, true_labels, k, alpha)
    if noisy_labels is None: return []

    methods_to_run = {
        'algo1': algo1, 'det': detAlg, 'huang': huang,
        'huang1': huang1, 'huang2': huang2, 'ours': Ours
    }
    
    raw_results = []
    pvals = np.linspace(0.01, 0.5, args.nTrials)

    # Loop over the number of iterations to get averaged results
    for iteration in range(args.nIters):
        # For each iteration, run every method across its hyperparameter trials
        for p_val in pvals:
            for name, func in methods_to_run.items():
                try:
                    start_time = time.time()
                    # Special handling for 'ours' if it has a different signature
                    if name == 'ours':
                        centers = func(test_data, noisy_labels.copy(), k, p_val, epsilon=0.1, repeat=5, size_subsets=10, sample_size_subsets=20)
                    else:
                        centers = func(test_data, noisy_labels.copy(), k, p_val)
                    exec_time = time.time() - start_time
                    
                    if centers is None or len(centers) == 0:
                        print(f"Warning: Method {name} returned no centers for k={k}, alpha={alpha}, p_val={p_val}.")
                        continue

                    # Evaluate the performance of the found centers
                    labels, cost = k_means_cost(test_data, centers)
                    ami = adjusted_mutual_info_score(true_labels, labels)
                    ari = adjusted_rand_score(true_labels, labels)
                    
                    raw_results.append({
                        'iteration': iteration, 'method': name, 'alpha': alpha, 'k': k,
                        'cost': cost, 'time': exec_time, 'ami': ami, 'ari': ari, 'param': p_val
                    })
                except Exception as e:
                    print(f"ERROR during k={k}, alpha={alpha}, method={name}, p_val={p_val}. Error: {e}")
    
    # Calculate and add the 'oracle' baseline metrics for comparison
    oracle_cost = kmeans_cost_label(test_data, noisy_labels, k)[1]
    oracle_ami = adjusted_mutual_info_score(true_labels, noisy_labels)
    oracle_ari = adjusted_rand_score(true_labels, noisy_labels)
    for iteration in range(args.nIters):
        raw_results.append({
            'iteration': iteration, 'method': 'oracle', 'alpha': alpha, 'k': k,
            'cost': oracle_cost, 'time': 0, 'ami': oracle_ami, 'ari': oracle_ari, 'param': 0
        })
        
    return raw_results

def find_best_results(raw_results, nIters):
    """
    Processes raw results to find the best parameter for each method in each iteration,
    and then averages the metrics of these best runs.

    Args:
        raw_results (list): A list of raw result dictionaries.
        nIters (int): The number of iterations performed.

    Returns:
        dict: A dictionary containing the final aggregated results for each method.
    """
    if not raw_results: return {}
    df = pd.DataFrame(raw_results)
    final_results = {}
    methods = df['method'].unique()

    for method in methods:
        method_df = df[df['method'] == method]
        if method == 'oracle':
            # Oracle is a baseline, so just take the mean (they should all be the same)
            final_results[method] = method_df.mean(numeric_only=True).to_dict()
            for key in ['cost_std', 'time_std', 'ami_std', 'ari_std']: final_results[method][key] = 0
        else:
            # For each iteration, find the parameter 'p_val' that resulted in the minimum cost
            best_run_indices = [method_df[method_df['iteration'] == i]['cost'].idxmin() 
                                for i in range(nIters) if not method_df[method_df['iteration'] == i].empty]
            
            if best_run_indices:
                best_runs_df = method_df.loc[best_run_indices]
                # Calculate the mean and standard deviation over these best runs
                final_results[method] = {
                    'cost': best_runs_df['cost'].mean(), 'cost_std': best_runs_df['cost'].std(),
                    'time': best_runs_df['time'].mean(), 'time_std': best_runs_df['time'].std(),
                    'ami': best_runs_df['ami'].mean(), 'ami_std': best_runs_df['ami'].std(),
                    'ari': best_runs_df['ari'].mean(), 'ari_std': best_runs_df['ari'].std(),
                }
    return final_results


def load_dataset(name, portion):
    """Loads and preprocesses the specified dataset."""
    print(f"Loading '{name}' dataset...")
    try:
        if name == 'cifar10':
            if not os.path.exists('dataset/cifar-10-batches-py'):
                raise FileNotFoundError("CIFAR-10 data directory not found.")
            data_dir = 'dataset/cifar-10-batches-py'
            data_ = [unpickle(os.path.join(data_dir, f'data_batch_{i}'))[b'data'] for i in range(1, 6)]
            full_data = np.concatenate(data_)
        elif name == 'phy':
            full_data = np.loadtxt("dataset/phy.dat")
        elif name == 'mnist':
            full_data = load_digits().data
        elif name == 'fashion_mnist':
            df = pd.read_csv('dataset/fashion-mnist.csv')
            full_data = df.drop('label', axis=1).values
        else:
            raise ValueError(f"Unknown dataset: {name}")

        np.random.seed(42) # for reproducibility
        np.random.shuffle(full_data)
        num_points = int(len(full_data) * portion)
        print(f"Successfully loaded '{name}'. Using {num_points} data points.")
        return full_data[:num_points, :]
    except FileNotFoundError:
        print(f"ERROR: Data file not found for dataset '{name}'. Please check the path.")
        print("Creating dummy data for demonstration.")
        return np.random.rand(1000, 64)
    except Exception as e:
        print(f"An error occurred while loading dataset '{name}': {e}")
        return None
            
def main(args):
    """
    Main function to orchestrate the sequential experiments.
    """
    test_data = load_dataset(args.dataset, args.portion)
    if test_data is None: return

    all_final_results = []
    methods_order = ['OPT', 'oracle', 'algo1', 'det', 'huang', 'huang1', 'huang2', 'ours']

    # --- Experiment 1: Fixed Alpha, Varying K ---
    print(f"\n{'='*20} Starting Experiment 1: Fixed Alpha = {args.fixed_alpha}, Varying K {'='*20}")
    for k in args.k_values:
        print(f"\n{'─'*15} Running for k = {k} {'─'*15}")
        # Generate "true" labels for the current k
        kmeans_scikit = KMeans(n_clusters=k, random_state=42, n_init=10).fit(test_data)
        true_labels = kmeans_scikit.labels_
        cost_opt = kmeans_cost_label(test_data, true_labels, k)[1]

        raw_results = run_single_experiment(k, args.fixed_alpha, args, test_data, true_labels)
        final_results = find_best_results(raw_results, args.nIters)
        final_results['OPT'] = {'cost': cost_opt, 'cost_std': 0, 'time': 0, 'time_std': 0, 'ami': 1.0, 'ami_std': 0, 'ari': 1.0, 'ari_std': 0}

        for method in methods_order:
            if method in final_results and final_results[method]:
                res = final_results[method]
                new_row = {'experiment': 'fixed_alpha', 'dataset': args.dataset, 'k': k, 'alpha': args.fixed_alpha, 'method': method, **res}
                all_final_results.append(new_row)

    # --- Experiment 2: Fixed K, Varying Alpha ---
    print(f"\n{'='*20} Starting Experiment 2: Fixed K = {args.fixed_k}, Varying Alpha {'='*20}")
    # Generate "true" labels once for the fixed k
    kmeans_scikit = KMeans(n_clusters=args.fixed_k, random_state=42, n_init=10).fit(test_data)
    true_labels = kmeans_scikit.labels_
    cost_opt = kmeans_cost_label(test_data, true_labels, args.fixed_k)[1]

    for alpha in args.alpha_values:
        print(f"\n{'─'*15} Running for alpha = {alpha} {'─'*15}")
        raw_results = run_single_experiment(args.fixed_k, alpha, args, test_data, true_labels)
        final_results = find_best_results(raw_results, args.nIters)
        final_results['OPT'] = {'cost': cost_opt, 'cost_std': 0, 'time': 0, 'time_std': 0, 'ami': 1.0, 'ami_std': 0, 'ari': 1.0, 'ari_std': 0}

        for method in methods_order:
            if method in final_results and final_results[method]:
                res = final_results[method]
                new_row = {'experiment': 'fixed_k', 'dataset': args.dataset, 'k': args.fixed_k, 'alpha': alpha, 'method': method, **res}
                all_final_results.append(new_row)

    # --- Save Final Results ---
    print(f"\n{'='*20} All experiments processed. Saving final results. {'='*20}")
    if not all_final_results:
        print("Warning: No results were generated to save.")
        return

    # Create and format the final DataFrame
    results_df = pd.DataFrame(all_final_results)
    results_df.rename(columns={'cost_std': 'cost_dev', 'time_std': 'time_dev', 'ami_std': 'ami_dev', 'ari_std': 'ari_dev'}, inplace=True)
    
    # Sort the DataFrame for clean presentation
    results_df['method'] = pd.Categorical(results_df['method'], categories=methods_order, ordered=True)
    results_df.sort_values(by=['experiment', 'k', 'alpha', 'method'], inplace=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = args.output_file or f"experiment_results_{args.dataset}_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False, float_format='%.4f')
    
    print(f"\n{'='*20} All experiments complete {'='*20}")
    print(f"Final results have been saved to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run clustering experiments with noisy labels sequentially.")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['phy', 'cifar10', 'mnist', 'fashion_mnist'], help='Dataset to use.')
    
    # Arguments for Experiment 1
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 20, 30], help='List of k values to test in Experiment 1.')
    parser.add_argument('--fixed_alpha', type=float, default=0.2, help='The fixed alpha (noise level) for Experiment 1.')

    # Arguments for Experiment 2
    parser.add_argument('--alpha_values', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4], help='List of alpha values to test in Experiment 2.')
    parser.add_argument('--fixed_k', type=int, default=10, help='The fixed number of clusters (k) for Experiment 2.')

    # General arguments
    parser.add_argument('--nIters', type=int, default=5, help='Number of iterations to average results over.')
    parser.add_argument('--nTrials', type=int, default=5, help='Number of hyperparameter values to try for each method.')
    parser.add_argument('--portion', type=float, default=1.0, help='Fraction of the dataset to use.')
    parser.add_argument('--output_file', type=str, default=None, help='Name for the output CSV file.')
    
    args, unknown = parser.parse_known_args()
    main(args)
