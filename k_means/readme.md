# README

## Algorithm Hyperparameters

### Oracle Algorithm
- **Hyperparameters**:
  - No hyperparameters (baseline algorithm)
- **Function**: `hard_noisy_oracle(data, label, k, alpha)`

### Algo1 Algorithm
- **Hyperparameters**:
  - `eps`: 0.01 to 0.5 (epsilon parameter)
- **Function**: `algo1(points, oracle_labels, k, eps)`
- **Optimization**: Grid search over eps values

### DET Algorithm
- **Hyperparameters**:
  - `eps`: 0.01 to 0.5 (epsilon parameter)
- **Function**: `detAlg(points, oracle_labels, k, eps)`
- **Optimization**: Grid search over eps values

### Huang Algorithm
- **Hyperparameters**:
  - `p_huang`: 0.01 to 0.5
- **Function**: `huang(points, oracle_labels, k, p_huang)`
- **Optimization**: Grid search over p_huang values

### Huang1 Algorithm
- **Hyperparameters**:
  - `p_huang`: 0.01 to 0.5
- **Function**: `huang1(points, oracle_labels, k, p_huang)`
- **Optimization**: Grid search over p_huang values

### Huang2 Algorithm
- **Hyperparameters**:
  - `p_huang`: 0.01 to 0.5
- **Function**: `huang2(points, oracle_labels, k, p_huang)`
- **Optimization**: Grid search over p_huang values

### Ours Algorithm
- **Hyperparameters**:
  - `repeat`: [1, 2, 3] (Number of repetitions)
  - `size_subsets`: [1, 5, 10, 20, 50, 100, 200, 500, 1000] (Size of subsets)
  - `sample_size_subsets`: [1, 5, 10, 20, 50, 100, 200, 500, 1000] (Sample size for subsets)
- **Function**: `Ours(data, oracle_labels, k, alpha, epsilon, repeat, size_subsets, sample_size_subsets)`
- **Default Values**:
  - `repeat = 0`: Auto-calculated as `ceil(log2(k))`
  - `size_subsets = 0`: Auto-calculated as `ceil(1/epsilon)`
  - `sample_size_subsets = 0`: Auto-calculated as `ceil(1/((1-alpha)*epsilon))`

## Dataset-Specific Best Parameters

### CIFAR-10 Dataset
- **Best Parameters for Ours Algorithm (k=10, α=0.2)**:
  - `repeat`: 1
  - `size_subsets`: 5
  - `sample_size_subsets`: 5
  - 

### Fashion-MNIST Dataset
- **Best Parameters for Ours Algorithm (k=10, α=0.2)**:
  - `repeat`: 5
  - `size_subsets`: 10
  - `sample_size_subsets`: 20


### Physics Dataset (PHY)
- **Best Parameters for Ours Algorithm (k=10, α=0.2)**:
  - `repeat`: 5
  - `size_subsets`: 10
  - `sample_size_subsets`: 20


### MNIST Digits Dataset
- **Best Parameters for Ours Algorithm (k=10, α=0.2)**:
  - `repeat`: 1
  - `size_subsets`: 5
  - `sample_size_subsets`: 10

## Global Experiment Parameters

### Common Parameters Across All Datasets
- **Noise Levels (α)**: [0.1, 0.2, 0.3, 0.4, 0.5]
- **Number of Clusters (k)**: [10, 20, 30, 40, 50]
- **Number of Iterations (nIters)**: 10
- **Number of Trials (nTrials)**: 10
- **Random Seed**: 42 

### Parameter Optimization Settings
- **Parameter Range**: [0.01, 0.5] (linear distribution)
- **Best Value Selection**: Minimum cost criterion

## Requirements

- Python 3.8+
- Required Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `scipy`

## Notes

1. **Memory Requirements**: Full dataset experiments require significant memory
2. **Computation Time**: Grid search can be time-consuming, especially for large parameter spaces
3. **Result Stability**: Multiple iterations ensure result reliability
4. **Parameter Sensitivity**: Different datasets may require different optimal parameters
5. **Scalability**: Algorithm performance varies with dataset size and dimensionality


