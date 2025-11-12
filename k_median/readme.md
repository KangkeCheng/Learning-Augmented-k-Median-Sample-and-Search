# README

## Dataset Descriptions

### MNIST
- **Description**: A dataset of handwritten digits commonly used for image classification and clustering tasks.
- **Parameters**:
  - `max_step`: 3
  - `sample_size_p`: 1
  - `sample_size_r`: 5
  - `sample_size_subsets`: 10
  - `step_rate`: 20

### Fashion-MNIST
- **Description**: A dataset of fashion items (e.g., shirts, shoes) used for image classification and clustering.
- **Parameters**:
  - `max_step`: 5
  - `sample_size_p`: 100
  - `sample_size_r`: 10
  - `sample_size_subsets`: 1000
  - `step_rate`: 10

### CIFAR-10
- **Description**: A dataset of 60,000 32x32 color images in 10 classes, used for image classification and clustering.
- **Parameters**:
  - `max_step`: 2
  - `sample_size_p`: 100
  - `sample_size_r`: 10
  - `sample_size_subsets`: 1000
  - `step_rate`: 5

### PHY
- **Description**: A dataset containing physical measurements, used for clustering and statistical analysis.
- **Parameters**:
  - `max_step`: 2
  - `sample_size_p`: 100
  - `sample_size_r`: 10
  - `sample_size_subsets`: 1000
  - `step_rate`: 5

## Algorithm Descriptions

### Huang Algorithm
- **Description**: Implements a fast sampling k-median algorithm using literal grid search.
- **Key Features**:
  - Efficient sampling of candidate centers.
  - Supports multi-level sampling for better accuracy.

### Ours Algorithm
- **Description**: A novel k-median clustering algorithm designed for noisy datasets.
- **Key Features**:
  - Uses geometric median computation.
  - Supports adaptive step sizes and multi-step displacement.

### NGU Algorithm
- **Description**: Implements a k-median clustering algorithm with noise handling.
- **Key Features**:
  - Random sampling of points.
  - Geometric median computation for robust clustering.

### DET Algorithm
- **Description**: A deterministic k-median clustering algorithm.
- **Key Features**:
  - Iterative refinement of cluster centers.
  - Handles large datasets efficiently.

## Algorithm Hyperparameters

### Huang Algorithm
- **Hyperparameters**:
  - `sample_size`: 5
  - `num_sampled_levels`: 1
  - `alpha`: 0.1 to 0.5
  - `epsilon`: 0.1
  - `max_step`: 1

### Ours Algorithm
- **Hyperparameters**:
  - `sample_size_p`: 100
  - `sample_size_r`: 10
  - `sample_size_subsets`: 1000
  - `step_rate`: 5
  - `max_step`: 2

### NGU Algorithm
- **Hyperparameters**:
  - `sample_size`: 20


### DET Algorithm
- **Hyperparameters**:
  - `eps`: 0.1
  - `iterN`: 20
.
   - Use the CSV files and text reports for further analysis.

## Requirements

- Python 3.8+
- Required Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `geom-median`
