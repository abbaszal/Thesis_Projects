import numpy as np


def split_data_equal(X: np.ndarray, y: np.ndarray, n_clients: int, shuffle: bool=False, random_seed: int = None):
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]

    n_entries_per_client = X.shape[0] // n_clients
    partitions = []
    for i in range(n_clients):
        start = i * n_entries_per_client
        end = (i + 1) * n_entries_per_client
        X_i = X[start:end]
        y_i = y[start:end]
        partitions.append((X_i, y_i))
    # Ensure the last partition gets all remaining data.
    partitions[-1] = (X[start:], y[start:])

    return partitions