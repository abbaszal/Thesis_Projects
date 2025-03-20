import numpy as np
from typing import Optional


def corrupt_data(X: np.ndarray, corruption_prob: float = 0.8, nan_prob: float = 0.5,
                 noise_std: Optional[float] = None):
    if noise_std is None:
        noise_std = 0.1
    
    X = X.copy()
    n_samples, n_features = X.shape
    mask = np.random.rand(n_samples) < corruption_prob
    mask_nan = mask & (np.random.rand(n_samples) < nan_prob)
    mask_noise = mask & (~mask_nan)
    
    corrupt_features = range(n_features)

    # Apply corruption
    for feature in corrupt_features:
        X[mask_noise, feature] += np.random.randn(mask_noise.sum()) * noise_std

    X[np.argwhere(mask_nan)] = np.nan
    
    return X 




def corrupt_labels(y: np.ndarray, corruption_prob: float = 0.2, random_seed: Optional[int] = None):
    if random_seed is not None:
        np.random.seed(random_seed) 

    y = y.copy()  # Avoid modifying the original array


    corruption_mask = np.random.rand(len(y)) < corruption_prob  

    unique_labels = np.unique(y)


    corrupted_labels = []
    for original_label in y[corruption_mask]:

        possible_labels = unique_labels[unique_labels != original_label]
        corrupted_label = np.random.choice(possible_labels)
        corrupted_labels.append(corrupted_label)
    
    # Apply corrupted labels
    y[corruption_mask] = corrupted_labels

    return y

