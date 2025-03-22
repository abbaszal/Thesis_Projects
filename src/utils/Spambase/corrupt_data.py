import numpy as np
from typing import Optional


def corrupt_data(
    X: np.ndarray, 
    y: np.ndarray, 
    corruption_prob: float = 0.8, 
    nan_prob: float = 0.5, 
    noise_std: Optional[float] = None,
    label_corruption_prob: float = 0.2,
    random_seed: Optional[int] = None
):
    if noise_std is None:
        noise_std = 0.1
        
    X = X.copy()
    y = y.copy()
    
    if random_seed is not None:
        np.random.seed(random_seed)  
    
    n_samples, n_features = X.shape
    mask = np.random.rand(n_samples) < corruption_prob
    mask_nan = mask & (np.random.rand(n_samples) < nan_prob)
    mask_noise = mask & (~mask_nan)
    
    # Corrupt feature values
    X += mask_noise[:, None] * np.random.randn(n_samples, n_features) * noise_std
    X[np.argwhere(mask_nan)] = np.nan
    

    corruption_mask = np.random.rand(n_samples) < label_corruption_prob  # Completely random selection
    
    unique_labels = np.unique(y)
    corrupted_labels = []
    for original_label in y[corruption_mask]:
        possible_labels = unique_labels[unique_labels != original_label]
        corrupted_label = np.random.choice(possible_labels)
        corrupted_labels.append(corrupted_label)
    
    y[corruption_mask] = corrupted_labels
    
    return X, y


def corrupt_clients(f, partitions, corrupt_client_indices, 
                    corruption_prob=0.6, nan_prob=0.5, noise_std=None, 
                    label_corruption_prob=0.1, base_seed=42):
    for idx in corrupt_client_indices:

        if noise_std is None:
            noise_std = 0.1  

        client_seed = int(base_seed + idx + noise_std * 100)
        np.random.seed(client_seed)  

        X_i, y_i = partitions[idx]

        X_i_corrupted, y_i_corrupted = f(
            X_i.copy(), y_i.copy(), 
            corruption_prob=corruption_prob, 
            nan_prob=nan_prob, 
            noise_std=noise_std, 
            label_corruption_prob=label_corruption_prob, 
            random_seed=client_seed
        )

        partitions[idx] = (X_i_corrupted, y_i_corrupted)
    

    return partitions, corrupt_client_indices