import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.HuGaDB.corrupt_data_hugadb import corrupt_data, corrupt_labels  


def prepare_partitions(client_idx, trial_seed, sample_size, num_corrupted_clients,
                       train_files_pattern, corruption_params, label_corruption_prob , label_encoder):
    """
    Prepares the training partition for a given client.
    Loads data, samples a fixed number of instances, and (if applicable)
    corrupts the data. Returns scaled features and encoded labels.
    """
    # Load training data for this client.
    train_file = train_files_pattern.format(client_idx)
    df_train_local = pd.read_csv(train_file).dropna(subset=['act'])
    
    # Sample fixed number of instances.
    df_train_local, _ = train_test_split(
        df_train_local,
        train_size=sample_size,
        random_state=trial_seed,
        stratify=df_train_local['act']
    )
    df_train_local = df_train_local.reset_index(drop=True).dropna()
    
    # Apply corruption if the client is low-quality.
    if client_idx <= num_corrupted_clients:
        X_train_local = df_train_local.drop('act', axis=1).values
        y_train_local = df_train_local['act'].values
        X_train_local = corrupt_data(
            X_train_local,
            corruption_prob=corruption_params['corruption_prob'],
            nan_prob=corruption_params['nan_prob'],
            noise_std=corruption_params['noise_std']
        )
        y_train_local = corrupt_labels(
            y_train_local,
            corruption_prob=label_corruption_prob
        )
        # Rebuild DataFrame using original column names.
        df_train_local = pd.DataFrame(
            X_train_local,
            columns=pd.read_csv(train_file).drop('act', axis=1).columns
        )
        df_train_local['act'] = y_train_local
        df_train_local = df_train_local.dropna()
    
    # Prepare features and labels.
    X_train = df_train_local.drop('act', axis=1).values
    y_train = label_encoder.transform(df_train_local['act'])  # Assumes label_encoder is global.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled, y_train



