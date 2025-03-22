import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  


def evaluate_coalitions(client_models, client_global_accuracies, n_clients, aggregator_func, X_test, y_test, corrupt_client_indices, approach='fedlr'):
    results = []
    n_combinations = 2 ** n_clients
    for i in range(1, n_combinations):
        bin_str = format(i, f'0{n_clients}b')
        client_indices = [j for j in range(n_clients) if bin_str[n_clients - 1 - j] == '1']
        
        if approach == 'fedlr':
            included_models = [client_models[j] for j in client_indices if client_models[j] is not None]
            if not included_models:
                continue
            aggregated_model = aggregator_func(included_models)
            global_acc = aggregated_model.score(X_test, y_test)
        else:  # fedfor approach
            included_models = [client_models[j] for j in client_indices]
            if not included_models:
                continue
            forest = aggregator_func()  # e.g., FederatedForest()
            for model in included_models:
                forest.add_model(model)
            y_pred_global = forest.predict(X_test)
            global_acc = np.mean(y_pred_global == y_test)
        
        row = {'Combination': bin_str, 'Clients': [j+1 for j in client_indices], 'Global Accuracy': global_acc}
        if not isinstance(client_global_accuracies, dict):
            client_global_accuracies = {i: acc for i, acc in enumerate(client_global_accuracies)}
        for j in range(n_clients):
            col_name = f'Client {j+1} Accuracy'
            acc = np.nan
            if approach == 'fedlr':
                acc = client_global_accuracies[j] if client_global_accuracies[j] is not None else np.nan
            if j in corrupt_client_indices:
                col_name += " (corrupted client)"
            else:
                acc = client_global_accuracies.get(j, np.nan)
            row[col_name] = acc
        results.append(row)
    return pd.DataFrame(results)