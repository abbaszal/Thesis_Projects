import pandas as pd 
import numpy as np       
import math            
import ast                
import matplotlib.pyplot as plt  
import seaborn as sns     
from collections import defaultdict  
from itertools import combinations  



def compute_and_visualize_shapley_values(df_results, client_local_accuracies, n_clients=10, plot=True, model_name="Model", print_df=True):


    subset_value = defaultdict(float)
    for index, row in df_results.iterrows():
        clients_included = row['Clients']

        if isinstance(clients_included, str):
            try:
                clients_included = ast.literal_eval(clients_included)
            except Exception as e:
                print(f"Warning: Unable to parse Clients column at index {index}: {clients_included}. Error: {e}")
                clients_included = []
        global_accuracy = row['Global Accuracy']

        subset = frozenset([client - 1 for client in clients_included])
        subset_value[subset] = global_accuracy


    all_subsets = []
    for k in range(n_clients + 1):
        for subset in combinations(range(n_clients), k):
            all_subsets.append(frozenset(subset))


    factorials = [math.factorial(k) for k in range(n_clients + 1)]
    n_factorial = factorials[n_clients]

    shapley_values = np.zeros(n_clients)


    for i in range(n_clients):
        shapley_value_i = 0.0
        for S in all_subsets:
            if i not in S:
                S_union_i = S.union({i})
                v_S = subset_value.get(S, 0)
                v_S_union_i = subset_value.get(S_union_i, 0)
                weight = (factorials[len(S)] * factorials[n_clients - len(S) - 1]) / n_factorial
                marginal_contribution = v_S_union_i - v_S
                shapley_value_i += weight * marginal_contribution
        shapley_values[i] = shapley_value_i

    grand_coalition = frozenset(range(n_clients))
    total_value = subset_value.get(grand_coalition, 0)
    if total_value == 0 or np.isnan(total_value):
        print("Warning: Grand coalition global accuracy is zero or NaN. Normalized Shapley values will not be computed normally.")
        normalized_shapley_values = shapley_values  
    else:
        normalized_shapley_values = shapley_values / total_value


    clients = [f'Client {i + 1}' for i in range(n_clients)]
    local_vals = [client_local_accuracies.get(i, np.nan) for i in range(n_clients)]
    df_output = pd.DataFrame({
        'Client': clients,
        'Local Accuracy': local_vals,
        'Normalized Shapley Value': normalized_shapley_values
    })

    sorted_df = df_output.sort_values(by='Normalized Shapley Value', ascending=False)
    
    if print_df:
        print(sorted_df)

    if plot:
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("tab10", n_colors=len(local_vals))
        for i, (local_val, shapley_val) in enumerate(zip(local_vals, normalized_shapley_values)):
            plt.scatter(local_val, shapley_val, color=colors[i], label=f"Client {i + 1}", s=100)
        plt.xlabel("Local Accuracy")
        plt.ylabel("Normalized Shapley Value")
        plt.title(f"{model_name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_output



