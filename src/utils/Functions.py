import numpy as np
import pandas as pd
import copy
import ast
import math
from itertools import chain, combinations
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import math
import tqdm
import copy
import ast
from itertools import chain, combinations
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Function to find best and worst clients based on local accuracies list
def find_best_worst_clients(client_local_accuracies):
    if not client_local_accuracies:
        print("The client local accuracies list is empty.")
        return None, None

    # Find the index of the best and worst clients
    best_client = max(range(len(client_local_accuracies)), key=lambda i: client_local_accuracies[i]) + 1
    worst_client = min(range(len(client_local_accuracies)), key=lambda i: client_local_accuracies[i]) + 1

    print(f"Best Client: {best_client}, Worst Client: {worst_client}")
    return best_client, worst_client






# Function to calculate mean global accuracy with/without specific client
def calculate_global_accuracy(df_results, client_id):

 
    with_client = df_results[df_results['Clients'].apply(lambda x: client_id in x)]
    without_client = df_results[df_results['Clients'].apply(lambda x: client_id not in x)]

 
    if not with_client.empty:
        mean_with_client = with_client['Global Accuracy'].mean()
        std_with_client = with_client['Global Accuracy'].std()
        min_with_client = with_client['Global Accuracy'].min()
        max_with_client = with_client['Global Accuracy'].max()
    else:
        mean_with_client = std_with_client = min_with_client = max_with_client = None


    if not without_client.empty:
        mean_without_client = without_client['Global Accuracy'].mean()
        std_without_client = without_client['Global Accuracy'].std()
        min_without_client = without_client['Global Accuracy'].min()
        max_without_client = without_client['Global Accuracy'].max()
    else:
        mean_without_client = std_without_client = min_without_client = max_without_client = None

    print(f"Statistics of Global Accuracy with Client {client_id}:")
    print(f"  Mean: {mean_with_client}")
    print(f"  Std Dev: {std_with_client}")
    print(f"  Min: {min_with_client}")
    print(f"  Max: {max_with_client}")

    print(f"\nStatistics of Global Accuracy without Client {client_id}:")
    print(f"  Mean: {mean_without_client}")
    print(f"  Std Dev: {std_without_client}")
    print(f"  Min: {min_without_client}")
    print(f"  Max: {max_without_client}")

  
    return {
        'with_client': {
            'mean': mean_with_client,
            'std': std_with_client,
            'min': min_with_client,
            'max': max_with_client,
            'data': with_client['Global Accuracy']
        },
        'without_client': {
            'mean': mean_without_client,
            'std': std_without_client,
            'min': min_without_client,
            'max': max_without_client,
            'data': without_client['Global Accuracy']
        }
    }






# Function to find the best global accuracy per client
def find_best_global_accuracy_per_client(df_results):

    best_global_accuracy_per_client = {}
    
    for client_idx in range(1, len(df_results.columns) - 2):
        with_client_rows = df_results[df_results['Clients'].apply(lambda x: client_idx in x)]
        if not with_client_rows.empty:
            best_row = with_client_rows.loc[with_client_rows['Global Accuracy'].idxmax()]
            best_global_accuracy_per_client[client_idx] = {
                "Best Global Accuracy": best_row['Global Accuracy'],
                "Best Combination": best_row['Clients']
            }
        else:
            print(f"No rows found for client {client_idx}.")

    if best_global_accuracy_per_client:
        df_best_global_per_client = pd.DataFrame.from_dict(best_global_accuracy_per_client, orient='index')
        df_best_global_per_client.index.name = "Client"
        df_best_global_per_client.reset_index(inplace=True)
        print("Best Global Accuracy per Client:\n", df_best_global_per_client)
        return df_best_global_per_client
    else:
        print("No best global accuracy data found.")
        return pd.DataFrame()
    
def find_worst_global_accuracy_per_client(df_results):

    worst_global_accuracy_per_client = {}
    
    for client_idx in range(1, len(df_results.columns) - 2):
        with_client_rows = df_results[df_results['Clients'].apply(lambda x: client_idx in x)]
        if not with_client_rows.empty:
            worst_row = with_client_rows.loc[with_client_rows['Global Accuracy'].idxmin()]
            worst_global_accuracy_per_client[client_idx] = {
                "Worst Global Accuracy": worst_row['Global Accuracy'],
                "Worst Combination": worst_row['Clients']
            }
        else:
            print(f"No rows found for client {client_idx}.")

    if worst_global_accuracy_per_client:
        df_worst_global_per_client = pd.DataFrame.from_dict(worst_global_accuracy_per_client, orient='index')
        df_worst_global_per_client.index.name = "Client"
        df_worst_global_per_client.reset_index(inplace=True)
        print("Worst Global Accuracy per Client:\n", df_worst_global_per_client)
        return df_worst_global_per_client
    else:
        print("No worst global accuracy data found.")
        return pd.DataFrame()




# Function to calculate client contributions to best/worst combinations
def calculate_client_contributions(df_results, best=True):

    combinations = []
    for client_idx in range(1, len(df_results.columns) - 2):
        with_client_rows = df_results[df_results['Clients'].apply(lambda x: client_idx in x)]
        if not with_client_rows.empty:
            row = with_client_rows.loc[with_client_rows['Global Accuracy'].idxmax() if best else with_client_rows['Global Accuracy'].idxmin()]
            combinations.append((row['Clients']))
        else:
            print(f"No rows found for client {client_idx}.")

    if combinations:
        flattened_combinations = sum(combinations, [])
        client_counts = Counter(flattened_combinations)
        client_contribution_df = pd.DataFrame(client_counts.items(), columns=['Client', 'Frequency'])
        client_contribution_df.sort_values(by='Frequency', ascending=False, inplace=True)
        print("Client Contributions:\n", client_contribution_df)
        return client_contribution_df
    else:
        print("No valid combinations found.")
        return pd.DataFrame()






# Function to visualize client contributions
def visualize_client_contributions(client_contribution_df, title):
    """
    Visualizes client contributions using a bar chart.
    """
    plt.bar(client_contribution_df['Client'], client_contribution_df['Frequency'])
    plt.xlabel('Client')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()



# Function to analyze worst combinations
def analyze_worst_combinations(df_results):
    """
    Analyzes the worst combinations for each client based on global accuracy.
    """
    worst_combinations = []
    for client_idx in range(1, len(df_results.columns) - 2):
        with_client_rows = df_results[df_results['Clients'].apply(lambda x: client_idx in x)]
        if not with_client_rows.empty:
            worst_row = with_client_rows.loc[with_client_rows['Global Accuracy'].idxmin()]
            worst_combinations.append((worst_row['Clients']))
        else:
            print(f"No rows found for client {client_idx}.")

    if worst_combinations:
        flattened_worst_combinations = sum(worst_combinations, [])
        client_worst_counts = Counter(flattened_worst_combinations)
        client_worst_contribution_df = pd.DataFrame(client_worst_counts.items(), columns=['Client', 'Frequency'])
        client_worst_contribution_df.sort_values(by='Frequency', ascending=False, inplace=True)

        print("\nClient Contribution to Worst Combinations:\n", client_worst_contribution_df)
        plt.bar(client_worst_contribution_df['Client'], client_worst_contribution_df['Frequency'])
        plt.xlabel('Client')
        plt.ylabel('Frequency in Worst Combinations')
        plt.title('Client Contribution to Worst Combinations')
        plt.show()

        return client_worst_contribution_df
    else:
        print("No worst combinations found.")
        return pd.DataFrame()




def compute_and_visualize_shapley_values(df_results, client_local_accuracies, n_clients=10, plot=True, model_name="Model", print_df=True):

    # Create a mapping from each coalition (as a frozenset of 0-indexed clients) to its global accuracy.
    subset_value = defaultdict(float)
    for index, row in df_results.iterrows():
        clients_included = row['Clients']
        # If the Clients value is stored as a string (e.g. "[1, 2, 3]"), convert it to a list.
        if isinstance(clients_included, str):
            try:
                clients_included = ast.literal_eval(clients_included)
            except Exception as e:
                print(f"Warning: Unable to parse Clients column at index {index}: {clients_included}. Error: {e}")
                clients_included = []
        global_accuracy = row['Global Accuracy']
        # Convert 1-indexed clients to 0-indexed and create a frozenset
        subset = frozenset([client - 1 for client in clients_included])
        subset_value[subset] = global_accuracy

    # Generate all possible subsets (coalitions) of clients (0-indexed)
    all_subsets = []
    for k in range(n_clients + 1):
        for subset in combinations(range(n_clients), k):
            all_subsets.append(frozenset(subset))

    # Precompute factorials for efficiency in weight computation
    factorials = [math.factorial(k) for k in range(n_clients + 1)]
    n_factorial = factorials[n_clients]

    # Initialize an array for Shapley values
    shapley_values = np.zeros(n_clients)

    # Compute the Shapley value for each client
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

    # Get the global accuracy of the grand coalition (all clients)
    grand_coalition = frozenset(range(n_clients))
    total_value = subset_value.get(grand_coalition, 0)
    if total_value == 0 or np.isnan(total_value):
        print("Warning: Grand coalition global accuracy is zero or NaN. Normalized Shapley values will not be computed normally.")
        normalized_shapley_values = shapley_values  # or, for instance, you might set them to zeros
    else:
        normalized_shapley_values = shapley_values / total_value

    # Prepare the output DataFrame
    clients = [f'Client {i + 1}' for i in range(n_clients)]
    # Ensure that the keys used in client_local_accuracies are integers 0..n_clients-1
    local_vals = [client_local_accuracies.get(i, np.nan) for i in range(n_clients)]
    df_output = pd.DataFrame({
        'Client': clients,
        'Local Accuracy': local_vals,
        'Normalized Shapley Value': normalized_shapley_values
    })

    # Sort and print the DataFrame for convenience
    sorted_df = df_output.sort_values(by='Normalized Shapley Value', ascending=False)
    
    # Print DataFrame only if print_df is True
    if print_df:
        print(sorted_df)

    # Plot the values only if requested
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









def calculate_and_plot_global_accuracy_with_differences(df_results, n_clients):

    client_ids = range(1, n_clients + 1)
    means_with = []
    means_without = []
    differences = []


    for client_id in client_ids:
        with_client = df_results[df_results['Clients'].apply(lambda x: client_id in x)]
        without_client = df_results[df_results['Clients'].apply(lambda x: client_id not in x)]

        mean_with_client = with_client['Global Accuracy'].mean() if not with_client.empty else 0
        mean_without_client = without_client['Global Accuracy'].mean() if not without_client.empty else 0

        # Print the means for debugging or insight
        print(f"Client {client_id}: Mean With Client = {mean_with_client:.2f}, Mean Without Client = {mean_without_client:.2f}")

        means_with.append(mean_with_client)
        means_without.append(mean_without_client)
        differences.append(mean_with_client - mean_without_client)


    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    indices = np.arange(len(client_ids))


    plt.bar(indices, means_with, bar_width, label='With Client', alpha=0.7, color='skyblue')
    plt.bar(indices + bar_width, means_without, bar_width, label='Without Client', alpha=0.7, color='orange')


    for i, diff in enumerate(differences):
        plt.plot(
            [indices[i] + bar_width / 2, indices[i] + bar_width / 2],
            [means_with[i], means_without[i]],
            color='black',
            linestyle='--'
        )

        plt.text(
            indices[i] + bar_width / 2,
            (means_with[i] + means_without[i]) / 2,
            f"{diff:.2f}",
            ha='center',
            va='center',
            fontsize=10,
            color='black'
        )


    plt.xlabel('Client ID')
    plt.ylabel('Mean Global Accuracy')
    plt.title('Mean Global Accuracy With and Without Each Client (Differences Highlighted)')
    plt.xticks(indices + bar_width / 2, [str(client) for client in client_ids])
    plt.legend()
    plt.tight_layout()


    plt.show()

    
    return {
        'mean_with': means_with,
        'mean_without': means_without,
        'differences': differences
    }


