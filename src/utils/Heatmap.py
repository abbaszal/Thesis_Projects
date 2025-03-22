import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_client_occurrences(file_path, best_clients_count, worst_clients_count, title_example="Example", plot_worst_only=False):

    df = pd.read_csv(file_path)
    

    filtered_df = df[df["Combination"] == 1111111111]
    

    client_columns = [col for col in df.columns if "Accuracy" in col and "Client" in col]
    client_avg_accuracies = filtered_df[client_columns].mean()
    

    sorted_clients = client_avg_accuracies.sort_values(ascending=False)
    best_clients = sorted_clients.head(best_clients_count)
    worst_clients = sorted_clients.tail(worst_clients_count)
    
    best_clients_numbers = [int(col.split()[1]) for col in best_clients.index]
    worst_clients_numbers = [int(col.split()[1]) for col in worst_clients.index]
    

    client_occurrences = {col: df[col].count() for col in client_columns}
    client_occurrences_series = pd.Series(client_occurrences).sort_values(ascending=False)
    all_clients = client_avg_accuracies.index.tolist()  
    all_clients_numbers = [int(col.split()[1]) for col in all_clients]
    

    selected_clients = worst_clients.index if plot_worst_only else all_clients
    selected_clients_numbers = worst_clients_numbers if plot_worst_only else all_clients_numbers
    

    unique_combinations = sorted(df["Combination"].unique(), reverse=True)  
    heatmap_data = np.zeros((len(unique_combinations), len(selected_clients)))
    
    for i, combination in enumerate(unique_combinations):
        subset = df[df["Combination"] == combination]
        for j, client in enumerate(selected_clients):
            heatmap_data[i, j] = subset[client].count()
    

    formatted_x_labels = [f"{num}\n{client_avg_accuracies.get(col, 0) * 100:.3f}%" for col, num in zip(selected_clients, selected_clients_numbers)]
    

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, annot=True, cmap="Greens", vmin=0, vmax=heatmap_data.max(),
                      center=heatmap_data.mean(), xticklabels=formatted_x_labels, yticklabels=unique_combinations)
    plt.xlabel("Client Number (with Avg Accuracy)")
    plt.ylabel("Combination")
    plt.title(f"Heatmap of {'Worst' if plot_worst_only else 'All'} Clients in Nash Combinations for {title_example}")
    

    xticks = ax.get_xticklabels()
    for tick in xticks:
        client_info = tick.get_text().split("\n")[0] 
        if client_info.isdigit():
            client_number = int(client_info)
            if plot_worst_only:
                if client_number in worst_clients_numbers:
                    tick.set_color("red")
            else:
                if client_number in best_clients_numbers:
                    tick.set_color("green")
                elif client_number in worst_clients_numbers:
                    tick.set_color("red")
    
    plt.show()