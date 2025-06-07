import pandas as pd
import matplotlib.pyplot as plt
import os

def encode_client(i: int, num_clients: int):
    encoded = [0] * num_clients
    encoded[i] = 1
    encoded = "".join(map(str, encoded))
    return encoded


def compute_shapley_values(coalition_values: dict):
    
    num_players = len(next(iter(coalition_values)))
    shapley = [0.0] * num_players

    # Precompute factorials
    factorial = [1] * (num_players + 1)
    for i in range(1, num_players + 1):
        factorial[i] = factorial[i - 1] * i

    def weight(s):
        return factorial[s] * factorial[num_players - s - 1] / factorial[num_players]

    for i in range(num_players):
        for coalition_bin, value in coalition_values.items():
            if coalition_bin[i] == '0':
                # Create new coalition with player i added
                new_coalition = list(coalition_bin)
                new_coalition[i] = '1'
                new_coalition_str = ''.join(new_coalition)

                if new_coalition_str in coalition_values:
                    marginal_contribution = coalition_values[new_coalition_str] - value
                    coalition_size = coalition_bin.count('1')
                    shapley[i] += weight(coalition_size) * marginal_contribution

    return shapley


def plot_local_acc_vs_shapley(local_accuracies: list, shapley_values: list, savepath: str|None=None, title: str|None=None):

    plt.figure(figsize=(5, 3))
    plt.scatter(local_accuracies, shapley_values)
    plt.xlabel('Local Accuracy')
    plt.ylabel('Shapley Value')
    if title:
        plt.title(title)
    plt.grid(linestyle=':')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.draw()



if __name__ == "__main__":

    csv_files = [
        "results/FedLR_HuGaDB_without_LQC/HuGaDB_results_with_LR.csv",
        "results/FedFor_HuGaDB_without_LQC/HuGaDB_results_with_FedFor.csv",
        "results/FedLR_Spambase_without_LQC/Spambase_results_with_LR.csv",
        "results/FedFor_Spambase_without_LQC/Spambase_results_with_FedFor.csv",
    ]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype={"Combination": str})

        coalition_values = {row["Combination"]: row["Global Accuracy"] for _, row in df.iterrows()}
        shapley_values = compute_shapley_values(coalition_values)

        num_players = len(next(iter(coalition_values)))
        local_accuracies = [df[df["Combination"] == encode_client(i, num_players)]["Global Accuracy"].values[0] for i in range(num_players)]

        title = csv_file.split("/")[1].split("_without")[0].replace("_", " ")

        os.makedirs("fig", exist_ok=True)

        # print(shapley_values)
        outfile = "fig/shapley_" + csv_file.split("/")[-1].replace(".csv", ".pdf")
        plot_local_acc_vs_shapley(local_accuracies, shapley_values, savepath=outfile, title=title)

    plt.show()
