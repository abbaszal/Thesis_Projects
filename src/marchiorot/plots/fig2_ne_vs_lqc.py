import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


patterns = [ "x" , "o", "*", "/"]

def find_top_k_ne(ne_hist_dict: dict[int, pd.DataFrame], k: int):
    """
    Find top k Nash equilibria with overall most occurrences.
    """

    ne_hist_df = pd.concat(ne_hist_dict.values())
    ne_hist_df = ne_hist_df.groupby('Nash equilibrium').agg({'count': 'sum'}).reset_index()
    top_k_ne = ne_hist_df.sort_values(by='count', ascending=False).head(k)
    top_k_ne = top_k_ne['Nash equilibrium'].tolist()
    return top_k_ne


def plot_ne_vs_lqc(ne_hist_dict: dict[int, pd.DataFrame], top_k_ne: list[str], title: str|None=None, savepath: str|None=None):

    plt.figure(figsize=(5, 4))

    x_range = np.arange(len(top_k_ne))
    width = 0.25
    for i, (num_bad, ne_hist) in enumerate(ne_hist_dict.items()):


        ne_hist_k = [ne_hist.set_index('Nash equilibrium')['count'].get(ne, 0) for ne in top_k_ne]
        ne_acc_k = [ne_hist.set_index('Nash equilibrium')['global_accuracy_mean'].get(ne, 0) for ne in top_k_ne]
        ne_acc_k = [f"{acc:.2f}" for acc in ne_acc_k]

        plt.bar_label(plt.bar(
            x_range + i * width, 
            ne_hist_k, 
            label=f"{num_bad} LQC",
            width=width,
            color=[0.8, 0.8, 0.8],
            edgecolor='black',
            hatch=patterns[i % len(patterns)],
            alpha=0.9
        ),
        ne_acc_k,
        fontsize=8
        )

    plt.xticks(x_range + width, top_k_ne)
    plt.xlabel('Nash equilibrium')
    plt.ylabel('Occurrences')
    plt.xticks(rotation=90)
    plt.grid(linestyle=':')
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.draw()
    



if __name__ == "__main__":

    k = 5  # limit to top k nash equilibria

    for model in ["FedLR", "FedFor"]:
        for dataset in ["Spambase", "HuGaDB"]:

            model_dataset = f"{model}_{dataset}"

            basepath = f"results/{model_dataset}_LQC_0_to_10"
            
            ne_hist_dict = {}
            for num_bad in [3, 6, 9]:
                csv_file = f"{basepath}/Nash_Equilibrium_Details_{model.lower()}_noise_3_c{num_bad}.csv"

                df = pd.read_csv(csv_file, sep=",", dtype={"Combination": str})

                ne_hist = df.groupby("Combination").agg(
                    count=("Combination", "count"),
                    global_accuracy_mean=("Global Accuracy", "mean"),
                    global_accuracy_std=("Global Accuracy", "std"),
                ).reset_index()

                ne_hist = ne_hist.rename({'Combination': 'Nash equilibrium'}, axis=1)
                ne_hist = ne_hist.sort_values(by="count", ascending=False).reset_index(drop=True)
                ne_hist['global_accuracy_std'] = ne_hist['global_accuracy_std'].fillna(0)  # single occurrences cause NaNs

                # print(ne_hist)

                ne_hist_dict[num_bad] = ne_hist

            top_k_ne = find_top_k_ne(ne_hist_dict, k=k)

            title = f"{model_dataset.replace('_', ' ')} (top {k} NE)"
            savepath = f"fig/ne_hist_vs_lqc_{model_dataset}.pdf"
            plot_ne_vs_lqc(ne_hist_dict, top_k_ne=top_k_ne, savepath=savepath, title=title)

    plt.show()
    
    