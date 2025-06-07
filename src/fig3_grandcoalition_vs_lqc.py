import pandas as pd 
import matplotlib.pyplot as plt
import os

noise_std_values = [0.1, 0.5, 2.0, 5.0]
markers = ['x','o','s','>', '*','v', '^', '<', 'd','h', 'D']
linestyles = ['-', '-.' , ':', '--']
colors = ['black', 'blue', 'fuchsia', 'red']

def plot_grandcoalition_vs_lqc(df: pd.DataFrame, title: str|None=None, savepath: str|None=None):
    
    plt.figure(figsize=(5, 3))
    for i, noise_std in enumerate(noise_std_values):
        plt.plot(
            df['num_bad_clients'], df[f"{noise_std:.1f}"], 
            marker=markers[i % len(markers)], 
            linestyle=linestyles[i % len(linestyles)], 
            label=r'$\sigma$=' + f'{noise_std}',
            color=colors[i % len(colors)],
            fillstyle='none'
        )

    plt.xlabel('Number of LQC')
    plt.ylabel('Grand coalition NE occurrences')
    plt.grid(linestyle=':')
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.draw()



if __name__ == "__main__":

    csv_files = [
        "results/FedLR_HuGaDB_LQC_0_to_10/nash_occurrence_results.csv",
        "results/FedFor_HuGaDB_LQC_0_to_10/nash_occurrence_results.csv",
        "results/FedLR_Spambase_LQC_0_to_10/nash_occurrence_results.csv",
        "results/FedFor_Spambase_LQC_0_to_10/nash_occurrence_results.csv",
    ]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=",")
        df = df.rename(columns={'Number of Bad Clients': 'num_bad_clients'})
        
        os.makedirs("fig", exist_ok=True)

        title = csv_file.split("/")[1].split("_LQC")[0].replace("_", " ")
        savepath = "fig/grand_coalition_vs_lqc_" + title.replace(" ", "_") + ".pdf"

        plot_grandcoalition_vs_lqc(df, title=title, savepath=savepath)
        
    plt.show()
    

        
