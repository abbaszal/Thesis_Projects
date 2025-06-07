import pandas as pd

import matplotlib.pyplot as plt

def plot_nash_equilibrium_distribution(hist_combination: pd.Series, savepath: str):

    plt.figure(figsize=(8, 4))
    plt.bar(hist_combination.index, hist_combination.values)
    plt.xlabel("Nash Equilibrium")
    plt.ylabel("Count")
    plt.title("Distribution of Nash Equilibrium")
    plt.grid(linestyle=':')
    plt.xticks(rotation=90)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.draw()


def get_ne_details(df: pd.DataFrame, savepath: str):

        count_full_coalition = df[df["Combination"] == "1111111111"]["Combination"].count()
        count_other_coalitions = df[df["Combination"] != "1111111111"]["Combination"].count() 

        details_full_coalition = df[df["Combination"] == "1111111111"].agg(
            global_accuracy_mean=("Global Accuracy", "mean"),
            global_accuracy_std=("Global Accuracy", "std"),
        ).rename(columns={"Global Accuracy": "Grand coalition"})

        details_other_coalitions = df[df["Combination"] != "1111111111"].agg(
            global_accuracy_mean=("Global Accuracy", "mean"),
            global_accuracy_std=("Global Accuracy", "std"),
        ).rename(columns={"Global Accuracy": "Other coalitions"})

        details = pd.concat([details_full_coalition, details_other_coalitions], axis=1).T
        details["count"] = [count_full_coalition, count_other_coalitions]

        details["global accuracy"] = details.apply(lambda x: f"{round(x['global_accuracy_mean'], 4)} ({round(x['global_accuracy_std'], 4)})", axis=1)
        details = details[["count", "global accuracy"]]

        details.to_latex(savepath, index=False)

        return details

if __name__ == "__main__":

    csv_files = [
        "results/FedFor_HuGaDB_LQC_0_to_10/Nash_Equilibrium_Details_fedfor_noise_3_c0.csv",
        "results/FedLR_HuGaDB_LQC_0_to_10/Nash_Equilibrium_Details_fedlr_noise_3_c0.csv",
        "results/FedFor_Spambase_LQC_0_to_10/Nash_Equilibrium_Details_fedfor_noise_3_c0.csv",
        "results/FedLR_Spambase_LQC_0_to_10/Nash_Equilibrium_Details_fedlr_noise_3_c0.csv",
    ]

    table_dict = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype={"Combination": str})

        print(df["Combination"].value_counts())

        hist_combination = df["Combination"].value_counts()

        algorithm_dataset = csv_file.split("/")[1].split("_LQC_")[0].lower()

        outfile = "fig/ne_hist_" + algorithm_dataset + ".pdf"

        # plot_nash_equilibrium_distribution(hist_combination, savepath=outfile)

        # outtex = "tables/ne_details" + csv_file.split("/")[-1].split("_Details")[-1].replace(".csv", ".tex")

        details = get_ne_details(df, savepath=None)
        # print(details)

        details["algorithm"] = algorithm_dataset.replace("_", " ")
        table_dict[algorithm_dataset] = details

    table = pd.concat(table_dict.values())

    table = table.reset_index(drop=False).rename(columns={"index": "Nash equilibrium"})
    table = table.set_index(["algorithm", "Nash equilibrium"])

    table = table[["count", "global accuracy"]]

    print(table)

    # table.to_latex("tables/ne_details.tex", index=True)

    # plt.show()