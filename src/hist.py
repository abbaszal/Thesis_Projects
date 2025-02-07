import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

def plot_nash_equilibrium_histogram(directory, file_pattern, title_suffix):
    total_occurrences = {}
    for filename in os.listdir(directory):
        if re.match(file_pattern, filename) and filename.endswith(".csv"):  
            file_path = os.path.join(directory, filename)

            df = pd.read_csv(file_path, dtype=str)

            expected_columns = ["Nash Equilibrium", "Occurrences"]
            available_columns = [col for col in expected_columns if col in df.columns]
            df = df[available_columns]

            df.rename(columns={df.columns[0]: "Nash Equilibrium", df.columns[1]: "Occurrences"}, inplace=True)

            df["Occurrences"] = pd.to_numeric(df["Occurrences"], errors="coerce").fillna(0)

            for index, row in df.iterrows():
                equilibrium = str(row["Nash Equilibrium"]).strip()
                occurrences = row["Occurrences"]

                if equilibrium in total_occurrences:
                    total_occurrences[equilibrium] += occurrences
                else:
                    total_occurrences[equilibrium] = occurrences

    total_df = pd.DataFrame(list(total_occurrences.items()), columns=["Nash Equilibrium", "Total Occurrences"])

    total_df = total_df.sort_values(by="Total Occurrences", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(total_df["Nash Equilibrium"], total_df["Total Occurrences"], color="skyblue", edgecolor="blue")
    plt.xlabel("Nash Equilibrium")
    plt.ylabel("Total Occurrences")
    plt.title(f"Histogram of Nash Equilibrium Occurrences For {title_suffix}")
    plt.xticks(rotation=90, fontsize=8)
    max_occurrences = max(total_df["Total Occurrences"])
    step_size = 5
    plt.yticks(np.arange(0, max_occurrences + step_size, step=step_size))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
