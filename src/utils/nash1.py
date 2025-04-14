import pandas as pd
import re

def remove_redundant_information(df_results: pd.DataFrame) -> pd.DataFrame:

    redundant_cols = [col for col in df_results.columns if re.match(r"Client \d+ Accuracy", col) is not None]
    df_results = df_results.drop(columns=redundant_cols)
    return df_results

def one_hot_coalition(player: int, n_players: int) -> str:

    one_hot = ["0"] * n_players
    one_hot[player - 1] = "1"
    return "".join(one_hot)

def evaluate_deviation(df_results: pd.DataFrame, coalition: str, player: int) -> bool:

    current_move = coalition[player - 1]
    one_hot_player = one_hot_coalition(player, len(coalition))
    standalone_payoff = df_results.loc[one_hot_player, "Global Accuracy"]
    if current_move == "0":
        new_coalition = list(coalition)
        new_coalition[player - 1] = "1"
        new_coalition = "".join(new_coalition)
        coalition_payoff = df_results.loc[new_coalition, "Global Accuracy"]
        return coalition_payoff + 0.000001 > standalone_payoff

    elif current_move == "1":

        coalition_payoff = df_results.loc[coalition, "Global Accuracy"]
        return standalone_payoff + 0.000001 > coalition_payoff

    else:
        raise ValueError("Unexpected coalition bit value.")

def find_nash_equilibria_v2(df_results: pd.DataFrame):

    df_results = df_results.copy()
    df_results = remove_redundant_information(df_results)


    df_results['Combination'] = df_results['Combination'].astype(str)
    n_clients = max(df_results['Combination'].apply(lambda x: len(x)))
    df_results['Combination'] = df_results['Combination'].apply(lambda x: x.zfill(n_clients))

    # Only keep the grand coalition
    grand_coalition = "1" * n_clients
    df_results = df_results[df_results['Combination'].apply(lambda x: x == grand_coalition or x.count("1") == 1)]
    df_results = df_results.set_index('Combination')


    is_nash = True
    for player in range(1, n_clients + 1):
        if evaluate_deviation(df_results, grand_coalition, player):
            is_nash = False
            break

    if is_nash:
    
        df_ne = df_results.loc[[grand_coalition]]
    else:
        df_ne = pd.DataFrame()  
    
    return df_ne.reset_index(drop=False)
