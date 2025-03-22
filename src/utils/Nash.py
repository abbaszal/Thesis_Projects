import pandas as pd
import re



def remove_redundant_information(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: No need to have "Client i accuracy" columns. The same information can be retrieved either from:
    - the global accuracy for the one-hot coalition #i `(0...1...0)`, if the i-th client does not join the current coalition
    - the global accuracy of the current coalition, if the i-th client joins the current coalition
    """
    redundant_cols = [col for col in df_results.columns if re.match(r"Client \d+ Accuracy", col) is not None]
    df_results = df_results.drop(columns=redundant_cols)
    return df_results


def one_hot_coalition(player: int, n_players: int) -> str:
    """
    Args:
        player (int): Player index (between `1` and `n_players`).

    Return:
        one_hot (str): one-hot coalition where only `player` is participating.
    """
    one_hot = ["0"] * n_players
    one_hot[player - 1] = "1"
    one_hot = "".join(one_hot)
    return one_hot

def evaluate_deviation(df_results: pd.DataFrame, coalition: str, player: int) -> bool:
    """
    Args:
        df_results (pd.DataFrame): DataFrame containing the results (should be indexed by the coalition)
        coalition (str): binary string representing the coalition.
        player (int): Player whose deviation should be evaluated (index between `1` and `n_players`).

    Return:
        wants_to_deviate (bool): True if player has incentive to deviate, otherwise False.
    """

    current_move = coalition[player - 1]
    
    one_hot_player = one_hot_coalition(player, len(coalition))
    standalone_payoff = df_results.loc[one_hot_player, "Global Accuracy"]
    if current_move == "0":
        # If current move is 0 (standalone training), the player has incentive to deviate if coalition payoff is higher
        new_coalition = list(coalition)
        new_coalition[player - 1] = "1"
        new_coalition = "".join(new_coalition)
        coalition_payoff = df_results.loc[new_coalition, "Global Accuracy"]
        return coalition_payoff + 0.000001 > standalone_payoff

    elif current_move == "1":
        coalition_payoff = df_results.loc[coalition, "Global Accuracy"]
        return standalone_payoff +  0.000001> coalition_payoff

    else:
        raise ValueError("This should not happen")




def find_nash_equilibria_v2(df_results: pd.DataFrame):

    df_results = df_results.copy()
    df_results = remove_redundant_information(df_results)

    df_results['Combination'] = df_results['Combination'].astype(str)
    n_clients = max(df_results['Combination'].apply(lambda x: len(x)))
    df_results['Combination'] = df_results['Combination'].apply(lambda x: x.zfill(n_clients))

    df_results = df_results.set_index('Combination')

    nash_equilibria = []

    for coalition in df_results.index:

        is_nash = True
        for player in range(1, n_clients+1):
            if (evaluate_deviation(df_results, coalition, player)):  # check if player has incentive for deviation
                is_nash = False
                break
        if is_nash:
            nash_equilibria.append(coalition)

    df_ne = df_results.loc[nash_equilibria]
    df_ne = df_ne.reset_index(drop=False)

    return df_ne
