import os
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)




# local imports
from utils.Spambase.split_data import split_data_equal
from utils.aggregate_functions import aggregate_lr_models
from utils.evaluate_coalitions import evaluate_coalitions
from utils.aggregate_functions import FederatedForest
from utils.DecisionTree import DecisionTree
from utils.Nash import find_nash_equilibria_v2




def load_and_preprocess_spambase(file_path, test_size, seed):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test





def train_models(partitions, seed, X_test, y_test, max_iter):
    client_models = []
    client_global_accuracies = []
    for X_i, y_i in partitions:
        mask = ~np.isnan(X_i).any(axis=1)
        X_clean, y_clean = X_i[mask], y_i[mask]
        if len(y_clean) == 0:
            client_models.append(None)
            client_global_accuracies.append(None)
            continue
        if args.model_type == 'logistic':
            model = LogisticRegression(random_state=seed,max_iter=max_iter)
            try:
                local_scaler = StandardScaler()
                model.fit(local_scaler.fit_transform(X_clean), y_clean)
                client_models.append(model)
                client_global_accuracies.append(model.score(X_test, y_test))
            except Exception:
                client_models.append(None)
                client_global_accuracies.append(None)
        else:
            model = DecisionTree(max_depth=max_iter, random_state=seed)
            model.fit(X_i, y_i)
            client_models.append(model)
            y_pred = model.predict(X_test)
            client_global_accuracies.append(accuracy_score(y_pred ,y_test))

    return client_models, client_global_accuracies



AGG_MAP = {
    'fedlr': aggregate_lr_models,
    'fedfor': FederatedForest
}


def basic_run(args, X_train, y_train, X_test, y_test):
    partitions = split_data_equal(X_train, y_train,n_clients=args.n_clients,shuffle=True,random_seed=args.random_seed)
    results = []
    for trial in range(args.n_trials):
        print(f"Trial {trial+1}/{args.n_trials}")
        models, accs = train_models(partitions, args.random_seed, X_test, y_test, args.max_iter)
        df_res = evaluate_coalitions(client_models=models,client_global_accuracies=accs,
            n_clients=args.n_clients,
            aggregator_func=AGG_MAP[args.approach],
            X_test=X_test,
            y_test=y_test,
            corrupt_client_indices=[],
            approach=args.approach
        )
        df_res['Trial'] = trial + 1
        results.append(df_res)
    df_all = pd.concat(results, ignore_index=True)
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.base_out)
    df_all.to_csv(out_path, index=False)
    print(f"results saved to {out_path}")






def evaluating_run(args, X_train, y_train, X_test, y_test):
    all_details = []
    for max_iter in args.max_iters:
        print(f"Running evaluating experiment: max_iter or max_depth={max_iter}")
        nash_counts = Counter()
        details = []
        acc_details = []
        for trial in range(args.n_trials):
            seed_comp = random.randint(0, 500)
            trial_seed = args.random_seed + trial + int(1000*max_iter) + 2*seed_comp
            partitions = split_data_equal( X_train, y_train, n_clients=args.n_clients, shuffle=True,random_seed=trial_seed)
            models, accs = train_models( partitions, trial_seed, X_test, y_test, max_iter)
            df_res = evaluate_coalitions(
                client_models=models,
                client_global_accuracies=accs,
                n_clients=args.n_clients,
                aggregator_func=AGG_MAP[args.approach],
                X_test=X_test,
                y_test=y_test,
                corrupt_client_indices=[],
                approach=args.approach
            )
            df_res['Trial'] = trial + 1
            df_res['Max Iter or Depth'] = max_iter
            df_nash = find_nash_equilibria_v2(df_res.reset_index())
            for combo in df_nash['Combination']:
                nash_counts[combo] += 1
            df_nash['Trial'] = trial + 1
            df_nash['Max Iter or Depth'] = max_iter
            details.append(df_nash)
            rec = {'Trial': trial+1, 'Max Iter or Depth': max_iter}
            for i in range(args.n_clients):
                rec[f'Client {i+1} Accuracy'] = accs[i] if accs[i] is not None else np.nan
            acc_details.append(rec)
        df_counts = pd.DataFrame(nash_counts.items(), columns=['Nash Equilibrium','Occurrences'])
        df_counts['Max Iter or Depth'] = max_iter
        os.makedirs(args.save_dir, exist_ok=True)
        cnt_path = os.path.join(
            args.save_dir, f"Nash_Equilibrium_Counts_{args.approach}_maxiter_{max_iter}.csv"
        )
        df_counts.to_csv(cnt_path, index=False)
        df_det = pd.concat(details, ignore_index=True)
        df_acc = pd.DataFrame(acc_details)
        df_comb = df_det.merge(df_acc, on=['Trial','Max Iter or Depth'], how='left')
        all_details.append(df_comb)
    final = pd.concat(all_details, ignore_index=True)
    detail_path = os.path.join(
        args.save_dir, f"Nash_Equilibrium_Details_{args.approach}.csv"
    )
    final.to_csv(detail_path, index=False)
    print(f"details saved to {detail_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # basic
    p_basic = subparsers.add_parser('basic')
    p_basic.add_argument('--file_path',   type=str,   default='data/spambase.data')
    p_basic.add_argument('--test_size',   type=float, default=0.2)
    p_basic.add_argument('--random_seed',   type=int,default=42)
    p_basic.add_argument('--n_clients',   type=int,   default=10)
    p_basic.add_argument('--n_trials',    type=int,   default=1)
    p_basic.add_argument('--max_iter',    type=int,   default=100)
    p_basic.add_argument('--approach',   choices=['fedlr','fedfor'])
    p_basic.add_argument('--model_type',  type=str ) #,default='logistic')
    p_basic.add_argument('--save_dir',    type=str, default=None,)
    p_basic.add_argument('--base_out',    type=str, default=None,)

    # evaluating
    p_eval = subparsers.add_parser('evaluating')
    p_eval.add_argument('--file_path',   type=str,   default='data/spambase.data')
    p_eval.add_argument('--test_size',   type=float, default=0.2)
    p_eval.add_argument('--random_seed', type=int, default=42)
    p_eval.add_argument('--n_clients',   type=int,   default=10)
    p_eval.add_argument('--n_trials',    type=int,   default=50)
    p_eval.add_argument('--max_iters',   type=int, nargs='+', default=[10,100])
    p_eval.add_argument('--approach',  choices=['fedlr','fedfor'])
    p_eval.add_argument('--model_type',  type=str ) #,default='logistic')
    p_eval.add_argument('--save_dir',    type=str, default=None,)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    X_train, X_test, y_train, y_test = load_and_preprocess_spambase(
        args.file_path, args.test_size, args.random_seed
    )

    if args.command == 'basic':
        if args.approach == 'fedlr':
            args.save_dir = 'results/FedLR_Spambase_without_LQC'
            args.base_out = 'Spambase_results_with_LR.csv'
        else:  # fedfor
            args.save_dir =  'results/FedFor_Spambase_without_LQC'
            args.base_out = 'Spambase_results_with_FedFor.csv'

    elif args.command == 'evaluating':
        if args.approach == 'fedlr':
            args.save_dir = 'results/FedLR_Spambase_without_LQC/evaluate_grand_combination_without_LQC'
        else:  # fedfor
            args.save_dir ='results/FedFor_Spambase_without_LQC/evaluate_grand_combination_without_LQC'

    if args.command == 'basic':
        basic_run(args, X_train, y_train, X_test, y_test)
    else:
        evaluating_run(args, X_train, y_train, X_test, y_test)

