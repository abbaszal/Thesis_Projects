import os
import random
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# local imports
from utils.aggregate_functions import aggregate_lr_models
from utils.evaluate_coalitions import evaluate_coalitions
from utils.aggregate_functions import FederatedForest
from utils.DecisionTree import DecisionTree
from utils.Nash import find_nash_equilibria_v2





def load_and_preprocess_global(train_pattern, test_pattern, subsample_test_size=None, test_random_state=42):

    df_train = pd.concat([pd.read_csv(train_pattern.format(i=i)) for i in range(1, 11)]).dropna()
    X_train = df_train.drop('act', axis=1)
    y_train = df_train['act']
    df_test = pd.concat([pd.read_csv(test_pattern.format(i=i)) for i in range(1, 11)]).dropna()
    X_test = df_test.drop('act', axis=1)
    y_test = df_test['act']

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #subsample of the test set (just for evaluating part)
    if subsample_test_size is not None:
        X_test_scaled, _, y_test_enc, _ = train_test_split(X_test_scaled, y_test_enc, train_size=subsample_test_size, random_state=test_random_state,stratify=y_test_enc)
        print(f"Subsampled test set to {subsample_test_size} samples.")

    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, le, scaler





def train_models(args, X_test_global, y_test_global, label_encoder, trial=None, trial_seed=None, max_iter=None, sample_size=None):


    client_models = []
    client_global_acc = {} 
    for client_idx in range(args.n_clients):

        df_local = pd.read_csv(args.train_pattern.format(i=client_idx+1)).dropna(subset=['act'])
        df_local, _ = train_test_split(df_local, train_size=sample_size, random_state=trial_seed, stratify=df_local['act'])
        df_local = df_local.reset_index(drop=True).dropna()
        X_local = df_local.drop('act', axis=1).values 
        y_local = label_encoder.transform(df_local['act'])
        rs = (args.random_seed + trial) 
        #max_iter = args.max_iters
        if max_iter is None:
          max_iter = args.max_iters



        if args.model_type == 'logistic':

            model = LogisticRegression(random_state=rs, max_iter=max_iter)
            local_scaler = StandardScaler()
            X_scaled = local_scaler.fit_transform(X_local)
            model.fit(X_scaled, y_local)
            client_models.append(model)
            client_global_acc[client_idx] = model.score(X_test_global, y_test_global)

        else:

            model = DecisionTree(max_depth=max_iter, random_state=rs)
            model.fit(X_local, y_local)
            client_models.append(model)
            y_pred = model.predict(X_test_global)
            acc = accuracy_score(y_test_global, y_pred)
            client_global_acc[client_idx] = acc


    return client_models, client_global_acc






AGG_MAP = {
    'fedlr': aggregate_lr_models,
    'fedfor': FederatedForest
}


def basic_run (args):

    # Global preprocessing
    X_train_global, X_test_global, y_train_global, y_test_global, label_encoder, scaler = \
        load_and_preprocess_global(args.train_pattern, args.test_pattern, subsample_test_size=None)

    results = []
    for trial in range(args.n_trials):
        print(f"Trial {trial+1}/{args.n_trials}")

        models, accs = train_models(
            args, X_test_global, y_test_global, label_encoder, trial=trial
        )

        df_res = evaluate_coalitions(
            client_models= models,
            client_global_accuracies= accs,
            n_clients=args.n_clients,
            aggregator_func=AGG_MAP[args.approach],
            X_test=X_test_global,
            y_test=y_test_global,
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






def evaluating_run(args):

    # Global preprocessing with subsample
    X_train_global, X_test_global, y_train_global, y_test_global, label_encoder, scaler = \
        load_and_preprocess_global(args.train_pattern, args.test_pattern, subsample_test_size=args.subsample_test_size, test_random_state=args.random_seed)

    all_details = []
    for max_iter in args.max_iters:
        print(f"Running evaluating experiment: max_iter or max_depth={max_iter}")
        nash_counts = Counter()
        details_for_param = []
        client_acc_details = []

        for trial in range(args.n_trials):

            rand_comp = random.randint(0, 500)
            trial_seed = args.random_seed + trial + int(1000 * max_iter) + 2 * rand_comp
            random.seed(trial_seed)
            np.random.seed(trial_seed)


            models, accs = train_models(
                args, X_test_global, y_test_global, label_encoder, trial= trial,
                trial_seed=trial_seed, max_iter=max_iter, sample_size=args.sample_size
            )

            df_res = evaluate_coalitions(
                client_models=models,
                client_global_accuracies=accs,
                n_clients=args.n_clients,
                aggregator_func=AGG_MAP[args.approach],
                X_test=X_test_global,
                y_test=y_test_global,
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
            details_for_param.append(df_nash)


            acc_record = {'Trial': trial+1, 'Max Iter or Depth': max_iter}
            for idx in range(args.n_clients):
                acc_record[f'Client {idx+1} Accuracy'] = accs.get(idx, np.nan)
            client_acc_details.append(acc_record)


        df_counts = pd.DataFrame(nash_counts.items(), columns=['Nash Equilibrium', 'Occurrences'])
        df_counts['Max Iter or Depth'] = max_iter
        os.makedirs(args.save_dir, exist_ok=True)
        counts_path = os.path.join(args.save_dir,f"Nash_Equilibrium_Counts_{args.approach}_maxiter_{max_iter}.csv")
        df_counts.to_csv(counts_path, index=False)


        df_details = pd.concat(details_for_param, ignore_index=True)
        df_acc = pd.DataFrame(client_acc_details)
        df_combined = df_details.merge(df_acc, on=['Trial', 'Max Iter or Depth'], how='left')
        all_details.append(df_combined)

    final_df = pd.concat(all_details, ignore_index=True)
    details_path = os.path.join(args.save_dir, f"Nash_Equilibrium_Details_{args.approach}.csv")
    final_df.to_csv(details_path, index=False)
    print(f"details saved to {details_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Basic
    p_basic = subparsers.add_parser('basic')
    p_basic.add_argument('--train_pattern', type=str,default='data/metadata/train_{i:02d}.csv')
    p_basic.add_argument('--test_pattern', type=str,default='data/metadata/test_{i:02d}.csv')
    p_basic.add_argument('--n_clients', type=int, default=10)
    p_basic.add_argument('--n_trials', type=int, default=1)
    p_basic.add_argument('--random_seed', type=int, default=42)
    p_basic.add_argument('--max_iters', type=int, default=100)
    p_basic.add_argument('--approach',   choices=['fedlr','fedfor'])
    p_basic.add_argument('--model_type', type=str ) #,default='logistic')
    p_basic.add_argument('--save_dir',    type=str, default=None,)
    p_basic.add_argument('--base_out',    type=str, default=None,)

    # Evaluating
    p_eval = subparsers.add_parser('evaluating')
    p_eval.add_argument('--train_pattern', type=str,default='data/metadata/train_{i:02d}.csv')
    p_eval.add_argument('--test_pattern', type=str,default='data/metadata/test_{i:02d}.csv')
    p_eval.add_argument('--subsample_test_size', type=int, default=950)
    p_eval.add_argument('--random_seed', type=int, default=42)
    p_eval.add_argument('--n_clients', type=int, default=10)
    p_eval.add_argument('--n_trials', type=int, default=50)
    p_eval.add_argument('--sample_size', type=int, default=350)
    p_eval.add_argument('--max_iters',type=int,nargs='+',default=[10, 100])
    p_eval.add_argument('--approach',  choices=['fedlr','fedfor'])
    p_eval.add_argument('--model_type',type=str ) #,default='logistic')
    p_eval.add_argument('--save_dir',    type=str, default=None,)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'basic':
        if args.approach == 'fedlr':
            args.save_dir = 'results/FedLR_HuGaDB_without_LQC'
            args.base_out = 'HuGaDB_results_with_LR.csv'
        else:  # fedfor
            args.save_dir =  'results/FedFor_HuGaDB_without_LQC'
            args.base_out = 'HuGaDB_results_with_FedFor.csv'

    elif args.command == 'evaluating':
        if args.approach == 'fedlr':
            args.save_dir = 'results/FedLR_HuGaDB_without_LQC/evaluate_grand_combination_without_LQC'
        else:  # fedfor
            args.save_dir ='results/FedFor_HuGaDB_without_LQC/evaluate_grand_combination_without_LQC'

    if args.command == 'basic':
        basic_run(args)
    else:
        evaluating_run(args)
