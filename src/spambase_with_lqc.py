import os
import random
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore",category=ConvergenceWarning)



#local imports
from utils.Spambase.split_data import split_data_equal
from utils.Spambase.corrupt_data_spambase import corrupt_data, corrupt_clients
from utils.aggregate_functions import aggregate_lr_models, FederatedForest
from utils.evaluate_coalitions import evaluate_coalitions
from utils.Nash import find_nash_equilibria_v2




def prepare_partitions(X_train,y_train,n_clients,random_seed,corruption_settings,noise_std,corrupt_client_indices,corrupt_function):

    parts=split_data_equal(X_train,y_train,n_clients=n_clients,shuffle=True,random_seed=random_seed)
    corrupted, _=corrupt_clients(corrupt_function,parts,corrupt_client_indices,
        corruption_prob=corruption_settings.get("corruption_prob",0.8),
        nan_prob=corruption_settings.get("nan_prob",0.5),
        noise_std=noise_std,
        label_corruption_prob=corruption_settings.get("label_corruption_prob",0.2),
        base_seed=random_seed)
    
    norm_parts=[]
    
    for Xp,yp in corrupted:
        ls=StandardScaler();Xp_n=ls.fit_transform(Xp)
        norm_parts.append((Xp_n,yp))
    return norm_parts



def train_models_fedlr(parts,random_seed,X_test,y_test,max_iter):

    models=[];accs=[]

    for Xp,yp in parts:

        mask=~np.isnan(Xp).any(axis=1)
        Xc, yc = Xp[mask], yp[mask]

        if len(yc)==0:
            models.append(None);accs.append(None);continue
        m=LogisticRegression(random_state=random_seed,max_iter=max_iter)

        try:
            m.fit(Xc,yc);models.append(m);accs.append(m.score(X_test,y_test))
        except:
            models.append(None);accs.append(None)

    return models,accs

def train_models_fedfor(parts,X_test,y_test,max_depth):

    models=[];accs={}

    for i,(Xp,yp) in enumerate(parts):

        m=DecisionTreeClassifier(max_depth=max_depth,random_state=np.random.randint(0,100000))
        m.fit(Xp,yp);models.append(m)
        p=m.predict(X_test);accs[i]=np.mean(p==y_test)

    return models,accs



def run_trial(approach,trial_seed,n_clients,X_train,y_train,X_test,y_test,hyper_param,
              noise_std,corrupt_client_indices,corruption_settings,corrupt_function):

    parts=prepare_partitions(X_train,y_train,n_clients,trial_seed,corruption_settings,noise_std,corrupt_client_indices,corrupt_function)

    if approach=="fedlr":
        models,accs=train_models_fedlr(parts,trial_seed,X_test,y_test,max_iter=hyper_param)
        df= evaluate_coalitions(models,accs,n_clients,aggregate_lr_models,X_test,y_test,corrupt_client_indices,approach)
        return df,accs
    
    elif approach=="fedfor":
        models,accs=train_models_fedfor(parts,X_test,y_test,max_depth=hyper_param)
        df= evaluate_coalitions(models,accs,n_clients,FederatedForest,X_test,y_test,corrupt_client_indices,approach)
        return df,accs






def run_experiment(approach, n_trials, n_clients, hyper_params, partitions_corrupted, corrupt_client_indices,
                    X_train, y_train, X_test, y_test, noise_std, save_dir, corrupt_function=corrupt_data, corruption_settings=None, base_random_seed=42):

    os.makedirs(save_dir, exist_ok=True)
    all_details = []

    for hyper_param in hyper_params:

        details_for_this_param = []
        client_accuracy_details = []

        if corrupt_client_indices is None:
            corrupt_client_indices = np.random.choice( n_clients, size=partitions_corrupted, replace=False)

        for trial in range(n_trials):

            rand_component = random.randint(0, 500)
            trial_seed = ( base_random_seed + trial + int(1000 * hyper_param) + 2 * rand_component)
            df_results, client_global_acc = run_trial(approach, trial_seed, n_clients, X_train, y_train, X_test, y_test, hyper_param, 
                                                      noise_std, corrupt_client_indices, corruption_settings, corrupt_function)



            df_nash = find_nash_equilibria_v2(df_results.reset_index())
            df_nash['Trial'] = trial + 1
            df_nash['Noise Std'] = noise_std
            df_nash['Corrupted Clients'] = len(corrupt_client_indices)
            df_nash['Max Iter or Depth'] = hyper_param
            details_for_this_param.append(df_nash)


            trial_acc = {
                'Trial': trial + 1,
                'Max Iter or Depth': hyper_param,
                'Noise Std': noise_std,
                'Corrupted Clients': len(corrupt_client_indices)
            }
            for j in range(n_clients):
                col_name = f'Client {j+1} Accuracy'
                if j in corrupt_client_indices:
                    col_name += ' (low-quality client)'

                if approach == 'fedlr':
                    trial_acc[col_name] = (
                        client_global_acc[j]
                        if client_global_acc[j] is not None
                        else np.nan
                    )
                else:
                    trial_acc[col_name] = client_global_acc.get(j, np.nan)

            client_accuracy_details.append(trial_acc)


        df_details = pd.concat(details_for_this_param, ignore_index=True)
        df_client_acc = pd.DataFrame(client_accuracy_details)
        df_combined = df_details.merge(
            df_client_acc,
            on=['Trial', 'Max Iter or Depth', 'Noise Std', 'Corrupted Clients'],
            how='left'
        )
        all_details.append(df_combined)

    final_details_df = pd.concat(all_details, ignore_index=True)

    details_path = os.path.join(save_dir,f"Nash_Equilibrium_Details_{approach}_noise_{noise_std}"f"_c{len(corrupt_client_indices)}.csv")
    if not os.path.exists(details_path):
        final_details_df.to_csv(details_path, index=False)

    return final_details_df




def parse_args():

    import argparse
    p=argparse.ArgumentParser()

    p.add_argument('--file_path',type=str,default='data/spambase.data')
    p.add_argument('--approach',choices=['fedlr','fedfor'])
    p.add_argument('--n_trials',type=int,default=50)
    p.add_argument('--n_clients',type=int,default=10)
    p.add_argument('--hyper_params',type=lambda s:list(map(int,s.split(','))),default=[10,100])
    p.add_argument('--noise_stds',type=lambda s:list(map(float,s.split(','))),default=[0.1,0.3,0.5,0.7,1,2,3,4,5])
    p.add_argument('--partitions_corrupted',type=lambda s: list(map(int, s.split(','))), default=None)
    p.add_argument('--corruption_prob',type=float,default=0.8)
    p.add_argument('--nan_prob',type=float,default=0.5)
    p.add_argument('--label_corruption_prob',type=float,default=0.2)
    p.add_argument('--save_dir',type=str,default=None)

    return p.parse_args()


if __name__=='__main__':

    args=parse_args()

    if args.partitions_corrupted is None:
        args.partitions_corrupted = list(range(args.n_clients + 1))

    # load data
    df=pd.read_csv(args.file_path,header=None)
    X=df.iloc[:,:-1].to_numpy();y=df.iloc[:,-1].to_numpy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler();X_train=scaler.fit_transform(X_train);X_test=scaler.transform(X_test)

    if not args.save_dir:
        base='results'
        name=f"{'FedLR' if args.approach=='fedlr' else 'FedFor'}_Spambase_LQC_0_to_10"
        multiplier = len(args.hyper_params)
        args.save_dir=os.path.join(base,name+(f"_{args.n_trials  * multiplier }Trials" if args.n_trials!=50 else""))


    results={noise:[] for noise in args.noise_stds}

    corruption_settings={
        'corruption_prob':args.corruption_prob,
        'nan_prob':args.nan_prob,
        'label_corruption_prob':args.label_corruption_prob
    }
    for noise in args.noise_stds:
        for cc in args.partitions_corrupted:
            client_indices = list(range(cc))
            path=run_experiment( args.approach,args.n_trials,args.n_clients,args.hyper_params,cc,
                client_indices,X_train,y_train,X_test,y_test,noise,args.save_dir,corrupt_data,corruption_settings)
            count=(path['Combination']==''.join(['1']*args.n_clients)).sum()
            results[noise].append(count)
            print(f"Noise Std: {noise},Bad Clients: {cc},Occurrences: {count}")

    results_df=pd.DataFrame(results, index= args.partitions_corrupted)
    results_df.index.name='Number of Bad Clients'
    out_csv=os.path.join(args.save_dir,'nash_occurrence_results.csv')
    results_df.to_csv(out_csv)
    print(f"Results saved to {out_csv}")
