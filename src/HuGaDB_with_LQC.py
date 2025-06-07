import os
import random
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")



#local imports
from utils.HuGaDB.prepare_partitions import prepare_partitions
from utils.aggregate_functions import aggregate_lr_models,FederatedForest
from utils.DecisionTree import DecisionTree
from utils.evaluate_coalitions import evaluate_coalitions
from utils.Nash import find_nash_equilibria_v2


def load_and_preprocess_global(train_pattern,test_pattern,subsample_test_size=None,test_random_state=42):
    df_train=pd.concat([pd.read_csv(train_pattern.format(i=i)) for i in range(1,11)]).dropna()
    X_train=df_train.drop('act',axis=1)
    y_train=df_train['act']
    df_test=pd.concat([pd.read_csv(test_pattern.format(i=i)) for i in range(1,11)]).dropna()
    X_test=df_test.drop('act',axis=1)
    y_test=df_test['act']
    le=LabelEncoder()
    y_train_enc=le.fit_transform(y_train)
    y_test_enc=le.transform(y_test)
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    if subsample_test_size is not None:
        X_test_scaled,_,y_test_enc,_=train_test_split(
            X_test_scaled,y_test_enc,train_size=subsample_test_size,
            random_state=test_random_state,stratify=y_test_enc)
        print(f"Subsampled test set to {subsample_test_size} samples.")
    return X_train_scaled,X_test_scaled,y_train_enc,y_test_enc,le,scaler




def train_models_fedlr(n_clients,trial_seed,sample_size,num_corrupted_clients,train_pattern,noise_std,corruption_settings,hyper_param):

    client_models=[];client_global_accuracies={}

    for client_idx in range(1,n_clients+1):

        with warnings.catch_warnings():warnings.filterwarnings("ignore",category=ConvergenceWarning)
        X_train_scaled,y_train=prepare_partitions(client_idx,trial_seed,sample_size,num_corrupted_clients,
                                                  train_pattern,noise_std, corruption_settings,label_encoder)
        
        model=LogisticRegression(random_state=trial_seed,max_iter=hyper_param)
        model.fit(X_train_scaled,y_train)
        acc_global=accuracy_score(y_test_global,model.predict(X_test_global_scaled))
        client_models.append(model);client_global_accuracies[client_idx-1]=acc_global

    return client_models,client_global_accuracies



def train_models_fedfor(n_clients,trial_seed,sample_size,num_corrupted_clients,train_pattern,noise_std,corruption_settings,hyper_param):

    client_models=[];client_global_accuracies={}

    for client_idx in range(1,n_clients+1):

        with warnings.catch_warnings():warnings.filterwarnings("ignore",category=ConvergenceWarning)

        X_train_scaled,y_train=prepare_partitions(client_idx,trial_seed,sample_size,num_corrupted_clients,
                                                  train_pattern,noise_std,corruption_settings,label_encoder)
        model=DecisionTree(max_depth=hyper_param,random_state=trial_seed)
        model.fit(X_train_scaled,y_train)
        acc_global=accuracy_score(y_test_global,model.predict(X_test_global_scaled))
        client_models.append(model);client_global_accuracies[client_idx-1]=acc_global

    return client_models,client_global_accuracies



def run_trial(approach,n_clients,trial_seed,sample_size,num_corrupted_clients,train_pattern,corruption_settings,hyper_param,noise_std,
              aggregator_func,X_test,y_test,corrupt_client_indices):
    

    if approach=='fedlr':
        client_models,client_global_accuracies=train_models_fedlr(n_clients,trial_seed,sample_size,num_corrupted_clients,train_pattern,noise_std,
                                                                  corruption_settings,hyper_param)
        

    else:
        client_models,client_global_accuracies=train_models_fedfor(n_clients,trial_seed,sample_size,num_corrupted_clients,train_pattern,noise_std,
                                                                   corruption_settings,hyper_param)
        

    df_results=evaluate_coalitions(client_models,client_global_accuracies,n_clients,aggregator_func,X_test,y_test,corrupt_client_indices,approach)
    df_nash=find_nash_equilibria_v2(df_results.reset_index())

    return df_results,df_nash,client_global_accuracies



def run_experiment(approach,n_trials,n_clients,hyper_params,noise_std,num_corrupted_clients,save_dir,sample_size,base_random_seed,data_root,subsample_size , corruption_settings):

    train_pattern=os.path.join(data_root,'train_{:02d}.csv')
    test_pattern=os.path.join(data_root,'test_{:02d}.csv')
    #corruption_params={'corruption_prob':0.8,'nan_prob':0.5,'noise_std':noise_std}
    #label_corruption_prob=0.2
    os.makedirs(save_dir,exist_ok=True)
    all_details=[]
    corrupt_client_indices=list(range(num_corrupted_clients))
    aggregator_func=aggregate_lr_models if approach=='fedlr' else (lambda:FederatedForest())

    for hyper_param in hyper_params:

        details_for_this_param=[]
        client_accuracy_details = []

        for trial in range(n_trials):
            rand_component = random.randint(0,500)
            trial_seed = base_random_seed+trial+int(1000*hyper_param)+rand_component
            random.seed(trial_seed);np.random.seed(trial_seed)

            df_results,df_nash,client_global_accuracies=run_trial(
                approach,n_clients,trial_seed,sample_size,num_corrupted_clients,
                train_pattern,corruption_settings,hyper_param,noise_std,
                aggregator_func,X_test_global_scaled,y_test_global,corrupt_client_indices)

            df_nash['Trial'] = trial + 1
            df_nash['Noise Std'] = noise_std
            df_nash['Corrupted Clients'] = num_corrupted_clients
            df_nash['Max Iter or Depth'] = hyper_param
            details_for_this_param.append(df_nash)

            trial_acc = {
                'Trial': trial + 1,
                'Max Iter or Depth': hyper_param,
                'Noise Std': noise_std,
                'Corrupted Clients': num_corrupted_clients
            }
            for j in range(n_clients):
                col_name = f'Client {j+1} Accuracy'
                if j in corrupt_client_indices:
                    col_name += "(low-quality client)"
                trial_acc[col_name] = client_global_accuracies[j] if client_global_accuracies[j] is not None else np.nan
            client_accuracy_details.append(trial_acc)

        df_details = pd.concat(details_for_this_param, ignore_index=True)
        df_client_accuracy = pd.DataFrame(client_accuracy_details)
        df_combined = df_details.merge(
            df_client_accuracy,
            on=['Trial', 'Max Iter or Depth', 'Noise Std', 'Corrupted Clients'],
            how='left'
        )
        all_details.append(df_combined)

    final_details_df = pd.concat(all_details, ignore_index=True)
    out_path = os.path.join(save_dir, f'Nash_Equilibrium_Details_{approach}_noise_{noise_std}_c{num_corrupted_clients}.csv')
    final_details_df.to_csv(out_path, index=False)

    return final_details_df




def parse_args():

    import argparse

    p=argparse.ArgumentParser()

    p.add_argument('--data_root',type=str,default='data/metadata')
    p.add_argument('--save_dir',type=str,default=None)
    p.add_argument('--approach',choices=['fedlr','fedfor'])
    p.add_argument('--n_trials',type=int,default=50)
    p.add_argument('--n_clients',type=int,default=10)
    p.add_argument('--hyper_params',type=lambda s:list(map(int,s.split(','))),default=[10,100])
    p.add_argument('--noise_stds',type=lambda s:list(map(float,s.split(','))),default=[0.1,0.3,0.5,0.7,1,2,3,4,5])
    p.add_argument('--corrupted_clients', type=lambda s: list(map(int, s.split(','))), default=None)
    p.add_argument('--corruption_prob',type=float,default=0.8)
    p.add_argument('--nan_prob',type=float,default=0.5)
    p.add_argument('--label_corruption_prob',type=float,default=0.2)
    p.add_argument('--sample_size',type=int,default=350)
    p.add_argument('--base_random_seed',type=int,default=42)
    p.add_argument('--subsample_size',type=int,default=950)

    return p.parse_args()

if __name__=='__main__':
    
    args=parse_args()

    if args.corrupted_clients is None:
        args.corrupted_clients = list(range(args.n_clients + 1))

    
    if not args.save_dir:
        base='results'
        name=f"{'FedLR' if args.approach=='fedlr' else 'FedFor'}_HuGaDB_LQC_0_to_10"
        multiplier = len(args.hyper_params)
        args.save_dir=os.path.join(base,name+(f"_{args.n_trials  * multiplier }Trials" if args.n_trials!=50 else""))



    # load global data
    train_pattern=os.path.join(args.data_root,'train_{i:02d}.csv')
    test_pattern=os.path.join(args.data_root,'test_{i:02d}.csv')
    X_train_global_scaled,X_test_global_scaled,y_train_global,y_test_global,label_encoder,scaler=load_and_preprocess_global(
        train_pattern,test_pattern,subsample_test_size=args.subsample_size,test_random_state=42)
    


    # run experiments
    results={noise:[] for noise in args.noise_stds}


    corruption_settings={
        'corruption_prob':args.corruption_prob,
        'nan_prob':args.nan_prob,
        'label_corruption_prob':args.label_corruption_prob
    }
    
    for noise in args.noise_stds:

        for cc in args.corrupted_clients:

            df=run_experiment(
                args.approach,args.n_trials,args.n_clients,args.hyper_params,
                noise,cc,args.save_dir,args.sample_size,args.base_random_seed,
                args.data_root,args.subsample_size, corruption_settings)
            count=(df['Combination']==''.join(['1']*args.n_clients)).sum()
            results[noise].append(count)
            print(f"Noise Std: {noise},Bad Clients: {cc},Occurrences: {count}")

    results_df=pd.DataFrame(results,index=args.corrupted_clients)
    results_df.index.name='Number of Bad Clients'
    out_csv=os.path.join(args.save_dir,'nash_occurrence_results.csv')
    results_df.to_csv(out_csv)
    print(f"Results saved to {out_csv}")