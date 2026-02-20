import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import os
import copy
import time

# Assuming these functions are defined in federated_utils_fedavg_copy.py
from federated_utils_fedavg_copy import *

# Declare paths
drebin_data_path = 'data/drebin.csv'
malgenome_data_path = 'data/malgenome.csv'
kronodroid_data_path = 'data/kronodroid.csv'
TUANDROMD_data_path = 'data/TUANDROMD.csv'

# --- Helper Functions for Metrics ---

def compute_weight_norm(model_params, reference_params):
    """
    Computes the L2 norm of the difference between two sets of parameters.
    This serves as a proxy for the gradient/update magnitude.
    """
    total_norm = 0.0
    for p, ref in zip(model_params, reference_params):
        param_diff = p.data - ref.data
        total_norm += torch.norm(param_diff, p=2).item() ** 2
    return np.sqrt(total_norm)

def compute_variance(values):
    """Computes the variance of a list of scalar values."""
    if len(values) < 2:
        return 0.0
    return np.var(values)

# --- Model Training Functions ---

def train_model(model, train_loader, loss_fn, optimizer, epochs):
    """
    Train local model (standard FedAvg - no proximal term).
    """
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

# --- Main Execution ---

all_avg = []
all_std = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

epoch_list = [32]
n_clients = [5,10,15,20]
n_round = [10]
dataset = ['Drebin', 'Malgenome', 'Kronodroid', 'Tuandromd']
seed = 123

# Load Data
Drebin_data = pd.read_csv(drebin_data_path, header=None)
Malgenome_data = pd.read_csv(malgenome_data_path)
Tuandromd_data = pd.read_csv(TUANDROMD_data_path)
kronodroid_data = pd.read_csv(kronodroid_data_path)
Kronodroid_data = kronodroid_data.iloc[:, range(1, kronodroid_data.shape[1])]

for epoch in epoch_list:
    setup_seed(seed)
    for d in range(len(dataset)):
        if d == 0:
            use_data = Drebin_data
        elif d == 1:
            use_data = Malgenome_data
        elif d == 2:
            use_data = Kronodroid_data
        elif d == 3:
            use_data = Tuandromd_data

        print('===================================================================================================')
        print('Working with:', dataset[d])
        print('===================================================================================================')

        for r in n_round:
            comms_round = r
            for cl in n_clients:
                number_of_clients = cl
                start_time = time.time()

                print('---------------------------------------------')
                print('No. of Clients:', number_of_clients)
                print('No. of Rounds:', comms_round)
                print('---------------------------------------------')

                features = np.array(use_data.iloc[:, range(0, use_data.shape[1] - 1)])
                labels = use_data.iloc[:, -1]

                X = preprocessing.StandardScaler().fit(features).transform(features)
                lb = LabelBinarizer()
                y = lb.fit_transform(labels)

                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=100)

                clients = create_clients(X_train, y_train, num_clients=number_of_clients, initial='client')
                clients_batched = dict()
                for (client_name, data) in clients.items():
                    clients_batched[client_name] = batch_data(data)

                test_batched = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.float32)
                    ),
                    batch_size=len(y_test), shuffle=False
                )

                print('|=======================|')
                print('|Traditional FedAvg 2017|')
                print('|=======================|')
                
                smlp_global = SimpleMLP(X.shape[1], 1)
                global_model = smlp_global
                all_results = list()

                lr = 0.00001
                loss_fn = nn.BCELoss()
                optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=lr/comms_round)

                for comm_round in range(comms_round):
                    # Get global model weights to be distributed among the local models
                    global_weights = [param.data.clone() for param in global_model.parameters()]
                    scaled_local_weight_list = []
                    
                    # List to store update norms for variance calculation
                    round_update_norms = []

                    # Randomize client data and select a subset of clients
                    client_names = list(clients_batched.keys())
                    random.shuffle(client_names)
                    
                    selection_rate = 0.8
                    num_selected = max(1, int(selection_rate * len(client_names)))
                    selected_clients = client_names[:num_selected]
                    
                    for client in selected_clients:
                        smlp_local = SimpleMLP(X.shape[1], 1)
                        local_model = smlp_local
                        
                        # Set local model weights to the global model's weights
                        local_model.load_state_dict({name: param.clone() for name, param in zip(local_model.state_dict(), global_weights)})
                        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)

                        train_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(clients_batched[client].dataset.tensors[0], dtype=torch.float32),
                                torch.tensor(clients_batched[client].dataset.tensors[1], dtype=torch.float32)
                            ),
                            batch_size=32, shuffle=True
                        )
                        
                        train_model(local_model, train_loader, loss_fn, optimizer, epoch)

                        # 1. Calculate Update Norm (Proxy for Gradient Norm)
                        # Difference between local weights after training and global weights at start of round
                        update_norm = compute_weight_norm(local_model.parameters(), global_weights)
                        round_update_norms.append(update_norm)

                        # 2. Scale and collect the local weights for aggregation
                        scaling_factor = weight_scalling_factor(clients_batched, client)
                        scaled_weights = scale_model_weights(local_model.state_dict().values(), scaling_factor)
                        scaled_local_weight_list.append(scaled_weights)

                        torch.cuda.empty_cache()

                    # 3. Calculate Variance of Updates for this round
                    avg_update_norm = np.mean(round_update_norms) if round_update_norms else 0.0
                    variance_update_norm = compute_variance(round_update_norms)

                    # Update the global model by averaging the local models' weights
                    average_weights = sum_scaled_weights(scaled_local_weight_list)
                    for param, avg_param in zip(global_model.parameters(), average_weights):
                        param.data.copy_(avg_param)

                    # Evaluate global model and collect results
                    for X_test_batch, Y_test_batch in test_batched:
                        global_acc, global_loss, global_f1, global_precision, global_recall, global_auc, global_fpr, global_specificity = test_model(
                            X_test_batch, Y_test_batch, global_model, comm_round
                        )
                        
                        # Append all metrics including new ones
                        all_results.append([
                            global_acc, 
                            global_loss, 
                            global_f1, 
                            global_precision, 
                            global_recall, 
                            global_auc, 
                            global_fpr, 
                            global_specificity,
                            avg_update_norm,      # New: Avg Gradient/Update Norm
                            variance_update_norm  # New: Variance of Updates
                        ])

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.6f} seconds")
                print(f"number of {number_of_clients} clients and round {comms_round} and dataset {dataset[d]} took {elapsed_time}")
                
                directory = f'results/round-{r}/{cl}-clients'
                os.makedirs(directory, exist_ok=True)
                
                # Updated Columns for CSV
                all_R = pd.DataFrame(all_results, columns=[
                    'global_acc', 'global_loss', 'global_f1', 'global_precision', 
                    'global_recall', 'global_auc', 'global_fpr', 'global_specificity',
                    'avg_update_norm', 'update_variance'
                ])
                
                flname = f'results/round-{r}/{cl}-clients/FedAvg-{dataset[d]}-{epoch}-results.csv'
                all_R.to_csv(flname, index=None)

                # Prepare summary stats
                all_avg.append(np.concatenate(([dataset[d], r, cl], np.mean(all_results, axis=0))))
                all_std.append(np.concatenate(([dataset[d], r, cl], np.std(all_results, axis=0))))

    # Save Summary DataFrames
    ALL_AVG = pd.DataFrame(all_avg, columns=[
        'Dataset', 'num of round', 'num of clients', 
        'global_acc', 'global_loss', 'global_f1', 'global_precision', 
        'global_recall', 'global_auc', 'global_fpr', 'global_specificity',
        'avg_update_norm', 'update_variance'
    ])
    ALL_AVG.to_csv(f'FedAvg-{epoch}-results.csv', index=None)

    ALL_STD = pd.DataFrame(all_std, columns=[
        'Dataset', 'num of round', 'num of clients', 
        'global_acc', 'global_loss', 'global_f1', 'global_precision', 
        'global_recall', 'global_auc', 'global_fpr', 'global_specificity',
        'avg_update_norm', 'update_variance'
    ])
    ALL_STD.to_csv(f'FedAvg-{epoch}-all-std-results.csv', index=None)