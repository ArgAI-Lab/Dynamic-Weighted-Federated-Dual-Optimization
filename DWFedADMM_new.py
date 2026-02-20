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
    """Computes the L2 norm of the difference between two sets of parameters."""
    total_norm = 0.0
    for p, ref in zip(model_params, reference_params):
        param_diff = p.data - ref.data
        total_norm += torch.norm(param_diff, p=2).item() ** 2
    return np.sqrt(total_norm)

def compute_variance(values):
    """Computes the variance of a list of scalar values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance

def compute_dynamic_weight(client_acc, best_acc):
    return client_acc / best_acc if best_acc > 0 else 1.0

def calculate_dynamic_weight(acc_i_r_minus_1, best_acc_r_minus_1):
    return acc_i_r_minus_1 / best_acc_r_minus_1 if best_acc_r_minus_1 > 0 else 1.0

def update_local_model_with_dynamic_weight(client_model, global_model, accuracy_i_r_minus_1, best_accuracy_r_minus_1):
    omega_i_r = calculate_dynamic_weight(accuracy_i_r_minus_1, best_accuracy_r_minus_1)
    for local_param, global_param in zip(client_model.parameters(), global_model.parameters()):
        local_param.data = omega_i_r * global_param.data + (1 - omega_i_r) * local_param.data
    return client_model

def evaluate_client_accuracy(client_model, X_test, y_test):
    client_model.eval()
    with torch.no_grad():
        outputs = client_model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = accuracy_score(y_test, predicted)
    return accuracy

def train_model_prox(model, global_model, train_loader, loss_fn, optimizer, client, rho, Z_weights, x_hats, epoch, mu=0.01):
    model.train()
    for i in range(epoch):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Add the proximal term and ADMM terms
            for z_weight, (param, param_global) in zip(Z_weights[int(client[-1])-1], zip(model.parameters(), global_model.parameters())):
                loss += (mu / 2) * torch.norm(param - param_global, p=2)**2 + torch.sum(z_weight * (param - param_global))

            loss.backward(retain_graph=True)
            optimizer.step()

def update_z_parameter(local_model, global_model, Z_weights, x_hats, client, rho):
    for i, (z_weight, (param, param_global)) in enumerate(zip(Z_weights[int(client[-1])-1], zip(local_model.parameters(), global_model.parameters()))):
        Z_weights[int(client[-1])-1][i] += rho * (param - param_global)
    
    for i, (x_hat, z_weight, param_model) in enumerate(zip(x_hats[int(client[-1])-1], Z_weights[int(client[-1])-1], local_model.parameters())):
        x_hats[int(client[-1])-1][i] = param_model + z_weight / rho

    return Z_weights, x_hats

# --- Main Execution ---

mu = 0.01
rho = 0.01
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
                print('|Traditional FedADMM 2022|')
                print('|=======================|')
                
                smlp_global = SimpleMLP(X.shape[1], 1)
                global_model = smlp_global
                all_results = list()

                lr = 0.00001
                loss_fn = nn.BCELoss()
                optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=lr/comms_round)
                
                Z_weights = []
                x_hats = []
                client_names = list(clients_batched.keys())

                prev_local_weights = {client: [param.data.clone() for param in global_model.parameters()] for client in client_names}
                client_accuracies = {client: 1.0 for client in client_names}

                for i in range(number_of_clients):
                    Z_weight = [torch.zeros_like(param) for param in global_model.parameters()]
                    Z_weights.append(Z_weight)
                    x_hat = [torch.zeros_like(param) for param in global_model.parameters()]
                    x_hats.append(x_hat)

                for comm_round in range(comms_round):
                    best_acc = max(client_accuracies.values())
                    global_weights = [param.data.clone() for param in global_model.parameters()]
                    
                    scaled_local_weight_list = list()
                    scaled_x_hats_list = list()
                    
                    # Lists to store new metrics for this round
                    round_update_norms = [] 
                    
                    random.shuffle(client_names)

                    selection_rate = 0.8
                    num_selected = max(1, int(selection_rate * len(client_names)))
                    selected_clients = client_names[:num_selected]
                    
                    for client in selected_clients:
                        smlp_local = SimpleMLP(X.shape[1], 1)
                        local_model = smlp_local
                        
                        # 1. Apply Dynamic Weighting to initialize local model
                        # Note: Removed the overwrite bug from original code where global weights re-loaded immediately after mixing
                        omega = compute_dynamic_weight(client_accuracies[client], best_acc)
                        new_state_dict = {}
                        for idx, (name, param) in enumerate(local_model.state_dict().items()):
                            g_w = global_weights[idx]
                            p_w = prev_local_weights[client][idx]
                            new_state_dict[name] = omega * g_w + (1.0 - omega) * p_w
                        local_model.load_state_dict(new_state_dict)

                        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
                
                        train_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(clients_batched[client].dataset.tensors[0], dtype=torch.float32),
                                torch.tensor(clients_batched[client].dataset.tensors[1], dtype=torch.float32)
                            ),
                            batch_size=32, shuffle=True
                        )

                        train_model_prox(local_model, global_model, train_loader, loss_fn, optimizer, client, rho, Z_weights, x_hats, epoch, mu)
                        Z_weights, x_hats = update_z_parameter(local_model, global_model, Z_weights, x_hats, client, rho)

                        # 2. Calculate Gradient/Update Norm (Difference between Local after train and Global before train)
                        update_norm = compute_weight_norm(local_model.parameters(), global_weights)
                        round_update_norms.append(update_norm)

                        # Evaluate local accuracy
                        tot_correct, tot_seen = 0, 0
                        local_model.eval()
                        with torch.no_grad():
                            for Xb, yb in train_loader:
                                preds = (local_model(Xb) > 0.5).float()
                                tot_correct += (preds == yb).all(dim=1).sum().item()
                                tot_seen += yb.size(0)
                        acc_i = tot_correct / tot_seen
                        client_accuracies[client] = acc_i

                        prev_local_weights[client] = [param.data.clone() for param in local_model.parameters()]

                        scaling_factor = weight_scalling_factor(clients_batched, client)
                        scaled_weights = scale_model_weights(local_model.state_dict().values(), scaling_factor)
                        scaled_local_weight_list.append(scaled_weights)
                
                        scaled_x_hats = scale_model_weights(x_hats[int(client[-1])-1], scaling_factor)
                        scaled_x_hats_list.append(scaled_x_hats)

                        torch.cuda.empty_cache()

                    # 3. Calculate Variance of Updates for this round
                    avg_update_norm = np.mean(round_update_norms) if round_update_norms else 0.0
                    variance_update_norm = compute_variance(round_update_norms)

                    # Aggregate
                    average_weights = sum_scaled_weights(scaled_x_hats_list)
                    for param, avg_param in zip(global_model.parameters(), average_weights):
                        param.data.copy_(avg_param)
                
                    # Evaluate global model
                    # Assuming test_model returns: acc, loss, f1, precision, recall, auc, fpr, specificity
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
                
                flname = f'results/round-{r}/{cl}-clients/DW-FedADMM-{dataset[d]}-{epoch}-results.csv'
                all_R.to_csv(flname, index=None)

                # Prepare summary stats
                # Mean and Std need to account for the new columns (index 8 and 9)
                all_avg.append(np.concatenate(([dataset[d], r, cl], np.mean(all_results, axis=0))))
                all_std.append(np.concatenate(([dataset[d], r, cl], np.std(all_results, axis=0))))

    # Save Summary DataFrames
    ALL_AVG = pd.DataFrame(all_avg, columns=[
        'Dataset', 'num of round', 'num of clients', 
        'global_acc', 'global_loss', 'global_f1', 'global_precision', 
        'global_recall', 'global_auc', 'global_fpr', 'global_specificity',
        'avg_update_norm', 'update_variance'
    ])
    ALL_AVG.to_csv(f'DW-FedADMM-{epoch}-results.csv', index=None)

    ALL_STD = pd.DataFrame(all_std, columns=[
        'Dataset', 'num of round', 'num of clients', 
        'global_acc', 'global_loss', 'global_f1', 'global_precision', 
        'global_recall', 'global_auc', 'global_fpr', 'global_specificity',
        'avg_update_norm', 'update_variance'
    ])
    ALL_STD.to_csv(f'DW-FedADMM-{epoch}-all-std-results.csv', index=None)