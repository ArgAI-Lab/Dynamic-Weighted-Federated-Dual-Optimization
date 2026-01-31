import warnings
warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"


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

from federated_utils_fedavg_copy import *
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
import time

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

# Declare paths (relative to /kaggle/working/)
drebin_data_path = 'data/drebin.csv'  # Use forward slashes (/) instead of backslashes (\)
malgenome_data_path = 'data/malgenome.csv'
kronodroid_data_path = 'data/kronodroid.csv'
TUANDROMD_data_path = 'data/TUANDROMD.csv'


Drebin_data = pd.read_csv(drebin_data_path, header = None)

Malgenome_data = pd.read_csv(malgenome_data_path)

Tuandromd_data=pd.read_csv(TUANDROMD_data_path)

kronodroid_data=pd.read_csv(kronodroid_data_path)
Kronodroid_data = kronodroid_data.iloc[:,range(1,kronodroid_data.shape[1])]






def train_model(model, train_loader, loss_fn, optimizer, epochs, mu=0.01):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

all_avg = []
all_std = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

selection_rate = 0.8

epoch_list = [32]
n_clients = [20]
n_round = [10]



dataset = ['Drebin', 'Malgenome' , 'Kronodroid', 'Tuandromd']# 

seed = 123

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

        for r in n_round:  # number of rounds loop
            comms_round = r
            for cl in n_clients:  # number of clients loop
                number_of_clients = cl
                start_time = time.time()  # Record start time

                print('---------------------------------------------')
                print('No. of Clients:', number_of_clients)
                print('No. of Rounds:', comms_round)
                print('---------------------------------------------')

                features = np.array(use_data.iloc[:, range(0, use_data.shape[1] - 1)])  # feature set
                labels = use_data.iloc[:, -1]  # labels --> B : Benign and S

                # Do feature scaling
                X = preprocessing.StandardScaler().fit(features).transform(features)

                # binarize the labels
                lb = LabelBinarizer()
                y = lb.fit_transform(labels)

                # split data into training and test set
                X_train, X_test, y_train, y_test = train_test_split(X,
                                                                    y, shuffle=True,
                                                                    test_size=0.2,
                                                                    random_state=100)

                # create clients -- Horizontal FL
                clients = create_clients(X_train, y_train, num_clients=number_of_clients, initial='client')

                # process and batch the training data for each client
                clients_batched = dict()
                for (client_name, data) in clients.items():
                    clients_batched[client_name] = batch_data(data)

                # process and batch the test set
                test_batched = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                                                         torch.tensor(y_test, dtype=torch.float32)),
                                                           batch_size=len(y_test), shuffle=False)

                # ==============================================
                # Traditional FedAvg 2017
                # ==============================================
                # -----------------------------------------------
                smlp_global = SimpleMLP(X.shape[1], 1)
                global_model = smlp_global
                all_results = list()

                # create optimizer
                lr = 0.00001
                loss = nn.BCELoss()
                optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=lr/comms_round)

                # initialize global model

                # -----------------------------------------------

                print('|=======================|')
                print('|Traditional FedAvg 2017|')
                print('|=======================|')

                # --- before starting rounds, initialize per-client storage ---
                client_names = list(clients_batched.keys())
                # initialize each client's "previous" weights to the initial global weights
                global_weights = [param.data.clone() for param in global_model.parameters()]
                prev_local_weights = {
                    c: [w.clone() for w in global_weights]
                    for c in client_names
                }
                # give everyone a uniform “previous” accuracy so ω_i,1 = 1
                local_acc = {c: 1.0 for c in client_names}
                # --- federated rounds with dynamic weighting ---
                for comm_round in range(comms_round):
                    # recompute best previous accuracy across all clients
                    best_acc = max(local_acc.values()) if max(local_acc.values()) > 0 else 1.0
                    # grab the latest global weights
                    global_weights = [param.data.clone() for param in global_model.parameters()]
                    scaled_local_weight_list = []
                    # optionally shuffle & select subset
                    random.shuffle(client_names)
                    num_selected = max(1, int(selection_rate * len(client_names)))
                    selected_clients = client_names[:num_selected]
                    for client in selected_clients:
                        # 1) compute dynamic coefficient ω
                        ω = local_acc[client] / best_acc
                        # 2) build a fresh local model and warm‑start it
                        local_model = SimpleMLP(X.shape[1], 1)
                        # create a new state_dict by mixing
                        new_state = {}
                        for idx, (name, param) in enumerate(local_model.state_dict().items()):
                            g_w = global_weights[idx]
                            p_w = prev_local_weights[client][idx]
                            new_state[name] = ω * g_w + (1.0 - ω) * p_w
                        local_model.load_state_dict(new_state)
                        # 3) train from the warm‑start
                        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
                        train_loader = DataLoader(
                            TensorDataset(
                                torch.tensor(clients_batched[client].dataset.tensors[0], dtype=torch.float32),
                                torch.tensor(clients_batched[client].dataset.tensors[1], dtype=torch.float32)
                            ),
                            batch_size=32, shuffle=True
                        )
                        train_model(local_model, train_loader, loss, optimizer, epoch)
                        # 4) evaluate on the client’s own data to get new Acc
                        #    (you can re‑use your test_model but on the local train_loader)
                        tot_correct, tot_seen = 0, 0
                        local_model.eval()
                        with torch.no_grad():
                            for Xb, yb in train_loader:
                                preds = (local_model(Xb) > 0.5).float()
                                tot_correct += (preds == yb).all(dim=1).sum().item()
                                tot_seen += yb.size(0)
                        acc_i = tot_correct / tot_seen
                        local_acc[client] = acc_i
                        # 5) store for next round
                        prev_local_weights[client] = [p.data.clone() for p in local_model.parameters()]
                        # 6) scale & collect for global aggregation
                        scaling_factor = weight_scalling_factor(clients_batched, client)
                        scaled = scale_model_weights(local_model.state_dict().values(), scaling_factor)
                        scaled_local_weight_list.append(scaled)
                        torch.cuda.empty_cache()
                    # --- standard averaging of scaled_local_weight_list to update global_model ---
                    average_weights = sum_scaled_weights(scaled_local_weight_list)
                    for param, avg_param in zip(global_model.parameters(), average_weights):
                        param.data.copy_(avg_param)

                    # Evaluate global model and collect results as before...
                    for X_test_batch, Y_test_batch in test_batched:
                        global_acc, global_loss, global_f1, global_precision, global_recall, global_auc, global_fpr, global_specificity = test_model(X_test_batch, Y_test_batch, global_model, comm_round)
                        all_results.append([global_acc, global_loss, global_f1, global_precision, global_recall, global_auc, global_fpr, global_specificity])
                
                # Create the directory if it does not exist
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.6f} seconds")
                # Create the directory if it does not exist
                print(f"number of {number_of_clients} clients and round  {comms_round} and dataset {dataset[d]} took {elapsed_time}")                
                directory = f'results/round-{r}/{cl}-clients'
                os.makedirs(directory, exist_ok=True)
                all_R = pd.DataFrame(all_results, columns=['global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
                flname = f'results/round-{r}/{cl}-clients/DW-FedAvg-{dataset[d]}-{epoch}-results.csv'
                all_R.to_csv(flname, index=None)

                all_avg.append(np.concatenate(([dataset[d], r, cl], np.mean(all_results, axis=0))))  # Storing avg values for each dataset
                all_std.append(np.concatenate(([dataset[d], r, cl], np.std(all_results, axis=0))))  # Storing std values for each dataset

    ALL_AVG = pd.DataFrame(all_avg, columns=['Dataset', 'num of round', 'num of clients', 'global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
    ALL_AVG.to_csv(f'DW-FedAvg-{epoch}-results.csv', index=None)

    ALL_STD = pd.DataFrame(all_std, columns=['Dataset', 'num of round', 'num of clients', 'global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
    ALL_STD.to_csv(f'DW-FedAvg-{epoch}-all-std-results.csv', index=None)

