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





def train_model_prox(model,global_model, train_loader, loss_fn, optimizer,client ,rho,Z_weights,x_hats, epoch,mu=0.01):
    model.train()
    for i in range(epoch):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Add the proximal term
            # for param, param_global in zip(model.parameters(), global_model.parameters()):
            #     loss += (mu / 2) * torch.norm(param - param_global, p=2)**2


            for z_weight, (param, param_global) in zip(Z_weights[int(client[-1])-1], zip(model.parameters(), global_model.parameters())):
                loss += (mu / 2) * torch.norm(param - param_global, p=2)**2 + torch.sum(z_weight * (param - param_global))

            # Now 'inner_product' contains the inner product between model.parameters() and global_model.parameters()

            loss.backward(retain_graph=True)  # Set retain_graph=True to retain the computation graph
            optimizer.step()



def update_z_parameter(local_model,global_model,Z_weights,x_hats,client,rho):
    # Update Z_weight
    for i, (z_weight, (param, param_global)) in enumerate(zip(Z_weights[int(client[-1])-1], zip(local_model.parameters(), global_model.parameters()))):
        Z_weights[int(client[-1])-1][i] += rho * (param - param_global)
    temp = copy.copy(x_hats[int(client[-1])-1])
    # temp2 = copy.copy(x_hats[int(client[-1])-1])

    # Update x_hats
    # for x_hat, z_weight, param_model in zip(x_hats[int(client[-1])-1],(Z_weights[int(client[-1])-1], local_model.parameters()) ):
    #     x_hat = param_model +  z_weight/rho
    for i, (x_hat, z_weight, param_model) in enumerate(zip(x_hats[int(client[-1])-1], Z_weights[int(client[-1])-1], local_model.parameters())):
        x_hats[int(client[-1])-1][i] = param_model + z_weight / rho
    # print(x_hats[int(client[-1])-1])
    # print(delta_x_hats[int(client[-1])-1])


    return Z_weights,x_hats
mu = 0.01
rho =0.01
all_avg = []
all_std = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




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


                print('|=======================|')
                print('|Traditional FedADMM 2022|')
                print('|=======================|')
                smlp_global = SimpleMLP(X.shape[1], 1)
                global_model = smlp_global
                all_results = list()

                # create optimizer
                lr = 0.00001
                loss = nn.BCELoss()
                optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.9, weight_decay=lr/comms_round)
                Z_weights=[]
                x_hats = []
                for i in range(number_of_clients):
                    Z_weight = [torch.zeros_like(param) for param in global_model.parameters()]
                    Z_weights.append(Z_weight)
                    x_hat = [torch.zeros_like(param) for param in global_model.parameters()]
                    x_hats.append(x_hat)
                # In your communication round loop
                for comm_round in range(comms_round):
                    # get the global model's weights - will serve as the initial weights for all local models
                    global_weights = [param.data.clone() for param in global_model.parameters()]
                    # initial list to collect local model weights after scaling
                    scaled_local_weight_list = list()
                    scaled_x_hats_list = list()                
                    # randomize client data - using keys
                    client_names = list(clients_batched.keys())
                    random.shuffle(client_names)
                    
                    # Client selection: select a subset (e.g., 50% of clients)
                    selection_rate = 0.8
                    num_selected = max(1, int(selection_rate * len(client_names)))
                    selected_clients = client_names[:num_selected]
                    print(selected_clients)
                    for client in selected_clients:
                        smlp_local = SimpleMLP(X.shape[1], 1)
                        local_model = smlp_local
                        # set local model weights to the global model's weights
                        local_model.load_state_dict({name: param.clone() for name, param in zip(local_model.state_dict(), global_weights)})
                        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
                
                        # fit local model with client's data
                        train_loader = DataLoader(
                            TensorDataset(torch.tensor(clients_batched[client].dataset.tensors[0], dtype=torch.float32),
                                          torch.tensor(clients_batched[client].dataset.tensors[1], dtype=torch.float32)),
                            batch_size=32, shuffle=True)
                

                        train_model_prox(local_model, global_model, train_loader, loss, optimizer, client, rho, Z_weights, x_hats, epoch, mu)
                        Z_weights,x_hats = update_z_parameter(local_model,global_model,Z_weights,x_hats,client,rho)
   
                
                        # scale the model weights and add to the list
                        scaling_factor = weight_scalling_factor(clients_batched, client)
                        scaled_weights = scale_model_weights(local_model.state_dict().values(), scaling_factor)
                        scaled_local_weight_list.append(scaled_weights)
                

                        # scale the model weights and add to the list
                        scaling_factor = weight_scalling_factor(clients_batched, client)
                        # scaled_weights = scale_model_weights(local_model.state_dict().values(), scaling_factor)
                        scaled_x_hats = scale_model_weights(x_hats[int(client[-1])-1], scaling_factor)
                        # print(local_model.state_dict().values())
                        scaled_x_hats_list.append(scaled_x_hats)
                        # scaled_local_weight_list.append(scaled_weights)

                        # clear session to free memory after each communication round
                        torch.cuda.empty_cache()

                    # ..
    
                    average_weights = sum_scaled_weights(scaled_x_hats_list)

                    # update global model
                    for param, avg_param in zip(global_model.parameters(), average_weights):
                        param.data.copy_(avg_param)
                
                    # Evaluate global model and collect results as before...
                    for X_test_batch, Y_test_batch in test_batched:
                        global_acc, global_loss, global_f1, global_precision, global_recall, global_auc, global_fpr, global_specificity = test_model(X_test_batch, Y_test_batch, global_model, comm_round)
                        all_results.append([global_acc, global_loss, global_f1, global_precision, global_recall, global_auc, global_fpr, global_specificity])
                
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.6f} seconds")
                # Create the directory if it does not exist
                print(f"number of {number_of_clients} clients and round  {comms_round} and dataset {dataset[d]} took {elapsed_time}")
                directory = f'results/round-{r}/{cl}-clients'
                os.makedirs(directory, exist_ok=True)
                all_R = pd.DataFrame(all_results, columns=['global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
                flname = f'results/round-{r}/{cl}-clients/FedADMM-{dataset[d]}-{epoch}-results.csv'
                all_R.to_csv(flname, index=None)

                all_avg.append(np.concatenate(([dataset[d], r, cl], np.mean(all_results, axis=0))))  # Storing avg values for each dataset
                all_std.append(np.concatenate(([dataset[d], r, cl], np.std(all_results, axis=0))))  # Storing std values for each dataset

    ALL_AVG = pd.DataFrame(all_avg, columns=['Dataset', 'num of round', 'num of clients', 'global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
    ALL_AVG.to_csv(f'FedADMM-{epoch}-results.csv', index=None)

    ALL_STD = pd.DataFrame(all_std, columns=['Dataset', 'num of round', 'num of clients', 'global_acc', 'global_loss', 'global_f1', 'global_precision', 'global_recall', 'global_auc', 'global_fpr', 'global_specificity'])
    ALL_STD.to_csv(f'FedADMM-{epoch}-all-std-results.csv', index=None)


