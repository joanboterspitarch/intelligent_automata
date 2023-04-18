import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from nn_relation_neig import *
import os
import sys
import pickle


class Inteligent_Neighborhood(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Inteligent_Neighborhood, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.linear2 = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = torch.where(x == 1, x, 0) # filter \varphi
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

def Data(path):

    with open(path, 'rb') as f:
        dict_data = pickle.load(f)
    
    # remove Bolulla from dict data
    dict_data.pop('Bolulla')

    for key in dict_data.keys():
        a, b = dict_data[key]
        a.Train[0] = False
        a = a[['Temperatura', 'Humedad', 'Rho', 'Theta', 'Train']].copy()
    
    N = b.shape[0]
    m = a.shape[1] - 1

    return N, m, dict_data

def Inputs(state, weather, k):

    x = weather[['Temperatura', 'Humedad', 'Rho', 'Theta']].copy()
    x = torch.from_numpy(x.iloc[k].values.astype(np.float32)).float()
    input = torch.cat(
        [state.flatten(), x],
        dim=0
        ).clone()
    
    return input

def update(gumbel, y, cont, inc):

    outcome = torch.where(
        y == 1,
        torch.where(
            cont == inc,
            y + 1,
            y
        ),
        torch.where(
            y == 0,
            gumbel,
            y
        )
    )

    return outcome

def Model(x, y0, seed, neighborhood_fn=None, inc=1):

    N = y0.shape[0] # number of states
    K = x.shape[0] # number of iterations

    np.random.seed(seed)
    torch.manual_seed(seed)

    #y0 = torch.from_numpy(y0).float()
    y = y0.flatten().clone() # initial state
    outcome = torch.zeros(size=(N**2, K+1)).float() # store the outcome of the model
    outcome[:, 0] = y.flatten().clone() # initial state
    cont = y.clone() # initial state

    for k in range(K):
        
        # compute probability of infection
        inputs = Inputs(state=y, weather=x, k=k)
        neighborhood = neighborhood_fn(inputs)
        probs = torch.stack([neighborhood, 1-neighborhood], dim=1).log()
        gumbel = F.gumbel_softmax(logits=probs, tau=1, hard=True)[:, 0].to(dtype=torch.float)
        
        # update the state
        y = update(gumbel, y, cont, inc).clone()
        outcome[:, k+1] = y.flatten().clone()
        cont[y==1] += 1
    return outcome

def Rel_Freq(samples):

    size, K = samples.shape[1], samples.shape[2] # number of samples

    # compute the relative frequency of each state
    X0 = torch.zeros(size=(size, K)).float()
    X1 = torch.zeros(size=(size, K)).float()
    X2 = torch.zeros(size=(size, K)).float()

    for i in range(K):
        sample = samples[:, :, i]
        X0[:, i] = torch.where(sample == 0, sample + 1, 0).mean(dim=0)
        X1[:, i] = torch.where(sample == 1, sample, 0).mean(dim=0)
        X2[:, i] = torch.where(sample == 2, sample - 1, 0).mean(dim=0)
    
    return X0, X1, X2

def Samples_Model(x, y0, neighborhood_fn=None, n_it=25, inc=1):

    N = y0.shape[0] # number of states
    K = x.shape[0] # number of iterations

    outcome_montecarlo = torch.zeros(size=(n_it, N**2, K+1)) # store the outcome of the model

    for i in range(n_it):
        outcome_montecarlo[i, :, :] = Model(x=x, y0=y0, neighborhood_fn=neighborhood_fn, seed=i, inc=inc).clone()
    
    return outcome_montecarlo


##################################################

path = 'data\\dict_data_final.pkl'
N, m , dict_data = Data(path=path)

###### train data ######

lr = 2
hidden_size = 4
neigh_relation = Inteligent_Neighborhood(input_size=N**2+m, hidden_size=hidden_size, output_size=N**2)
#neigh_relation.load_state_dict(torch.load('pesos_modelo.pt'))
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neigh_relation.parameters(), lr=lr)
neigh_relation.train()
l = []
########################

inc = 1 # number of iterations as infected to become dead
#dict_data = {k: dict_data[k] for k in ['Artana', 'Beneixama']}
gamma = 0.975
dict_save = {}

for epoch in range(epochs):
    print('Epoch: ', epoch)
    optimizer = torch.optim.Adam(neigh_relation.parameters(), lr=lr, weight_decay=1e-5)
    loss_total = 0
    optimizer.zero_grad()
    for key in dict_data.keys():
        loss = 0
        print('Incendio: ', key)
        x, y = dict_data[key]
        y = torch.from_numpy(y).long()
        y0 = y[:, :, 0].clone()
        train = x.Train.values.copy()
        x = x[['Temperatura', 'Humedad', 'Rho', 'Theta']].copy()
        # generate a new sample
        outcome = Samples_Model(x=x, y0=y0, neighborhood_fn=neigh_relation, inc=inc)
        # compute the relative frequency of each state
        X0, X1, X2 = Rel_Freq(outcome)
        # stores probabilities
        indices = np.argwhere(train == True).flatten()
        r = 0 # counter
        for ind in indices:
            # compute the loss for each state with target
            prob_estimates = torch.stack([X0[:, ind], X1[:, ind], X2[:, ind]], dim=1)
            # compute the loss
            loss += (gamma)**(r) * criterion(prob_estimates.clone(), y[:, :, r + 1].flatten()).clone()
            r += 1
        print('Incendio: ', key, 'Loss: ', loss.item())
        loss.backward()
        loss_total += loss.item()
        if epoch == epochs - 1:
            dict_save[key] = (X0.clone(), X1.clone(), X2.clone(), y.clone(), loss.item())
            plt.imshow(X0[:, -1].reshape(N, N).detach().numpy())
            plt.title('X0 de ' + key)
            plt.show()
    lr = lr/1.25
    l.append(loss_total)
    optimizer.step()
    print('Epoch: ', epoch, 'Loss: ', loss_total)

torch.save(neigh_relation.state_dict(), 'pesos_modelo_2.pt')
l = np.array(l)
plt.plot(l)
plt.show()