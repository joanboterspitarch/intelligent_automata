import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class Inteligent_Neighborhood(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Inteligent_Neighborhood, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
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

def Training(data, N, m, epochs=5, hidden_size=4, lr=0.5, inc=1):

    neigh_relation = Inteligent_Neighborhood(input_size=N**2+m, hidden_size=hidden_size, output_size=N**2)
    #neigh_relation.load_state_dict(torch.load('pesos.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neigh_relation.parameters(), lr=lr)
    
    neigh_relation.train()

    l = []

    for epoch in range(epochs):
        #print('Epoch: ', epoch)
        loss = 0
        for key in data.keys():
            print('Incendio: ', key)
            r = 0 # counter
            x, y = data[key]
            initial_state = y[:, :, 0].copy()
            y = y[:, :, 1:].copy()
            train = x.Train.values.copy()
            x = x[['Temperatura', 'Humedad', 'Rho', 'Theta']].copy()

            # generate a new sample
            outcome = Samples_Model(x, initial_state, neighborhood_fn=neigh_relation, inc=inc)

            # compute the relative frequency of each state
            X0, X1, X2 = Rel_Freq(outcome)

            # stores probabilities
            indices = np.argwhere(train == True).flatten()
            for ind in indices:
                prob_estimates = torch.stack([X0[:, ind], X1[:, ind], X2[:, ind]], dim=1)

                # compute the loss
                loss += criterion(prob_estimates.clone(), torch.from_numpy(y[:, :, r]).long().flatten().clone()).clone()
                r += 1
        
        l.append(loss.item())
        # update the parameters
        optimizer.zero_grad()
        print('Calculo de gradiente ...')
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, 'Loss: ', loss.item())
    
    return neigh_relation, prob_estimates, X0, X1, X2, np.array(l)


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

def Samples_Model(x, initial_state, neighborhood_fn=None, n_it=25, inc=1):

    N = initial_state.shape[0] # number of states
    K = x.shape[0] # number of iterations

    outcome_montecarlo = torch.zeros(size=(n_it, N**2, K+1)) # store the outcome of the model

    for i in range(n_it):
        outcome_montecarlo[i, :, :] = Model(x, initial_state, neighborhood_fn=neighborhood_fn, seed=i, inc=inc).clone()
    
    return outcome_montecarlo

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

def Model(x, initial_state, seed, neighborhood_fn=None, inc=1):

    N = initial_state.shape[0] # number of states
    K = x.shape[0] # number of iterations

    np.random.seed(seed)
    torch.manual_seed(seed)

    y = torch.from_numpy(initial_state).float().flatten().clone() # initial state
    outcome = torch.zeros(size=(N**2, K+1)).float() # store the outcome of the model
    outcome[:, 0] = y.flatten().clone() # initial state
    cont = y.clone() # initial state

    for k in range(K):
        
        inputs = torch.cat([y.flatten(), torch.from_numpy(x.iloc[k].values.astype(np.float32)).float()], dim=0).clone()
        neighborhood = neighborhood_fn(inputs) # probability of infection
        probs = torch.stack([neighborhood, 1-neighborhood], dim=1).log()
        gumbel = F.gumbel_softmax(logits=probs, tau=1, hard=True)[:, 0].to(dtype=torch.float)
        
        # update the state

        y = update(gumbel, y, cont, inc).clone()
        outcome[:, k+1] = y.flatten().clone()
        cont[y==1] += 1

    return outcome      