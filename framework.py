import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import networkx as nx
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import csv
import sys

from model import fill_in_taus, transfer_matrix, GetImportation
'''
class masked_loss(nn.Module):
    def __init__(self,all_weight):
        super(masked_loss, self).__init__()
        self.all_weight = all_weight

    def forward(self, y_pred, y_true, model):
        mask = (y_true != 0)
        
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        y_true_normalized = (y_true + 1) / 2
        
        loss = F.binary_cross_entropy(y_pred,y_true,weight=torch.tensor(self.all_weight))
        return loss
'''
# The neural network to predict the importation probability for each patient using each patients' EHR
class Importation(nn.Module):
    def __init__(self, feature_dim, output_dim=1):
        super(Importation, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.sigmoid(out)
        out = torch.clamp(out, min=0, max=1)
        return out

class Parameter(nn.Module):
    def __init__(self, feature_dim, parameter_dim=18):
        super(Parameter, self).__init__()

        self.para = nn.Parameter(torch.rand(parameter_dim))
        
        self.low = torch.tensor([0,0,\
                   0,0,\
                   0,0,0,\
                   0.9,0.9,0.9,\
                   0,0,0,0,0,0,0,0])
        self.up = torch.tensor([10,10,\
                    10,10,\
                    2,0.02,0.1,\
                    1,1,1,\
                    0.02,0.02,0.02,0.02,0.002,0.02,0.02,0.01])

        self.low.requires_grad = False
        self.up.requires_grad = False

    def forward(self):
        #output = self.para

        output = self.low + (self.up-self.low)*torch.sigmoid(self.para)

        return output

# Agent based model simulator
class ABM(nn.Module):
    def __init__(self, P, H, L):
        super(ABM, self).__init__()
        self.P = P
        self.H = H
        self.L = L

    def simulate(self, Rt, states, loads, t, imp, parameter):
        #imp represents the new importaion probability, should be preprocessed to ensure
        #only new incoming patients have non-zero value
        #(may not need)rembmber to set states=0 for non-in-hospital patients
        #The ABM initialziation is not here, but should use predicted importation probabiliy to init

        mu_init,sigma_init,\
        mu_imp,sigma_imp,\
        alpha,beta,delta,\
        phi_p,phi_h,phi_l,\
        tau_p2p,tau_p2h,tau_p2l,tau_h2p,tau_h2h,tau_h2l,tau_l2p,tau_l2h = parameter

        #rands = torch.rand(P)

        states[:self.P] = ((1-delta)*states.clone() + beta*loads.clone()*(1-states.clone()) + imp.view(-1,1))[:self.P]

        states[:self.P] = torch.clamp(states.clone()[:self.P], min=0, max=1)
        
        loads = torch.matmul(Rt, loads)
        loads = loads + alpha*states

        #print (t,states)
        #print (t,loads)

        return states, loads

class Network(nn.Module):
    def __init__(self, feature_dim, As, P, H, L, ImportationIndex):
        super(Network, self).__init__()
        self.ImportationNN = Importation(feature_dim,1)
        self.ParameterNN = Parameter(len(As),18)
        self.ABM = ABM(P,H,L)
        self.As = As
        self.P = P
        self.H = H
        self.L = L

        for t in range(len(self.As)):
            self.As[t].requires_grad = False

        self.ImportationIndex = ImportationIndex
        self.ImportationIndex.requires_grad = False

    def forward(self, features):

        probability = self.ImportationNN.forward(features)
        parameter = self.ParameterNN.forward()

        mu_init,sigma_init,\
        mu_imp,sigma_imp,\
        alpha,beta,delta,\
        phi_p,phi_h,phi_l,\
        tau_p2p,tau_p2h,tau_p2l,tau_h2p,tau_h2h,tau_h2l,tau_l2p,tau_l2h = parameter.clone()

        taus = (tau_p2p, tau_p2h, tau_p2l,
                tau_h2p, tau_h2h, tau_h2l,
                tau_l2p, tau_l2h)

        Rs = transfer_matrix(self.As, self.P, self.H, self.L, phi_p, phi_h, phi_l, taus)

        states_init = self.ImportationIndex[0].view(-1,1).clone()
        states_init[:self.P] = states_init[:self.P].clone() * probability.clone()

        loads_init = states_init.view(-1,1).clone()*self.ImportationIndex[0].view(-1,1).clone() * torch.normal(mu_init*torch.ones(self.P+self.H+self.L), sigma_init*torch.ones(self.P+self.H+self.L)).view(-1,1).clone()
        states_all = states_init.view(-1,1).clone()
        loads_all = loads_init.view(-1,1).clone()

        for t in range(1,len(self.As)):
            #print (t)

            states_imp = self.ImportationIndex[t].view(-1,1).clone()
            states_imp[:self.P] = states_imp[:self.P].clone() * probability.clone()
            
            states_new, loads_new = self.ABM.simulate(Rs[t], states_all[:,t-1].view(-1,1), loads_all[:,t-1].view(-1,1), t, states_imp.clone(), parameter.clone())

            states_new = torch.clamp(states_new, min=0, max=1)
            states_all = torch.cat((states_all, states_new.clone()), dim=1)
            loads_all = torch.cat((loads_all, loads_new.clone()), dim=1)
            
        return (states_all[:self.P,:]).T, probability, parameter
