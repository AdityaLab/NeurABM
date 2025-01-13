import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
import numpy as np
import warnings
import random
import pickle
import math
import csv

def fill_in_taus(A, P, H, L, taus):
    
    tau_p2p, tau_p2h, tau_p2l, tau_h2p, tau_h2h, tau_h2l, tau_l2p, tau_l2h = taus
    
    tau_matrix = torch.zeros(P+H+L,P+H+L)

    tau_matrix[:P,:P] = tau_p2p
    tau_matrix[:P,P:P+H] = tau_h2p
    tau_matrix[:P,P+H:P+H+L] = tau_l2p
    
    tau_matrix[P:P+H,:P] = tau_p2h
    tau_matrix[P:P+H,P:P+H] = tau_h2h
    tau_matrix[P:P+H,P+H:P+H+L] = tau_h2l
    
    tau_matrix[P+H:P+H+L,:P] = tau_p2l
    tau_matrix[P+H:P+H+L,P:P+H] = tau_h2l

    return A*tau_matrix

def transfer_matrix(As, P, H, L, phi_p, phi_h, phi_l, taus):

    Rs = []
    N = P+H+L
    
    pat_idx = torch.arange(P)
    hcw_idx = torch.arange(P, P+H)
    loc_idx = torch.arange(P+H, N)

    row_scale = torch.ones(N)
    row_scale[:P] = phi_p
    row_scale[P:P+H] = phi_h
    row_scale[P+H:] = phi_l
    D = torch.diag(row_scale, 0)
    
    for A in As:

        R = fill_in_taus(A, P, H, L, taus)
        R = D * R

        Rs.append(R)
            
    return Rs

def GetImportation(As, P, H, L):
    
    ImportationIndex = []
    
    for t in range(len(As)):
        if (t == 0):
            degree = np.squeeze(np.asarray((As[t] > 0).sum(axis=0)))[:P]
            degree[np.where(degree > 0)[0]] = 1
            
            incoming = np.zeros(P+H+L)
            incoming[np.where(degree == 1)[0]] = 1

            ImportationIndex.append(incoming)
        else:
            
            degree = np.squeeze(np.asarray((As[t] > 0).sum(axis=0)))[:P]
            degree[np.where(degree > 0)[0]] = 1
            
            yesterday_degree = np.squeeze(np.asarray((As[t-1] > 0).sum(axis=0)))[:P]
            yesterday_degree[np.where(yesterday_degree > 0)[0]] = 1
            
            incoming = np.zeros(P+H+L)
            incoming[np.where((degree - yesterday_degree) == 1)[0]] = 1

            ImportationIndex.append(incoming)
  
    return np.array(ImportationIndex)
