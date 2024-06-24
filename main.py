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

torch.autograd.set_detect_anomaly(True)

from framework import Importation, Parameter, ABM, Network
from model import fill_in_taus, transfer_matrix, GetImportation

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--outputfile', type=str)
    parser.add_argument('--startdate', type=int)
    parser.add_argument('--enddate', type=int)
    parser.add_argument('--predictdate', type=int, default=7)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print ('Reading data...')

    with open(args.datafile, 'rb') as pkl:
        FullGs, FeatureDict, CaseDict, mapping = pickle.load(pkl)

    print ('Processing data...')

    FullNodeList = []
    for counter in mapping.keys():
        FullNodeList.append(mapping[counter])

    FullP,FullH,FullL = 0,0,0
    for NodeNow in FullNodeList:
        if ('MRN' in NodeNow):
            FullP = FullP + 1
        elif ('wID' in NodeNow):
            FullH = FullH + 1
        else:
            FullL = FullL + 1
    N = FullP + FullH + FullL

    Gs = FullGs[args.startdate-1:args.enddate+args.predictdate]
    As = []
    for G in Gs:
        A = nx.to_scipy_sparse_matrix(G, nodelist=FullNodeList, dtype=np.float64, format='coo')
        A = A.tocsr()
        As.append(A)

    Degree = np.zeros(N)
    for i in range(len(As)):
        Degree = Degree + np.squeeze(np.asarray((As[i] > 0).sum(axis=0)))

    Index = np.where(Degree > 0)[0]
    NodeList = np.array(FullNodeList)[np.where(Degree > 0)[0]]
    NodeList = NodeList.tolist()

    P,H,L = 0,0,0
    for NodeNow in NodeList:
        if ('MRN' in NodeNow):
            P = P + 1
        elif ('wID' in NodeNow):
            H = H + 1
        else:
            L = L + 1

    As = []
    for G in Gs:
        A = nx.to_scipy_sparse_matrix(G, nodelist=NodeList, dtype=np.float64, format='coo')
        A = A.tocsr()
        As.append(A)

    GsPred = FullGs[args.enddate:args.enddate+args.predictdate]

    AsPred = []
    for G in GsPred:
        A = nx.to_scipy_sparse_matrix(G, nodelist=FullNodeList, dtype=np.float64, format='coo')
        A = A.tocsr()
        AsPred.append(A)

    Degree = np.zeros(N)
    for i in range(len(AsPred)):
        Degree = Degree + np.squeeze(np.asarray((AsPred[i] > 0).sum(axis=0)))

    Index = np.where(Degree > 0)[0]
    NodeListPred = np.array(FullNodeList)[np.where(Degree > 0)[0]]
    NodeListPred = NodeListPred.tolist()

    PPred,HPred,LPred = 0,0,0
    for NodeNow in NodeListPred:
        if ('MRN' in NodeNow):
            PPred = PPred + 1
        elif ('wID' in NodeNow):
            HPred = HPred + 1
        else:
            LPred = LPred + 1

    AsPred = []
    for G in GsPred:
        A = nx.to_scipy_sparse_matrix(G, nodelist=NodeList, dtype=np.float64, format='coo')
        A = A.tocsr()
        AsPred.append(A)

    PredIndex = []
    for p in range(PPred):
        assert (NodeListPred[p] == NodeList[NodeList.index(NodeListPred[p])])
        PredIndex.append(NodeList.index(NodeListPred[p]))

    assert (len(PredIndex) == PPred)
    
    T = len(As)

    Feature = []
    FeatureName = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10']#,'Imported_Case','Ward']
    
    y_true = []
    
    Cases = []

    for p in range(P):
        FeatureNow = []
        for f in FeatureName:
            FeatureNow.append(FeatureDict[NodeList[p]][f])
        Feature.append(FeatureNow)
        y_true.append(FeatureDict[NodeList[p]]['Imported_Case'])
               
        Cases.append(CaseDict[NodeList[p]])

    Feature = np.array(Feature)
    Cases = np.array(Cases).T

    Cpt_gt = Cases[args.startdate-1:args.enddate+args.predictdate,:]
    Cpt_gt = np.where(Cpt_gt == -1, 0, Cpt_gt)
    
    Cases = Cases[args.startdate-1:args.enddate,:]

    ImportationIndex = GetImportation(As, P, H, L)
    
    pos_weight = np.count_nonzero(Cases == -1)/np.count_nonzero(Cases == 1)
    all_weight = np.copy(Cases)
    all_weight = np.where(all_weight == 1, args.weight*pos_weight, all_weight)
    all_weight = np.where(all_weight == -1, 1, all_weight)
    Cases = np.where(Cases == -1, 0, Cases)

    As_tensor = []
    for t in range(T):
        As_tensor.append(torch.tensor(As[t].toarray()).float())

    ImportationIndex = torch.tensor(ImportationIndex).float()
    Feature = torch.tensor(Feature).float()
    Cases = torch.tensor(Cases).float()

    print ('Training...')

    Model = Network(len(FeatureName),As_tensor,P,H,L,ImportationIndex)
    Model.to(device)

    loss_fn = nn.BCELoss(weight=torch.tensor(all_weight))
    optimizer = optim.Adam(Model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):

        optimizer.zero_grad()

        output, prob, para = Model.forward(Feature)
        loss = loss_fn(output[:-args.predictdate,:], Cases)

        loss.backward(retain_graph=True)
        optimizer.step()
        
        print ('Epoch {}, total loss {:.4f}'.format(epoch,loss.item()))

    Model.eval()

    imp_pred = prob.detach().numpy()
    imp_true = np.array(y_true)
    case_pred = output.detach().numpy()[-1,:]
    case_true = np.array(Cpt_gt)[-1,:]

    print (imp_pred.shape)
    print (imp_true.shape)
    print (case_pred.shape)
    print (case_true.shape)

    with open(args.outputfile, 'wb') as pkl:
        pickle.dump((imp_pred,imp_true,case_pred,case_true), pkl)

    print ('Results saved in ' + args.outputfile + '.')
