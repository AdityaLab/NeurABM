import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':

    with open('data/NeurABM.pkl', 'rb') as pkl:
        NeurABM = pickle.load(pkl)

    with open('data/SIS-ABM.pkl', 'rb') as pkl:
        SIS = pickle.load(pkl)

    with open('data/NN.pkl', 'rb') as pkl:
        NN = pickle.load(pkl)

    with open('data/Tree.pkl', 'rb') as pkl:
        Tree = pickle.load(pkl)

    with open('data/NB.pkl', 'rb') as pkl:
        NB = pickle.load(pkl)

    with open('data/XGBoost.pkl', 'rb') as pkl:
        XGBoost = pickle.load(pkl)

    with open('data/Autoencoder+KNN.pkl', 'rb') as pkl:
        AEKNN = pickle.load(pkl)

    with open('data/Stay.pkl', 'rb') as pkl:
        Stay = pickle.load(pkl)
            
    with open('data/SILI-ABM.pkl', 'rb') as pkl:
        SILI = pickle.load(pkl)

    Data = [NeurABM,SIS,NN,Tree,NB,XGBoost,AEKNN,Stay,SILI]
    
    Method = ['NeurABM','SIS-ABM','Feedforward neural network','Decision tree','Naive bayes','XGBoost','Autoencoder + KNN','Length of stay','SILI-ABM']

    plt.figure(figsize=(10.24,8.96))

    for i in range(len(Method)):
        if i == 0:
            plt.plot(Data[i]['x'], Data[i]['y'], color='red', linewidth=3, label=str(Method[i]))
        elif i == 3:
            plt.plot(Data[i]['x'], Data[i]['y'], color='C0', linewidth=3, label=str(Method[i]))
        else:
            plt.plot(Data[i]['x'], Data[i]['y'], color='C'+str(i), linewidth=3, label=str(Method[i]))
      
    plt.axvline(x = 0.25,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axvline(x = 0.5,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axvline(x = 0.75,color='#929591',linewidth=1,linestyle="dashdot")

    plt.axhline(y = 0.25,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axhline(y = 0.5,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axhline(y = 0.75,color='#929591',linewidth=1,linestyle="dashdot")

    for ii in np.arange(0.05, 1.01, 0.05):
        plt.axvline(x=ii,color='#929591',linewidth=1,alpha=0.3,linestyle="dashdot")
    for ii in np.arange(0.05, 1.01, 0.05):
        plt.axhline(y=ii,color='#929591',linewidth=1,alpha=0.3,linestyle="dashdot")
        
    plt.xlabel('Percentage of screened patients (%)',fontsize=24)
    plt.ylabel('Percentage of observed carriers (%)',fontsize=24)
    plt.xticks([0,0.2,0.4,0.6,0.8,1],[0,20,40,60,80,100],fontsize=24)
    plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,20,40,60,80,100],fontsize=24)
    plt.xlim(0,1)
    plt.xlim(0,1)
    plt.legend(loc=4,fontsize=18)

    plt.tight_layout()
    plt.savefig('Figure5.pdf')
