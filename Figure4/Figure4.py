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

    ###### Figure 4a ######

    plt.figure(figsize=(10.24,8.96))

    plt.fill_between([0, 0.25, 0.25, 0], [0, 0, 1, 1], color='grey', alpha=0.5)
    plt.text(0.045, 0.5, 'Clinically inapplicable',color='#3F3F3F',fontsize=32,rotation=90,horizontalalignment='center', verticalalignment='center')

    for i in range(len(Method)):
        if i == 0:
            plt.plot(Data[i]['precision'], Data[i]['recall'], color='red', linewidth=3, label=str(Method[i])+', AUPRC: {:.2f}'.format(Data[i]['auprc']))
        elif i == 3:
            plt.plot(Data[i]['precision'], Data[i]['recall'], color='C0', linewidth=3, label=str(Method[i])+', AUPRC: {:.2f}'.format(Data[i]['auprc']))
        else:
            plt.plot(Data[i]['precision'], Data[i]['recall'], color='C'+str(i), linewidth=3, label=str(Method[i])+', AUPRC: {:.2f}'.format(Data[i]['auprc']))

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
       
    plt.xlabel('Precision',fontsize=32)
    plt.ylabel('Recall',fontsize=32)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'],fontsize=24)
    plt.yticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'],fontsize=24)
    plt.title('Precision-recall curve',fontsize=32)
    plt.legend(loc=1,fontsize=18)

    plt.tight_layout()
    plt.savefig('Figure4a.pdf')

    ###### Figure 4b ######

    plt.figure(figsize=(10.24,8.96))

    for i in range(len(Method)):
        if i == 0:
            plt.plot(Data[i]['NPV_threshold'], Data[i]['NPV'], color='red', linewidth=3, label=str(Method[i]))
            plt.scatter(Data[i]['NPV_threshold_25'], Data[i]['NPV_25'], marker='o', color='red', s=400)
            plt.scatter(Data[i]['NPV_threshold_50'], Data[i]['NPV_50'], marker='s', color='red', s=400)
            plt.scatter(Data[i]['NPV_threshold_75'], Data[i]['NPV_75'], marker='^', color='red', s=400)
        elif i == 3:
            plt.plot(Data[i]['NPV_threshold'], Data[i]['NPV'], color='C0', linewidth=3, label=str(Method[i]))
            plt.scatter(Data[i]['NPV_threshold_25'], Data[i]['NPV_25'], marker='o', color='C0', s=400)
            plt.scatter(Data[i]['NPV_threshold_50'], Data[i]['NPV_50'], marker='s', color='C0', s=400)
            plt.scatter(Data[i]['NPV_threshold_75'], Data[i]['NPV_75'], marker='^', color='C0', s=400)
        else:
            plt.plot(Data[i]['NPV_threshold'], Data[i]['NPV'], color='C'+str(i), linewidth=3, label=str(Method[i]))
            plt.scatter(Data[i]['NPV_threshold_25'], Data[i]['NPV_25'], marker='o', color='C'+str(i), s=400)
            plt.scatter(Data[i]['NPV_threshold_50'], Data[i]['NPV_50'], marker='s', color='C'+str(i), s=400)
            plt.scatter(Data[i]['NPV_threshold_75'], Data[i]['NPV_75'], marker='^', color='C'+str(i), s=400)
 
    plt.axvline(x = 0.25,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axvline(x = 0.5,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axvline(x = 0.75,color='#929591',linewidth=1,linestyle="dashdot")

    plt.axhline(y = 0.85,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axhline(y = 0.9,color='#929591',linewidth=1,linestyle="dashdot")
    plt.axhline(y = 0.95,color='#929591',linewidth=1,linestyle="dashdot")

    for ii in np.arange(0.05, 1.01, 0.05):
        plt.axvline(x=ii,color='#929591',linewidth=1,alpha=0.3,linestyle="dashdot")
    for ii in np.arange(0.81, 1.002, 0.01):
        plt.axhline(y=ii,color='#929591',linewidth=1,alpha=0.3,linestyle="dashdot")
 
    plt.xlabel('Threshold',fontsize=32)
    plt.ylabel('Negative predictive value',fontsize=32)
    plt.xlim(0,1)
    plt.ylim(0.8,1)
    plt.xticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'],fontsize=24)
    plt.yticks([0.8,0.85,0.9,0.95,1],['0.8','0.85','0.9','0.95','1'],fontsize=24)
    plt.title('NPV',fontsize=32)
    plt.legend(loc=4,fontsize=18)

    plt.tight_layout()
    plt.savefig('Figure4b.pdf')

    ###### Figure 4c ######

    plt.figure(figsize=(10.24,8.96))

    for i in range(len(Method)):

        if i == 0:
            plt.plot(Data[i]['fpr'], Data[i]['tpr'], color='red', linewidth=3, label=str(Method[i])+', AUC-ROC: {:.2f}'.format(Data[i]['aucroc']))
        elif i == 3:
            plt.plot(Data[i]['fpr'], Data[i]['tpr'], color='C0', linewidth=3, label=str(Method[i])+', AUC-ROC: {:.2f}'.format(Data[i]['aucroc']))
        else:
            plt.plot(Data[i]['fpr'], Data[i]['tpr'], color='C'+str(i), linewidth=3, label=str(Method[i])+', AUC-ROC: {:.2f}'.format(Data[i]['aucroc']))

    plt.plot([0,1], [0,1], color='black', linewidth=3)

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

    plt.xlabel('False positive rate (1-Specificity)',fontsize=32)
    plt.ylabel('True positive rate (Sensitivity)',fontsize=32)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'],fontsize=24)
    plt.yticks([0,0.25,0.5,0.75,1],['0','0.25','0.5','0.75','1'],fontsize=24)
    plt.title('ROC curve',fontsize=32)
    plt.legend(loc=4,fontsize=18)

    plt.tight_layout()
    plt.savefig('Figure4c.pdf')
