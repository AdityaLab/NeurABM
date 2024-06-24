# Identifying Importation and Asymptomatic Spreaders of Multi-drug Resistant Organisms in Hospital Settings

This code showcases that the our NeurABM framework identifies not only importation cases but also nosocomial infection cases better than other machine learning or modeling-based baselines.

We demonstrate our NeurABM in identifying MRSA importation and nosocomial infection cases in the ICUs of the UVA hospital in 2019. We used EHR data from the UVA hospital to construct patient contact networks (used by the ABM) and collect patient risk factors (used by the neural network). We use the SIS-ABM model (Cui, Jiaming, et al. "Using spectral characterization to identify healthcare-associated infection (HAI) patients for clinical contact precaution." Scientific Reports 13.1 (2023): 16197.) as the ABM for disease transmission in NeurABM. Ground-truth MRSA infections are identified from lab test results for each patient in the EHR. 

For each week k, we used the contact networks, patient risk factors, and lab test results until week k-1 to train the NeurABM and identify importation cases before week k-1. We then ran the SIS-ABM model for 7 more days to infer the infection states of patients for week k, which correspond to nosocomial infections. Note that only data prior to week k are used in this process. 

## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. 

Then run the following commands:

```bash
conda env create --prefix ./envs/neurabm --file environment.yml
source activate ./envs/neurabm
```

## Directory structure

```
- data
       - 2019.pkl -> EHR data as pkl file
- run.sh -> shell file to run the NeurABM
- main.py -> code to train NeurABM
- framework.py -> NeurABM framework implementation
- model.py -> SIS-ABM model code
- environment.yml -> environment file
- output -> save results
```

## Dataset

The dataset is at `data` folder. It contains the synthetic EHR data used for NeurABM. 

## Demo

We provde a demo code to run the NeurABM
Run:

```
chmod 777 run.sh
./run.sh
```
This will save the results in output folder
