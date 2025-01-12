# Identifying and Forecasting Importation and Asymptomatic Spreaders of Multi-drug Resistant Organisms in Hospital Settings

This code showcases that the our NeurABM framework identifies not only importation cases but also forecasts nosocomial infection cases better than other machine learning or modeling-based baselines.

We demonstrate our NeurABM in identifying MRSA importation and nosocomial infection cases in the ICUs of the UVA hospital in 2019. We used EHR data from the UVA hospital to construct patient contact networks (used by the ABM) and collect patient risk factors (used by the neural network). We use the SIS-ABM model (Cui, Jiaming, et al. "Using spectral characterization to identify healthcare-associated infection (HAI) patients for clinical contact precaution." Scientific Reports 13.1 (2023): 16197.) as the ABM for disease transmission in NeurABM. Ground-truth MRSA infections are identified from lab test results for each patient in the EHR. 

The outputs of our model are available in this repo. The electronic health record (EHR) data used in developing the models is not available since it is highly sensitive, and we do not have permission to release it.

## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. 

Then run the following commands:

```bash
conda env create --prefix ./envs/neurabm --file environment.yml
source activate ./envs/neurabm
```

## Directory structure

```
-Figure2 -> This folder allows you to reproduce Figure 2 in the main article.
       - data -> Experiment results to reproduce Figure 2.
	- Figure2.py -> Running this code directly will reproduce Figure 2.
-Figure3 -> This folder allows you to reproduce Figure 3 in the main article.
       - data -> Experiment results to reproduce Figure 3.
	- Figure3.py -> Running this code directly will reproduce Figure 3.
-Figure4 -> This folder allows you to reproduce Figure 4 in the main article.
       - data -> Experiment results to reproduce Figure 4.
	- Figure4.py -> Running this code directly will reproduce Figure 4.
-Figure5 -> This folder allows you to reproduce Figure 5 in the main article.
       - data -> Experiment results to reproduce Figure 5.
       - Figure5.py -> Running this code directly will reproduce Figure 5.
- data
       - synthetic.pkl -> EHR data as pkl file
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

We provde a demo code and a synthetic dataset to run the NeurABM. The demo code usually takes 1-2 hours to run,

```
./run.sh
```
This will save the results in output folder.
