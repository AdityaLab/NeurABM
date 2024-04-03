# Combining Agent-based Models and Neural Networks to Identify Importation and Nosocomial Infection Cases of Healthcare Associated Infections

## Setup

First install Anaconda. The dependencies are listed in `environment.yml` file. 

Then run the following commands:

```bash
conda env create --prefix ./envs/neurabm --file environment.yml
source activate ./envs/neurabm
```

## Directory structure

```
-data
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
