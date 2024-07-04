# End to End ML project 

# How to run?
### STEPS:
Clone the repository
```bash
https://github.com/shulaxshan/pet_adoption_status_prediction
```

### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n venv python=3.12.0 -y
```

```bash
conda activate venv
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```


## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/shulazshan/pet_adoption_status_prediction.mlflow

export MLFLOW_TRACKING_USERNAME=shulazshan

export MLFLOW_TRACKING_PASSWORD=

```

