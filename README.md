# Week 4 – AI Project Deployment & Capstone
Iris Species Prediction API using Python, Scikit-learn, Joblib and Flask.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
## Train and evaluate
```powershell
python train_model.py
```
The full fitted pipeline is saved in `models/iris_model.joblib`.

## Start the API
```powershell
python app.py
```


## Run API tests (in a second terminal)
```powershell
python -m unittest test_app.py
```

## Example prediction (PowerShell)
```powershell
$body = @{
  "sepal length (cm)" = 5.1
  "sepal width (cm)" = 3.5
  "petal length (cm)" = 1.4
  "petal width (cm)" = 0.2
} | ConvertTo-Json

```
The prediction response contains the species and probabilities. This is an educational example using the public Iris dataset. Load Joblib/Pickle files only from trusted sources.
