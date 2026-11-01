from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
BASE=Path(__file__).resolve().parent
model=joblib.load(BASE/"models/iris_model.joblib")
meta=json.loads((BASE/"models/model_metadata.json").read_text(encoding="utf-8"))
features=meta["feature_names"]; classes=meta["target_names"]
app=Flask(__name__)
@app.get("/")
def home():
    return jsonify({"message":"Iris Species Prediction API","features":features,"usage":"POST JSON to /predict"})
@app.get("/health")
def health():
    return jsonify({"status":"ok","model_loaded":model is not None})
@app.post("/predict")
def predict():
    data=request.get_json(silent=True)
    if not isinstance(data,dict): return jsonify({"error":"Body must be a JSON object"}),400
    missing=[f for f in features if f not in data]
    if missing: return jsonify({"error":"Missing required feature(s)","missing":missing}),400
    try: values={f:float(data[f]) for f in features}
    except (TypeError,ValueError): return jsonify({"error":"All feature values must be numeric"}),400
    if any(v<0 for v in values.values()): return jsonify({"error":"Feature values must be non-negative"}),400
    frame=pd.DataFrame([values],columns=features)
    idx=int(model.predict(frame)[0]); probs=model.predict_proba(frame)[0]
    return jsonify({"predicted_class":classes[idx],"predicted_class_id":idx,
      "probabilities":{classes[i]:round(float(probs[i]),4) for i in range(len(classes))}})
if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)
