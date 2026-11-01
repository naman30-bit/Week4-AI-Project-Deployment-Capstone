from pathlib import Path
import json, joblib, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
B=Path(__file__).resolve().parent
for d in ["data","models","outputs"]: (B/d).mkdir(exist_ok=True)
iris=load_iris(as_frame=True); df=iris.frame.copy()
df["species"]=df["target"].map(dict(enumerate(iris.target_names))); df.to_csv(B/"data/iris_dataset.csv",index=False)
X=df[iris.feature_names]; y=df["target"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Pipeline([("scaler",StandardScaler()),("classifier",RandomForestClassifier(n_estimators=200,random_state=42))])
model.fit(Xtr,ytr); p=model.predict(Xte)
metrics={"accuracy":accuracy_score(yte,p),"precision_macro":precision_score(yte,p,average="macro"),"recall_macro":recall_score(yte,p,average="macro"),"f1_macro":f1_score(yte,p,average="macro")}
joblib.dump(model,B/"models/iris_model.joblib")
(B/"models/model_metadata.json").write_text(json.dumps({"feature_names":iris.feature_names,"target_names":list(iris.target_names),"test_metrics":metrics},indent=2))
(B/"outputs/metrics.json").write_text(json.dumps(metrics,indent=2))
pd.DataFrame(confusion_matrix(yte,p),index=iris.target_names,columns=iris.target_names).to_csv(B/"outputs/confusion_matrix.csv")
(B/"outputs/classification_report.txt").write_text(classification_report(yte,p,target_names=iris.target_names))
print("Training and testing completed. Model saved to models/iris_model.joblib")
print(json.dumps(metrics,indent=2))
print("Start API with: python app.py")
