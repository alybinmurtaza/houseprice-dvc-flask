import json, yaml, joblib
from sklearn.metrics import r2_score

p = yaml.safe_load(open("params.yaml"))
X_train, X_test, y_train, y_test = joblib.load(p["data"]["features_path"])
m = joblib.load("models/model.pkl")
r2 = float(r2_score(y_test, m.predict(X_test)))
json.dump({"r2": r2}, open("metrics/test_metrics.json","w"))
print(f"evaluate: R2={r2:.4f}")
