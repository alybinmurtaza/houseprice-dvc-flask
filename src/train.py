import os, yaml, json, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

p = yaml.safe_load(open("params.yaml"))
X_train, X_test, y_train, y_test = joblib.load(p["data"]["features_path"])

m = RandomForestRegressor(
    n_estimators=p["model"]["n_estimators"],
    max_depth=p["model"]["max_depth"],
    random_state=p["model"]["random_state"],
)

m.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)
joblib.dump(m, "models/model.pkl")

pred = m.predict(X_test)
mae = float(mean_absolute_error(y_test, pred))
os.makedirs("metrics", exist_ok=True)
json.dump({"mae": mae}, open("metrics/train_metrics.json","w"))

print(f"train: model.pkl saved, MAE={mae:.4f}")
