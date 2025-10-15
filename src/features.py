import os, yaml, joblib, pandas as pd
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))
csv = p["data"]["processed_path"]
feat_path = p["data"]["features_path"]

df = pd.read_csv(csv)

# TODO: set your real target name. If "Price" exists use it; else fallback to last column.
target = "Price"
if target not in df.columns:
    target = df.columns[-1]  # last column fallback

X = df.drop(columns=[target], errors="ignore")
y = df[target] if target in df.columns else df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=p["split"]["test_size"], random_state=p["split"]["random_state"]
)

os.makedirs(os.path.dirname(feat_path), exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), feat_path)
print(f"features: saved {feat_path} with X_train={X_train.shape}, X_test={X_test.shape}")
