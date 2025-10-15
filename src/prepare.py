import os, yaml, pandas as pd

p = yaml.safe_load(open("params.yaml"))
raw = p["data"]["raw_path"]
out = p["data"]["processed_path"]

df = pd.read_csv(raw)

# TODO: adjust columns for your CSV.
# For a safe baseline: keep numeric columns only and drop NaNs.
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
df_simple = df[num_cols].dropna()

os.makedirs(os.path.dirname(out), exist_ok=True)
df_simple.to_csv(out, index=False)
print(f"prepare: wrote {out} with shape {df_simple.shape}")
