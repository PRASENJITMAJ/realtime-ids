
import os
import pandas as pd
import numpy as np
import arff # type: ignore
from sklearn.preprocessing import StandardScaler

# 📘 2) Define column names
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

# 📘 3) Define functions to load data
def load_txt(path):
    return pd.read_csv(path, names=col_names, index_col=False)

def load_arff(path):
    data = arff.load(open(path, 'r'))
    df = pd.DataFrame(data['data'], columns=[c[0] for c in data['attributes']])
    if "class" in df.columns:
        df.rename(columns={"class": "label"}, inplace=True)
    return df[col_names]

# 📘 4) Set NSL-KDD folder path here (update this accordingly)
base_path = "C:\Users\majum\OneDrive\Pictures\Realtime_IDS\NSL-KDD"  # update this
train_txt = os.path.join(base_path, "KDDTrain+.txt")
train_arff = os.path.join(base_path, "KDDTrain+.arff")
test_txt = os.path.join(base_path, "KDDTest+.txt")
test_arff = os.path.join(base_path, "KDDTest+.arff")

# 📘 5) Load data
dfs = []
for txt, arf in [(train_txt, train_arff), (test_txt, test_arff)]:
    if os.path.isfile(txt):
        dfs.append(load_txt(txt))
    elif os.path.isfile(arf):
        dfs.append(load_arff(arf))
    else:
        raise FileNotFoundError(f"File not found: {txt} or {arf}")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded NSL-KDD dataset with shape: {df.shape}")

# 📘 6) Binary label mapping
df["label"] = df["label"].apply(lambda x: 0 if "normal" in x else 1)

# 📘 7) Drop duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 📘 8) One-hot encode categorical columns
df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"], drop_first=True)

numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col != "label"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


print("🧼 Missing values:", df.isna().sum().sum())
print("🔁 Duplicate rows:", df.duplicated().sum())
print("🔎 Label counts:\n", df["label"].value_counts())


out_path = os.path.join(base_path, "NSL-KDD-phase1-processed.csv")
df.to_csv(out_path, index=False)
print(f"📁 Saved cleaned dataset to: {out_path}")