import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === PATHS ===
csv_path = r"C:\Users\majum\OneDrive\Pictures\Realtime_IDS\NSL-KDD\NSL-KDD-phase1-processed.csv"
save_dir = r"C:\Users\majum\OneDrive\Pictures\Realtime_IDS\data\processed"
os.makedirs(save_dir, exist_ok=True)

# === LOAD CSV ===
df = pd.read_csv(csv_path)
X = df.drop(columns=["label"]).values
y = df["label"].values

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === SCALING ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === SAVE .NPY FILES ===
np.save(os.path.join(save_dir, "X_train.npy"), X_train_scaled)
np.save(os.path.join(save_dir, "X_test.npy"), X_test_scaled)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)

# === SAVE CSV FILES ===
train_df = pd.DataFrame(X_train_scaled)
train_df["label"] = y_train
train_df.to_csv(os.path.join(save_dir, "train_processed.csv"), index=False)

test_df = pd.DataFrame(X_test_scaled)
test_df["label"] = y_test
test_df.to_csv(os.path.join(save_dir, "test_processed.csv"), index=False)

print("✅ Saved all .npy and .csv files to:", save_dir)
