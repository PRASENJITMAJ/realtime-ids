import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Path to your final cleaned dataset
df = pd.read_csv(r"C:/Users/majum/OneDrive/Pictures/Realtime_IDS/NSL-KDD/NSL-KDD-phase1-processed.csv")

# Extract features
X = df.drop(columns=["label"])

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
joblib.dump(scaler, r"C:/Users/majum/OneDrive/Pictures/Realtime_IDS/trained_models/scaler.pkl")
print("✅ Scaler saved successfully!")
