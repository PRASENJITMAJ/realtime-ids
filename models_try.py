# train_and_save_models.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_and_save_models(base_dir: str, data_path: str) -> pd.DataFrame:
    model_dir = os.path.join(base_dir, "trained_models")
    graph_dir = os.path.join(base_dir, "model_graphs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"🚀 Training {name}...")
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC AUC": roc_auc
        }

        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(graph_dir, f"{name}_confusion_matrix.png"))
        plt.close()

        # Save ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(os.path.join(graph_dir, f"{name}_roc_curve.png"))
        plt.close()

    # Save performance summary
    performance_df = pd.DataFrame(results).T.round(4)
    performance_df.to_csv(os.path.join(model_dir, "model_performance_summary.csv"))
    print(f"\nAll models and outputs saved under: {base_dir}")
    return performance_df

if __name__ == "__main__":
    
    base_directory = r'C:\Users\majum\OneDrive\Pictures\Realtime_IDS'
    processed_csv   = r'C:/Users/majum/OneDrive/Pictures/Realtime_IDS/NSL-KDD/NSL-KDD-phase1-processed.csv'

    summary = train_and_save_models(base_directory, processed_csv)
    print("\nModel Performance Summary:\n")
    print(summary)