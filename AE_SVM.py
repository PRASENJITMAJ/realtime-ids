import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def train_autoencoder_svm(base_dir: str, data_path: str) -> pd.DataFrame:
    ae_dir = os.path.join(base_dir, "AE_SVM")
    model_dir = os.path.join(ae_dir, "trained_models")
    graph_dir = os.path.join(ae_dir, "model_graphs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # AutoEncoder Architecture
    input_dim = X_train_scaled.shape[1]
    encoding_dim = input_dim // 2

    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(inputs)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)

    autoencoder = tf.keras.Model(inputs, decoded)
    encoder = tf.keras.Model(inputs, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    # Train AutoEncoder
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2
    )

    autoencoder.save(os.path.join(model_dir, "autoencoder_model.keras"))
    encoder.save(os.path.join(model_dir, "encoder_model.keras"))

    # Plot AE loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("AutoEncoder Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(graph_dir, "AE_loss_curve.png"))
    plt.close()

    # Encode features
    X_train_encoded = encoder.predict(X_train_scaled)
    X_test_encoded = encoder.predict(X_test_scaled)

    # Train SVM on encoded features
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_encoded, y_train)
    joblib.dump(svm, os.path.join(model_dir, "svm_on_encoded.pkl"))

    y_pred = svm.predict(X_test_encoded)
    y_proba = svm.predict_proba(X_test_encoded)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics_df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'ROC AUC': [roc_auc]
    }, index=['AE+SVM'])

    metrics_df.to_csv(os.path.join(model_dir, "AE_SVM_performance_summary.csv"))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("AE+SVM - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(graph_dir, "AE_SVM_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AE+SVM - ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(graph_dir, "AE_SVM_roc_curve.png"))
    plt.close()

    print(f"\n✅ AE+SVM model and results saved in: {ae_dir}")
    return metrics_df


if __name__ == "__main__":
    base_dir = r'C:\Users\majum\OneDrive\Pictures\Realtime_IDS'
    data_path = r'C:/Users/majum/OneDrive/Pictures/Realtime_IDS/NSL-KDD/NSL-KDD-phase1-processed.csv'
    result = train_autoencoder_svm(base_dir, data_path)
    print("\n📊 AE+SVM Results:\n", result)