import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


DATA_DIR    = os.path.join("data", "processed")  
FEAT_DIR    = os.path.join("data", "features")     
os.makedirs(FEAT_DIR, exist_ok=True)


X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"Loaded:  X_train {X_train.shape},  X_test {X_test.shape}")


pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

print(f"PCA: {X_train.shape[1]}  →  {X_train_pca.shape[1]} components "
      f"(95 % variance)")

# Save PCA features
np.save(os.path.join(FEAT_DIR, "X_train_pca.npy"), X_train_pca)
np.save(os.path.join(FEAT_DIR, "X_test_pca.npy"),  X_test_pca)


input_dim     = X_train.shape[1]
encoding_dim  = input_dim // 2           

# AE architecture
inp = Input(shape=(input_dim,))
enc = Dense(encoding_dim, activation="relu")(inp)
dec = Dense(input_dim,    activation="linear")(enc)     

autoencoder = Model(inp, dec)
encoder     = Model(inp, enc)     

autoencoder.compile(optimizer="adam", loss="mse")

# Train/validation split for AE training
X_tr, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

print("Training AutoEncoder …")
autoencoder.fit(
    X_tr, X_tr,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[EarlyStopping(monitor="val_loss",
                             patience=5,
                             restore_best_weights=True)],
    verbose=2
)

# Extract bottleneck features
X_train_ae = encoder.predict(X_train, verbose=0)
X_test_ae  = encoder.predict(X_test,  verbose=0)

print(f"AE bottleneck dimension: {X_train_ae.shape[1]}")

# Save AE features
np.save(os.path.join(FEAT_DIR, "X_train_ae.npy"), X_train_ae)
np.save(os.path.join(FEAT_DIR, "X_test_ae.npy"),  X_test_ae)


np.save(os.path.join(FEAT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(FEAT_DIR, "y_test.npy"),  y_test)

print("Phase 2 complete ✔  PCA & AE features saved to data/features/")

