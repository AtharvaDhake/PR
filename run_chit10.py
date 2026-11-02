import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset - use extracted creditcard CSV
data = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\CreditCard_extracted\Credit Card Dataset\creditcard.csv").dropna()

# Separate features and labels
X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use only normal transactions for training
X_train = X_train[y_train.values == 0]

# Build Autoencoder
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(14, activation='relu')(input_layer)
encoded = Dense(7, activation='relu')(encoded)
decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)
model = Model(input_layer, decoded)

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=20, batch_size=64,
          validation_split=0.1, verbose=1)

# Calculate reconstruction error
reconstructed = model.predict(X_test)
error = np.mean((X_test - reconstructed) ** 2, axis=1)

# Set a strict threshold (99.9th percentile)
train_pred = model.predict(X_train)
train_error = np.mean((X_train - train_pred) ** 2, axis=1)

threshold = np.percentile(train_error, 99.9)

# Detect fraud
y_pred = (error > threshold).astype(int)

print("\n--- Detection Results ---")
print("Actual Fraud Cases in Test Set:", int(sum(y_test)))
print("Predicted Fraud Cases:", int(sum(y_pred)))

# Plot results with a logarithmic scale
plt.figure(figsize=(10, 6))
plt.hist(error[y_test == 0], bins=50, alpha=0.6, label="Normal", color="blue")
plt.hist(error[y_test == 1], bins=50, alpha=0.6, label="Fraud", color="red")
plt.yscale('log')
plt.xlabel("Reconstruction Error")
plt.ylabel("Count (Log Scale)")
plt.title("Credit Card Fraud Detection (Log Scale with 99.9% Threshold)")
plt.legend()
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit10_plot.png")
plt.close()
print("Chit 10 completed successfully!")
