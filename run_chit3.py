import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

# Load CIFAR-10 CSV dataset - use extracted path
data = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\CIFAR10_extracted\train.csv").dropna()

# Separate features and labels
X = data.drop("label", axis=1).values
y = data["label"].values.astype(int)

# Normalize pixel values
X = X / 255.0

# Reshape images into 32x32x3
X = X.reshape(-1, 32, 32, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build FNN model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("CIFAR-10 FNN Accuracy")
plt.legend()
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit3_plot.png")
plt.close()
print("Chit 3 completed successfully!")
