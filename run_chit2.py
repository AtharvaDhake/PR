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

# Load MNIST CSV dataset - use extracted path
data = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\MNIST_extracted\MNIST\mnist_784_csv.csv").dropna()

# Separate features and labels
X = data.drop("class", axis=1).values
y = data["class"].values.astype(int)

# Normalize pixel values
X = X / 255.0

# Reshape images into 28x28
X = X.reshape(-1, 28, 28)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build FNN model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# Compile model using 'adam' for faster convergence
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
plt.title("MNIST FNN Accuracy")
plt.legend()
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit2_plot.png")
plt.close()
print("Chit 2 completed successfully!")
