import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

# Load dataset - use extracted MNIST CSV
data = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\MNIST_extracted\MNIST\mnist_784_csv.csv").dropna()
X = data.drop("class", axis=1).values

# Normalize
X = X / 255.0

# Build Autoencoder
inp = Input(shape=(784,))

# Encoder with sparsity constraint
encoded = Dense(
    128,
    activation="relu",
    activity_regularizer=l1(0.00001)
)(inp)

# Decoder
decoded = Dense(784, activation="sigmoid")(encoded)

# Create model
autoencoder = Model(inp, decoded)

# Compile
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train
autoencoder.fit(
    X, X,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

# Reconstruct images
reconstructed = autoencoder.predict(X[:10])

# Display images
plt.figure(figsize=(12, 4))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit9_plot.png")
plt.close()
print("Sparse Autoencoder implemented successfully!")
print("Chit 9 completed successfully!")
