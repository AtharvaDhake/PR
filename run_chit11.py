import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Load dataset - use extracted MNIST CSV
data = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\MNIST_extracted\MNIST\mnist_784_csv.csv").dropna()

# Separate features and labels
X = data.drop("class", axis=1).values

# Normalize images
X = X / 255.0

# Reshape images
X = X.reshape(-1, 28, 28, 1)

# Split dataset
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

# Add noise
noise = np.random.normal(0, 0.3, X_train.shape)
X_train_noisy = np.clip(X_train + noise, 0, 1)

noise = np.random.normal(0, 0.3, X_test.shape)
X_test_noisy = np.clip(X_test + noise, 0, 1)

# Build Autoencoder
inp = Input(shape=(28, 28, 1))

x = Flatten()(inp)

encoded = Dense(128, activation="relu")(x)
encoded = Dense(64, activation="relu")(encoded)

decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(decoded)

decoded = Reshape((28, 28, 1))(decoded)

autoencoder = Model(inp, decoded)

# Compile model
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train model
autoencoder.fit(
    X_train_noisy, X_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test_noisy, X_test)
)

# Reconstruct images
denoised = autoencoder.predict(X_test_noisy[:10])

# Display results
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test_noisy[i].squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 10, i + 11)
    plt.imshow(denoised[i].squeeze(), cmap="gray")
    plt.axis("off")

plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit11_plot.png")
plt.close()

print("Image denoising using Autoencoder completed successfully!")
print("Chit 11 completed successfully!")
