import os
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset - use extracted paths
train = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\FashionMNIST_extracted\Fashion MNIST\fashion-mnist_train.csv").dropna()
test = pd.read_csv(r"c:\Users\athar\OneDrive\Desktop\PR\FashionMNIST_extracted\Fashion MNIST\fashion-mnist_test.csv").dropna()

X_train = train.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
y_train = train["label"].values

X_test = test.drop("label", axis=1).values.reshape(-1, 28, 28, 1)
y_test = test["label"].values

# Build CNN model
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Fashion MNIST CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit4_plot.png")
plt.close()
print("Chit 4 completed successfully!")
