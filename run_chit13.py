import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load dataset - use extracted cats_and_dogs_filtered
train = tf.keras.utils.image_dataset_from_directory(
    r"c:\Users\athar\OneDrive\Desktop\PR\CatsDogs_extracted\cats_and_dogs_filtered\train",
    image_size=(160, 160),
    batch_size=32,
)

val = tf.keras.utils.image_dataset_from_directory(
    r"c:\Users\athar\OneDrive\Desktop\PR\CatsDogs_extracted\cats_and_dogs_filtered\validation",
    image_size=(160, 160),
    batch_size=32,
)

print(train.class_names)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

base_model.trainable = False

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train,
    validation_data=val,
    epochs=5
)

# Evaluate model
loss, accuracy = model.evaluate(val)
print("Accuracy:", accuracy * 100, "%")

# Plot accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit13_plot.png")
plt.close()

print("Chit 13 completed successfully!")
