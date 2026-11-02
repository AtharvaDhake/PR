import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset - use extracted brain tumour folder
train = tf.keras.utils.image_dataset_from_directory(
    r"c:\Users\athar\OneDrive\Desktop\PR\BrainTumour_extracted\Training",
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val = tf.keras.utils.image_dataset_from_directory(
    r"c:\Users\athar\OneDrive\Desktop\PR\BrainTumour_extracted\Training",
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test = tf.keras.utils.image_dataset_from_directory(
    r"c:\Users\athar\OneDrive\Desktop\PR\BrainTumour_extracted\Testing",
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

num_classes = len(train.class_names)
print("Tumor classes:", train.class_names)

# Build CNN
model = Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train, validation_data=val, epochs=5)

print("Test Accuracy:", model.evaluate(test)[1])

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit7_plot.png")
plt.close()
print("Chit 7 completed successfully!")
