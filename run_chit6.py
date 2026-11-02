import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Use PR/Food or ~/Downloads/Food
food_dir = r"c:\Users\athar\OneDrive\Desktop\PR\Food"
if not os.path.exists(food_dir):
    food_dir = os.path.expanduser("~/Downloads/Food")

# Load dataset
train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(food_dir, "train"),
    image_size=(128, 128),
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=42
)

val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(food_dir, "train"),
    image_size=(128, 128),
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test = tf.keras.utils.image_dataset_from_directory(
    os.path.join(food_dir, "test"),
    image_size=(128, 128),
    batch_size=8,
    shuffle=False
)

num_classes = len(train.class_names)
print("Food classes:", train.class_names)

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

# Compile and train
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train, validation_data=val, epochs=5)

# Evaluate
eval_res = model.evaluate(test)
print("Test Accuracy:", eval_res[1])

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.title("Food CNN Accuracy")
plt.savefig(r"c:\Users\athar\OneDrive\Desktop\PR\chit6_plot.png")
plt.close()

print("Chit 6 completed successfully!")
