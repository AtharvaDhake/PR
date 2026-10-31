# 📘 Deep Learning Lab Practice IV — Code Explanation Guide
**Department of Information Technology | SE (IT) — Semester IV**

> This file explains **what each line of code does** in all 14 chit notebooks. Use this to explain the code to your examiner step by step.

---

## 📌 CHIT 1 — TensorFlow & PyTorch Basics
**File:** `chit1.ipynb` | **Assignment:** Demonstrate use of TensorFlow and PyTorch

### Part A: TensorFlow — Constant Operations & Linear Regression

```python
import tensorflow as tf
import numpy as np
```
- Import **TensorFlow** (deep learning library by Google) and **NumPy** (for numeric arrays).

```python
a = tf.constant(15)
b = tf.constant(20)
print("Addition:", a + b)
```
- `tf.constant()` creates an **immutable tensor** (like a fixed number in TF).
- TensorFlow overloads `+`, so `a + b` does tensor addition → prints `35`.

```python
x = np.random.rand(100).astype(np.float32)
y = 0.2 * x + 0.2
```
- Create 100 random input values `x` and compute output `y = 0.2x + 0.2`.
- This is the **true relationship** the model must discover.

```python
W = tf.Variable(0.0)
b = tf.Variable(0.0)
```
- `tf.Variable` → **learnable parameters** (unlike constants, these change during training).
- `W` = weight, `b` = bias. Both start at 0.

```python
optimizer = tf.keras.optimizers.Adam(0.1)
```
- **Adam optimizer** — an advanced version of gradient descent. Learning rate = 0.1.

```python
for step in range(1000):
    with tf.GradientTape() as tape:
        y_pred = W * x + b
        loss = tf.reduce_mean((y_pred - y) ** 2)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
```
- **`tf.GradientTape()`** records all operations inside it so gradients can be computed.
- `y_pred = W*x + b` → linear prediction.
- `loss = MSE` — mean squared error between prediction and actual `y`.
- `tape.gradient(loss, [W, b])` → computes dLoss/dW and dLoss/db.
- `apply_gradients()` → updates W and b using Adam rule.
- After 1000 steps, W ≈ 0.2 and b ≈ 0.2 (the true values!).

### Part B: PyTorch — Tensor Operations

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
```
- Create a 2×2 **tensor** (multi-dimensional array) from a Python list.

```python
torch.ones(2, 2)    # Matrix of ones
torch.zeros(2, 2)   # Matrix of zeros
torch.rand(2, 2)    # Random values between 0 and 1
```
- Utility functions to create commonly used tensors.

```python
print("Shape:", x.shape)      # (2, 2)
print("Data type:", x.dtype)  # torch.int64
```
- `.shape` → dimensions of the tensor. `.dtype` → data type (int, float, etc.).

```python
x + x        # Element-wise addition
x * x        # Element-wise multiplication
x.T          # Transpose (rows become columns)
x.numpy()    # Convert tensor to NumPy array
```
- PyTorch supports all standard mathematical operations on tensors directly.

---

## 📌 CHIT 2 — Feedforward Neural Network on MNIST
**File:** `chit2.ipynb` | **Assignment:** Implement FNN using Keras on MNIST Digit Dataset

```python
train = tf.keras.utils.image_dataset_from_directory(
    os.path.expanduser("~/Downloads/MNIST/train"),
    image_size=(28, 28), color_mode="grayscale", batch_size=128
)
```
- Load MNIST images from local folders. MNIST digits are 28×28 grayscale images.
- `color_mode="grayscale"` → 1 channel (no colour).
- `batch_size=128` → process 128 images at a time during training.

```python
train = train.map(lambda x, y: (x / 255.0, y))
```
- **Normalize** pixel values from [0, 255] to [0, 1]. This helps the model converge faster.

```python
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])
```
- **`Flatten`** → reshape 28×28 image into a flat 784-element vector (28×28=784).
- **`Dense(128, relu)`** → fully connected hidden layer with 128 neurons. ReLU removes negative values.
- **`Dense(10, softmax)`** → output layer. 10 neurons = 10 digit classes (0-9). Softmax converts scores to probabilities that sum to 1.

```python
model.compile(optimizer="sgd",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```
- **SGD** = Stochastic Gradient Descent — basic optimizer.
- **`sparse_categorical_crossentropy`** → loss for multi-class classification where labels are integers (0,1,2...).
- **`accuracy`** → metric to track during training.

```python
history = model.fit(train, epochs=10, validation_data=test)
```
- Train the model for 10 epochs. `history` saves accuracy/loss per epoch for plotting.

---

## 📌 CHIT 3 — Feedforward Neural Network on CIFAR-10
**File:** `chit3.ipynb` | **Assignment:** Implement FNN using Keras on CIFAR Dataset

```python
train = tf.keras.utils.image_dataset_from_directory(
    os.path.expanduser("~/Downloads/CIFAR10/train"),
    image_size=(32, 32), batch_size=128
)
```
- Load CIFAR-10 images. These are 32×32 **colour** (RGB) images in 10 categories (airplane, car, bird, cat, etc.).

```python
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
])
```
- `Flatten(32, 32, 3)` → 32×32×3 = **3072 input features** (because 3 colour channels R, G, B).
- Larger hidden layer (256) needed because CIFAR-10 is more complex than MNIST.
- Same softmax output for 10 classes.

> **Key Difference from CHIT 2:** CIFAR-10 uses colour (3 channels, 3072 features) vs MNIST grayscale (1 channel, 784 features).

---

## 📌 CHIT 4 — CNN on Fashion MNIST
**File:** `chit4.ipynb` | **Assignment:** Build image classification model using CNN on Fashion MNIST

```python
train = pd.read_csv("fashion-mnist_train.csv").dropna()
X_train = train.drop(['label'], axis=1).values
y_train = train['label'].values
X_train = X_train.reshape(-1, 28, 28, 1)
```
- Load Fashion-MNIST from CSV (785 columns: 1 label + 784 pixel values).
- `reshape(-1, 28, 28, 1)` → convert flat 784 pixels back to 28×28×1 image shape for CNN.

```python
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
```
- **`Rescaling(1./255)`** → normalize pixel values to [0,1].
- **`Conv2D(32, (3,3), relu)`** → 32 filters, each 3×3, slide over the image detecting features (edges, textures). Output: 26×26×32 feature maps.
- **`MaxPooling2D((2,2))`** → halve the size (26×26 → 13×13). Keeps only strongest activations. Reduces computation and overfitting.
- **Second Conv2D(64)** → 64 deeper filters detecting more complex patterns.
- **`Flatten()`** → convert 3D feature maps to 1D vector for Dense layers.
- **`Dense(128, relu)`** → learn from extracted features.
- **`Dense(10, softmax)`** → 10 fashion categories (T-shirt, Trouser, Dress, Sneaker, etc.).

---

## 📌 CHIT 5 — CNN on Pneumonia X-Ray Dataset
**File:** `chit5.ipynb` | **Assignment:** Build CNN model on Pneumonia X-Ray Dataset

```python
train = tf.keras.utils.image_dataset_from_directory(
    os.path.expanduser("~/Downloads/Pneumonia X Ray/train"),
    color_mode="grayscale", image_size=(128, 128),
    validation_split=0.2, subset="training", seed=42
)
```
- Load chest X-Ray images resized to 128×128 grayscale.
- `validation_split=0.2` → use 80% for training, 20% for validation automatically.
- Directory must have sub-folders named `NORMAL/` and `PNEUMONIA/`.

```python
Dense(2, activation="softmax")
```
- **Binary classification** (Normal vs Pneumonia) → output layer has only 2 neurons.

> Same CNN architecture as Chit 4 but larger images (128×128) and only 2 output classes.

---

## 📌 CHIT 6 — CNN on Food Dataset
**File:** `chit6.ipynb` | **Assignment:** Build image classification model using CNN on Food Dataset

```python
num_classes = len(train.class_names)
Dense(num_classes, activation="softmax")
```
- `class_names` is automatically detected from sub-folder names in the food directory.
- Output layer size is **dynamic** — depends on how many food categories exist.

> Same CNN structure but for colour food images (128×128×3). Multi-class output.

---

## 📌 CHIT 7 — CNN on Brain Tumor Dataset
**File:** `chit7.ipynb` | **Assignment:** Build Brain Tumor Classification model with CNN

```python
train = tf.keras.utils.image_dataset_from_directory(
    os.path.expanduser("~/Downloads/BrainTumor/train"),
    image_size=(128, 128), batch_size=32, ...
)
num_classes = len(train.class_names)
```
- Load MRI brain scan images. Classes could be: `glioma`, `meningioma`, `notumor`, `pituitary`.
- Colour MRI images at 128×128. Output layer adapts to number of tumor classes.

---

## 📌 CHIT 8 — Recurrent Neural Network using NumPy (from scratch)
**File:** `chit8.ipynb` | **Assignment:** Build Recurrent Neural Network using the NumPy library

```python
data = np.sin(np.arange(0, 100, 0.1))
```
- Create a synthetic **sine wave** time series dataset (1000 points).

```python
for i in range(len(data) - 10):
    X.append(data[i:i+10])   # 10 past values as input
    y.append(data[i+10])      # 11th value is target to predict
```
- **Sliding window** approach: use 10 past timesteps to predict the next value.

```python
Wxh = np.random.randn(input_size, hidden_size) * 0.1   # Input to Hidden weights
Whh = np.random.randn(hidden_size, hidden_size) * 0.1  # Hidden to Hidden weights
Why = np.random.randn(hidden_size, 1) * 0.1            # Hidden to Output weights
bh = np.zeros((1, hidden_size))   # Hidden bias
by = np.zeros((1, 1))             # Output bias
```
- Initialize all RNN weight matrices manually using NumPy.

```python
def forward(X):
    h = np.zeros((len(X), hidden_size))
    states = [h]
    for t in range(X.shape[1]):
        h = np.tanh(X[:, t, :] @ Wxh + h @ Whh + bh)
        states.append(h)
    output = h @ Why + by
    return output, states
```
- **RNN Forward Pass:** At each timestep `t`:
  - `h = tanh(x_t × Wxh + h_prev × Whh + bh)` — hidden state update using BOTH current input AND previous hidden state.
  - This is what makes it "recurrent" — it has **memory** of past inputs.
- Final hidden state `h` → output prediction `y`.

```python
dz = dh * (1 - h**2)     # Derivative of tanh activation
np.clip(g, -1, 1, out=g) # Gradient Clipping — prevent exploding gradients
Wxh -= lr * dWxh          # Update weights with Batch Gradient Descent
```
- **BPTT (Backpropagation Through Time):** Calculate gradients through each timestep backwards.
- Gradient clipping prevents the "exploding gradient" problem in RNNs.

---

## 📌 CHIT 9 — Sparse Autoencoder on MNIST
**File:** `chit9.ipynb` | **Assignment:** Autoencoder to reconstruct MNIST digits with sparsity constraint

```python
data = pd.read_csv("mnist_784_csv.csv").dropna()
X = data.drop("class", axis=1).values
X = X / 255.0
```
- Load MNIST from local CSV (784 pixel columns + 1 class column).
- Normalize pixels to [0,1]. Drop the label since autoencoder is **unsupervised**.

```python
encoded = Dense(128, activation="relu",
                activity_regularizer=l1(0.00001))(inp)
```
- **Encoder:** Compress 784 inputs → 128 hidden units.
- **`activity_regularizer=l1()`** → L1 Sparsity Constraint: penalizes large neuron activations, forcing most neurons to be near 0. Only a few "important" neurons stay active → **sparse representation**.

```python
decoded = Dense(784, activation="sigmoid")(encoded)
```
- **Decoder:** Reconstruct 128 compressed features back to 784 pixels.
- `sigmoid` → output between 0 and 1 (matches normalized pixel range).

```python
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(X, X, ...)   # Input = Output (self-supervised learning)
```
- Autoencoder is trained to reconstruct its own input.
- `binary_crossentropy` → appropriate for values in [0,1] range.

---

## 📌 CHIT 10 — Autoencoder for Anomaly Detection (Credit Card Fraud)
**File:** `chit10.ipynb` | **Assignment:** Use Autoencoder for anomaly detection on credit card dataset

```python
data = pd.read_csv("creditcard.csv").dropna()
X_train = X_train[y_train == 0]   # ONLY normal transactions for training
```
- **Core idea:** Train autoencoder ONLY on normal (Class=0) transactions.
- Normal transactions → low reconstruction error. Fraud → HIGH error (never seen before).

```python
inp = Input(shape=(X_train.shape[1],))
encoded = Dense(14, activation="relu")(inp)
encoded = Dense(7, activation="relu")(encoded)
decoded = Dense(14, activation="relu")(encoded)
decoded = Dense(X_train.shape[1], activation="sigmoid")(decoded)
```
- **Bottleneck Autoencoder:** 30 → 14 → 7 → 14 → 30.
- The bottleneck (7 neurons) forces the model to learn a compressed representation of normal transactions.

```python
error = np.mean((X_test - reconstructed) ** 2, axis=1)
threshold = np.percentile(train_error, 95)
y_pred = (error > threshold).astype(int)
```
- **Reconstruction Error** = MSE between original and reconstructed transaction.
- **Threshold** set at 95th percentile of normal error → anything above = anomaly/FRAUD.
- Plot histogram showing clear separation between normal and fraud error distributions.

---

## 📌 CHIT 11 — Image Denoising Autoencoder on MNIST
**File:** `chit11.ipynb` | **Assignment:** Implement image denoising using autoencoders on MNIST

```python
noise = np.random.normal(0, 0.3, X.shape)
X_noisy = np.clip(X + noise, 0, 1)
```
- Add **Gaussian noise** (mean=0, std=0.3) to images.
- `np.clip(0, 1)` → keep pixel values in valid [0,1] range.

```python
autoencoder.fit(X_noisy, X_clean, ...)
# Input: noisy image, Target: CLEAN image
```
- **Key Difference from CHIT 9:** Input ≠ Target. Input is noisy, target is clean.
- Autoencoder learns to **remove noise** and reconstruct clean images.

---

## 📌 CHIT 12 — Object Detection using Transfer Learning (Faster R-CNN)
**File:** `chit12.ipynb` | **Assignment:** Implement object detection using Transfer Learning on Food Dataset

```python
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
```
- Load **Faster R-CNN** pretrained on COCO dataset (91 object categories including food items).
- `model.eval()` → inference mode (no gradient tracking, no dropout).

```python
with torch.no_grad():
    prediction = model([image_tensor])[0]
```
- `torch.no_grad()` → do not compute gradients (faster, less memory, inference only).
- `prediction` contains `boxes`, `labels`, `scores` for all detected objects.

```python
threshold = 0.5
keep = prediction["scores"] >= threshold
boxes = prediction["boxes"][keep].cpu()
labels = prediction["labels"][keep].cpu()
```
- Only keep detections with confidence ≥ 50%.
- `.cpu()` → move tensor from GPU back to CPU for processing.

```python
annotated_image = draw_bounding_boxes(
    image_for_display, boxes=boxes, labels=box_labels, colors="red", width=3
)
```
- Draw red bounding boxes with category name and confidence score on each detected object.

---

## 📌 CHIT 13 — Image Classification using Transfer Learning (Animal Dataset)
**File:** `chit13.ipynb` | **Assignment:** Implement image classification using Transfer Learning on Animal Dataset

```python
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False
```
- Load **MobileNetV2** pretrained on ImageNet (1.4 million images, 1000 classes).
- `include_top=False` → remove the original 1000-class classifier head.
- `trainable=False` → **freeze** all layers. Their pretrained weights do NOT change.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),  # MobileNetV2 expects [-1, 1] range
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),            # Compress feature maps to vector
    tf.keras.layers.Dense(len(class_names), activation="softmax")  # New classification head
])
```
- `Rescaling(1/127.5, offset=-1)` → scales pixels from [0,255] to [-1,1] (MobileNetV2 specific requirement).
- `GlobalAveragePooling2D` → average each feature map across all spatial positions → compact vector.
- New Dense layer → only this layer is trained, using features from frozen MobileNetV2.

```python
history = model.fit(train, validation_data=test, epochs=5)
```
- Only 5 epochs needed because most of the model is already pre-trained on ImageNet features!

---

## 📌 CHIT 14 — Continuous Bag of Words (CBOW)
**File:** `chit14.ipynb` | **Assignment:** Implement the Continuous Bag of Words (CBOW) Model

```python
sentence = "the cat sits on the mat"
words = sentence.split()
vocab = list(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
```
- Build **vocabulary**: unique word → integer index mapping.

```python
window = 2
for i in range(window, len(words) - window):
    context = words[i-window:i] + words[i+1:i+window+1]  # 2 words before + 2 words after
    X.append([word_to_idx[w] for w in context])
    y.append(word_to_idx[words[i]])   # CENTER word is the target to predict
```
- **CBOW Training Pairs:** Context window of 2 on each side. Predict center word from 4 surrounding words.
- Example: `["the", "cat", "on", "the"]` → predicts `"sits"`.

```python
W1 = np.random.rand(vocab_size, embedding_size)  # Word embedding matrix (vocab × 5)
W2 = np.random.rand(embedding_size, vocab_size)  # Output projection (5 × vocab)
```
- `W1`: Each row is the **word vector/embedding** (5 dimensions for this example).
- `W2`: Projects from embedding space back to vocabulary size for prediction.

```python
h = np.mean(W1[context], axis=0)   # Average of context word embeddings = "Bag of Words"
```
- **"Bag of Words"** means word ORDER doesn't matter — we just average the context embeddings.

```python
out = h @ W2
out = np.exp(out - np.max(out))
probs = out / np.sum(out)           # Softmax: convert to probabilities
```
- Multiply hidden state by output matrix → apply Softmax → probability of each word being the target.

```python
error = probs.copy()
error[target] -= 1                  # Cross-entropy loss gradient (true label probability -1)
W2 -= lr * np.outer(h, error)
W1[idx] -= lr * (error @ W2.T) / len(context)
```
- **Manual Backpropagation:** Update `W2` and `W1` using gradient of cross-entropy loss.
- After training, `W1` rows contain meaningful **word vectors** (similar words → similar vectors).

---

*Generated for Deep Learning Lab Practice IV — Examination Preparation*

