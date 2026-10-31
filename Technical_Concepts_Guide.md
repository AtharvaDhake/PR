# 🧠 Deep Learning Lab Practice IV — Technical Concepts Guide
**Department of Information Technology | SE (IT) — Semester IV**

> This file explains all the **technical concepts** used across all 14 chit notebooks. Use this to answer the examiner's theoretical questions confidently.

---

## ⚡ SECTION 1: Deep Learning Frameworks

### 1.1 TensorFlow (Chit 1)
- **What is it?** Open-source machine learning framework developed by Google.
- **Key Feature:** Supports automatic differentiation via `tf.GradientTape`.
- **Two types of objects:**
  - `tf.constant` → immutable, fixed value, cannot be changed
  - `tf.Variable` → mutable, changes during training (weights, biases)
- **GradientTape:** Records operations to compute gradients automatically.
  ```
  with tf.GradientTape() as tape:
      loss = compute_loss()
  gradients = tape.gradient(loss, variables)
  ```

### 1.2 PyTorch (Chit 1)
- **What is it?** Open-source deep learning framework by Facebook (Meta).
- **Tensor:** Multi-dimensional array, like NumPy but runs on GPU.
- **Key Tensor Operations:**
  | Operation | Method |
  |-----------|--------|
  | Create from list | `torch.tensor(data)` |
  | All ones | `torch.ones(shape)` |
  | All zeros | `torch.zeros(shape)` |
  | Random | `torch.rand(shape)` |
  | Transpose | `tensor.T` |
  | To NumPy | `tensor.numpy()` |
  | To GPU/CPU | `tensor.to(device)` |

### 1.3 Keras (Chits 2–7, 9–11, 13)
- **What is it?** High-level API for building neural networks, runs on top of TensorFlow.
- **Why use Keras?** Simple, readable code with `model.fit()`, `model.compile()`, `model.evaluate()`.

---

## ⚡ SECTION 2: Neural Network Fundamentals

### 2.1 Neuron / Perceptron
- **Formula:** `output = activation(weights × input + bias)`
- **Bias:** Extra parameter that shifts the activation function. Allows the model to fit data that doesn't pass through origin.

### 2.2 Activation Functions

| Activation | Formula | Used In | Why |
|------------|---------|---------|-----|
| **ReLU** | `max(0, x)` | Hidden layers | Simple, fast, solves vanishing gradient |
| **Sigmoid** | `1/(1+e^-x)` | Binary output / Autoencoder output | Output in [0,1] |
| **Softmax** | `e^xi / Σe^xj` | Multi-class output | Converts scores to probabilities summing to 1 |
| **Tanh** | `(e^x - e^-x)/(e^x + e^-x)` | RNN hidden states | Output in [-1,1], zero-centered |

### 2.3 Loss Functions

| Loss | Used For | Chits |
|------|----------|-------|
| **Mean Squared Error (MSE)** | Regression, Reconstruction | Chit 1, 8, 10 |
| **Binary Cross-Entropy** | Binary classification, Autoencoder | Chit 9, 11 |
| **Sparse Categorical Cross-Entropy** | Multi-class (integer labels) | Chit 2–7, 13 |
| **Categorical Cross-Entropy** | Multi-class (one-hot labels) | Chit 14 |

### 2.4 Optimizers

| Optimizer | Full Name | Used In | Notes |
|-----------|-----------|---------|-------|
| **SGD** | Stochastic Gradient Descent | Chit 2, 3 | Simple, may oscillate |
| **Adam** | Adaptive Moment Estimation | Chit 1, 4–11, 13, 14 | Combines momentum + RMSprop. Most popular |
| **Batch GD** | Batch Gradient Descent | Chit 8 | Uses ALL training data per update |

**Adam Formula:**
```
m = β₁·m + (1-β₁)·g         # Momentum (1st moment)
v = β₂·v + (1-β₂)·g²        # Velocity / RMS (2nd moment)
w = w - α · m / (√v + ε)    # Parameter update
```

### 2.5 Training Process
1. **Forward Pass** → compute prediction
2. **Compute Loss** → measure error
3. **Backward Pass** → compute gradients via backpropagation
4. **Update Weights** → optimizer adjusts parameters
5. Repeat for all epochs

### 2.6 Key Training Terms

| Term | Meaning |
|------|---------|
| **Epoch** | One full pass through the entire training dataset |
| **Batch Size** | Number of samples processed before updating weights |
| **Learning Rate** | Step size for weight update. Too high → overshoot. Too low → slow convergence |
| **Overfitting** | Model memorizes training data, performs poorly on new data |
| **Underfitting** | Model is too simple to capture patterns |
| **Validation Split** | Portion of training data held out to monitor overfitting |

---

## ⚡ SECTION 3: Feedforward Neural Networks (FNN)
**Used in:** Chit 2, Chit 3

### Architecture
```
Input Layer (784 or 3072) → Hidden Layer(s) → Output Layer (10)
```

### How it Works
1. Input image is **flattened** from 2D (28×28) to 1D (784 pixels).
2. Each neuron in one layer is **fully connected** to every neuron in the next.
3. Data flows in **one direction** (forward only) — no loops.

### MNIST vs CIFAR-10 Comparison

| Feature | MNIST (Chit 2) | CIFAR-10 (Chit 3) |
|---------|---------------|------------------|
| Image Size | 28×28 | 32×32 |
| Channels | 1 (grayscale) | 3 (RGB colour) |
| Input Features | 784 | 3072 |
| Classes | 10 (digits 0-9) | 10 (airplane, car, etc.) |
| Dataset Size | 70,000 images | 60,000 images |

---

## ⚡ SECTION 4: Convolutional Neural Networks (CNN)
**Used in:** Chit 4 (Fashion MNIST), Chit 5 (Pneumonia), Chit 6 (Food), Chit 7 (Brain Tumor)

### Why CNN over FNN for Images?
- FNN ignores **spatial structure** — treats each pixel independently.
- CNN preserves **local patterns** — nearby pixels (edges, textures) are processed together.
- CNN is **translation invariant** — detects features anywhere in the image.

### CNN Architecture Layers

#### 1. Conv2D Layer
```python
Conv2D(32, (3, 3), activation="relu")
```
- `32` = number of **filters** (also called kernels or feature detectors)
- `(3,3)` = filter size (3×3 patch of pixels at a time)
- Each filter **slides** (convolves) across the entire image
- Detects one type of feature: edges, curves, textures, shapes
- Output size formula: `(Input_W - Filter_W + 1) × (Input_H - Filter_H + 1) × Num_Filters`
- Example: 28×28 image with 3×3 filter → **26×26×32** output

#### 2. MaxPooling2D Layer
```python
MaxPooling2D((2, 2))
```
- Takes the **maximum value** from each 2×2 window
- **Halves** the spatial dimensions: 26×26 → 13×13
- **Purpose:** Reduce computation, retain important features, reduce overfitting
- Makes CNN robust to small translations/shifts in the image

#### 3. Flatten Layer
```python
Flatten()
```
- Converts 3D feature maps (height × width × channels) into a 1D vector for Dense layers.

#### 4. Dense Layers (Fully Connected)
```python
Dense(128, activation="relu")    # Hidden layer — learn combinations of features
Dense(10, activation="softmax")  # Output layer — classify into N categories
```

### CNN Flow Summary
```
Input Image → Conv2D (feature extraction) → MaxPool (downsampling)
            → Conv2D (deeper features) → MaxPool
            → Flatten → Dense → Dense → Output Probabilities
```

### Rescaling Layer
```python
tf.keras.layers.Rescaling(1./255)
```
- Divides all pixel values by 255 to get values in [0, 1].
- **Why normalize?** Prevents large input values from dominating gradients, speeds up training.

---

## ⚡ SECTION 5: Recurrent Neural Networks (RNN)
**Used in:** Chit 8

### Why RNN for Sequential Data?
- FNN and CNN don't have **memory** — each input is processed independently.
- RNN maintains a **hidden state** — a form of memory that captures information from past timesteps.
- Ideal for: Time series, Speech, Text, Sequences.

### Vanilla RNN Equations

```
h_t = tanh(x_t · Wxh + h_{t-1} · Whh + bh)    # Hidden state update
y_t = h_t · Why + by                              # Output
```

**Where:**
- `x_t` = current input at timestep t
- `h_t` = current hidden state (memory)
- `h_{t-1}` = previous hidden state
- `Wxh, Whh, Why` = weight matrices
- `bh, by` = bias vectors

### Weight Matrices in RNN

| Matrix | Shape | Role |
|--------|-------|------|
| `Wxh` | [input_size × hidden_size] | Input → Hidden |
| `Whh` | [hidden_size × hidden_size] | Hidden → Hidden (recurrence) |
| `Why` | [hidden_size × output_size] | Hidden → Output |

### Sine Wave Prediction Task
- Generate 1000-point sine wave.
- **Sliding Window (size=10):** Use 10 past values to predict the 11th.
- Train RNN to learn the sine wave pattern.
- After training, RNN can predict future values accurately.

### Problems with RNN
- **Vanishing Gradient:** Gradients shrink exponentially through many timesteps → model forgets long-term dependencies.
- **Exploding Gradient:** Gradients grow exponentially → weights blow up.
- **Solutions:** Gradient Clipping (used in Chit 8), LSTM, GRU.

### Gradient Clipping (Chit 8)
```python
np.clip(g, -1, 1, out=g)
```
- If gradient value is outside [-1, 1], clip it to the boundary.
- Prevents exploding gradients from causing NaN weights.

### Backpropagation Through Time (BPTT)
- Like regular backpropagation but **unrolled through timesteps**.
- Gradients flow backwards through each timestep.
- Derivative of tanh: `d/dx tanh(x) = 1 - tanh²(x)`
- In code: `dz = dh * (1 - h**2)`

---

## ⚡ SECTION 6: Autoencoders
**Used in:** Chit 9 (Sparse Reconstruction), Chit 10 (Anomaly Detection), Chit 11 (Denoising)

### What is an Autoencoder?
An Autoencoder is a **self-supervised neural network** that learns to:
1. **Encode** input into a compressed lower-dimensional representation (latent space)
2. **Decode** the compressed representation back to the original input

```
Input (784) → Encoder → Bottleneck (32) → Decoder → Reconstructed Output (784)
```

### Architecture
```python
# Encoder
inp = Input(shape=(784,))
latent = Dense(128, activation="relu")(inp)   # Compress
latent = Dense(32, activation="relu")(latent)  # Bottleneck

# Decoder
x = Dense(128, activation="relu")(latent)      # Expand
out = Dense(784, activation="sigmoid")(x)      # Reconstruct

# Model: input → output
autoencoder = Model(inp, out)
autoencoder.fit(X, X, ...)    # Target = Input (self-supervised)
```

### Three Types of Autoencoders Used

#### Sparse Autoencoder (Chit 9)
- Adds **L1 Regularization** on hidden layer activations.
- Forces most neurons to be inactive (near 0) → **sparse** representation.
- Why? Forces the network to learn only the most important features.
- `activity_regularizer=l1(1e-5)` → adds `λ × Σ|activation|` to the loss.

#### Anomaly Detection Autoencoder (Chit 10)
- Trained ONLY on **normal** data (non-fraud transactions).
- Normal data → reconstructed well (low error).
- **Anomaly/Fraud** → model has never seen it → reconstructed poorly (HIGH error).
- Detection rule: `Reconstruction Error > Threshold → ANOMALY`.
- **Threshold** = 95th percentile of normal transaction errors.

#### Denoising Autoencoder (Chit 11)
- Input = **noisy** image (clean + Gaussian noise).
- Target = **clean** image.
- Model learns to separate signal (image) from noise.
- Training: `autoencoder.fit(X_noisy, X_clean)` — Input ≠ Target.

### Reconstruction Error
```python
error = np.mean((X_test - reconstructed) ** 2, axis=1)  # MSE per sample
```
- Small error → model recognized this sample (normal).
- Large error → model doesn't know how to reconstruct it (anomaly).

---

## ⚡ SECTION 7: Transfer Learning
**Used in:** Chit 12 (Object Detection — Faster R-CNN), Chit 13 (Animal Classification — MobileNetV2)

### What is Transfer Learning?
- Use a model **pre-trained on a large dataset** (ImageNet, COCO) as a starting point.
- The pre-trained model has already learned low-level features: edges, textures, shapes.
- Fine-tune only the last few layers for your specific task.

### Why Transfer Learning?
- **Less data needed** — pretrained features generalize well.
- **Faster training** — only train a few layers, not from scratch.
- **Better accuracy** — leverages knowledge from millions of images.

### Two Pretrained Models Used

#### MobileNetV2 (Chit 13 — Image Classification)
- **Trained on:** ImageNet (1.4 million images, 1000 classes)
- **Architecture:** Lightweight, mobile-friendly CNN
- **Special requirement:** Input pixels must be in **[-1, 1]** range (not [0,1]).
- `Rescaling(1/127.5, offset=-1)` → converts [0,255] to [-1,1].
- **Frozen layers:** All 154 original layers frozen → only new Dense head is trained.
- `GlobalAveragePooling2D()` → average each feature map → compact feature vector.

#### Faster R-CNN with ResNet50 FPN (Chit 12 — Object Detection)
- **Trained on:** COCO dataset (91 object categories)
- **Full pipeline:**
  1. **Region Proposal Network (RPN)** → suggests "regions of interest" likely containing objects.
  2. **Feature extraction (ResNet50)** → extract features from each proposed region.
  3. **Classification head** → assign class label to each region.
  4. **Regression head** → refine bounding box coordinates.
- **Output per image:** `boxes`, `labels`, `scores` for all detected objects.
- `model.eval()` → inference mode: no gradient computation, dropout disabled.
- `torch.no_grad()` → prevents gradient tracking during inference (faster, saves memory).

### Frozen vs Unfrozen Layers
```python
base_model.trainable = False    # FROZEN — weights do not change during training
base_model.trainable = True     # UNFROZEN — weights update during training (fine-tuning)
```

---

## ⚡ SECTION 8: Natural Language Processing

### Word Embeddings (Chit 14)
- **Problem:** Words are text — computers work with numbers. How to represent words?
- **Solution:** Map each word to a **dense vector** in N-dimensional space.
- Similar words → similar vectors (close in vector space).
- Example (5 dimensions): `"king" → [0.2, 0.8, -0.4, 0.1, 0.6]`

### Continuous Bag of Words (CBOW) Model (Chit 14)
- **Goal:** Learn word embeddings by training a model to predict a **center word** from its **surrounding context words**.
- **"Bag of Words"** → word order doesn't matter, just the presence of context words.

### CBOW Training Process
```
Context: ["the", "cat", "on", "the"]  →  Target: "sits"
```

**Step-by-step:**
1. Look up embedding vectors for each context word from `W1` (embedding matrix).
2. **Average** all context vectors → hidden state `h`.
3. Multiply `h × W2` → output scores for all vocabulary words.
4. Apply **Softmax** → probabilities.
5. Compute **Cross-Entropy Loss** comparing probabilities to true target word.
6. Backpropagate → update `W1` and `W2`.

**After training:**
- `W1` matrix rows = **learned word embeddings**.
- Cosine similarity between embeddings → measure word similarity.

### One-Hot Encoding (needed for CBOW)
- Word at index 3 → `[0, 0, 0, 1, 0, 0, ...]` (vocab-size vector).
- But embeddings (dense vectors) are much more compact and meaningful.

---

## ⚡ SECTION 9: Regularization Techniques

### Dropout (Chit 4)
```python
layers.Dropout(0.5)  # Randomly set 50% of neurons to 0 during training
```
- During training: randomly "drops" 50% of neurons in each forward pass.
- During inference: all neurons active but outputs scaled down.
- **Why?** Prevents overfitting — model cannot rely on any single neuron.

### L1 Regularization / Sparsity Constraint (Chit 9)
```python
activity_regularizer=l1(0.00001)
```
- Adds `λ × Σ|activation|` to the loss function.
- Encourages activations to be exactly 0 (sparse).
- Model learns to use only essential features.

### L2 Regularization (Weight Decay)
- Adds `λ × Σ(weight²)` to loss.
- Encourages weights to be small, prevents overfitting.
- Not used directly in these chits but conceptually important.

---

## ⚡ SECTION 10: Data Loading & Preprocessing

### Loading Images from Directory (Keras)
```python
tf.keras.utils.image_dataset_from_directory(
    "path/to/dataset",
    image_size=(128, 128),    # Resize all images
    batch_size=32,            # Batch size
    color_mode="grayscale",   # 1 channel OR "rgb" for 3 channels
    validation_split=0.2,     # 20% for validation
    subset="training",        # "training" or "validation"
    seed=42                   # Reproducibility
)
```
- **Requirement:** Folder must have **sub-folders for each class**.
  ```
  dataset/
    ├── class_A/  (image files)
    ├── class_B/  (image files)
    └── class_C/  (image files)
  ```
- **`class_names`** property → automatically reads sub-folder names as class labels.

### Loading Tabular/CSV Data (Pandas)
```python
data = pd.read_csv("creditcard.csv").dropna()
X = data.drop("Class", axis=1).values   # Features
y = data["Class"].values                 # Labels
```

### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- `test_size=0.2` → 80% train, 20% test.
- `stratify=y` → maintain same class proportions in both splits (important for imbalanced data).

### Normalization
| Method | Formula | Used For |
|--------|---------|----------|
| **Min-Max Scaling** | `(x - min) / (max - min)` → [0,1] | Credit card features (MinMaxScaler) |
| **Divide by 255** | `x / 255.0` → [0,1] | Image pixels |
| **Divide by 127.5 - 1** | `x/127.5 - 1` → [-1,1] | MobileNetV2 inputs |

---

## ⚡ SECTION 11: Object Detection Concepts (Chit 12)

### What is Object Detection?
- Not just WHAT is in the image (classification), but **WHERE** (localization).
- Output: **Bounding boxes** (x1, y1, x2, y2) + **Class labels** + **Confidence scores**.

### Faster R-CNN Pipeline
```
Image → Feature Extraction (ResNet50) → Region Proposal Network
      → ROI Pooling → Classification + Bounding Box Regression
      → Final Detections (filtered by confidence threshold)
```

### Confidence Threshold
```python
threshold = 0.5
keep = prediction["scores"] >= threshold
```
- Only show detections where the model is at least **50% confident**.
- Too low → many false positives. Too high → miss real objects.

### COCO Dataset
- 330,000 images with 91 object categories.
- Common categories: person, car, bicycle, airplane, cat, dog, pizza, donut, etc.
- The Faster R-CNN model uses COCO category indices (1-90) for labels.

---

## ⚡ SECTION 12: Quick Summary Table

| Chit | Topic | Model Type | Dataset | Key Technique |
|------|-------|-----------|---------|--------------|
| 1 | TF & PyTorch Basics | Linear Regression | Synthetic | GradientTape, Tensors |
| 2 | FNN on MNIST | Feedforward NN | MNIST digits | Flatten, Dense, SGD |
| 3 | FNN on CIFAR | Feedforward NN | CIFAR-10 | 3072 input features |
| 4 | CNN on Fashion MNIST | CNN | Fashion MNIST CSV | Conv2D, MaxPool |
| 5 | CNN on Pneumonia | CNN | Chest X-Ray | Binary CNN, 128×128 |
| 6 | CNN on Food | CNN | Food images | Dynamic num_classes |
| 7 | CNN on Brain Tumor | CNN | MRI scans | Multi-class CNN |
| 8 | RNN from Scratch | Vanilla RNN (NumPy) | Sine wave | BPTT, Gradient Clipping |
| 9 | Sparse Autoencoder | Autoencoder | MNIST CSV | L1 Sparsity, Reconstruction |
| 10 | Anomaly Detection | Bottleneck Autoencoder | Credit Card CSV | Reconstruction Error Threshold |
| 11 | Image Denoising | Denoising Autoencoder | MNIST CSV + Noise | Noisy→Clean mapping |
| 12 | Object Detection | Faster R-CNN (Transfer) | Local image | Bounding boxes, COCO |
| 13 | Animal Classification | MobileNetV2 (Transfer) | Animal images | Frozen layers, GAP |
| 14 | Word Embeddings | CBOW (NumPy) | Text sentence | Word vectors, Softmax |

---

## ⚡ SECTION 13: Likely Viva / Examiner Questions & Answers

**Q: What is the difference between tf.constant and tf.Variable?**
> `tf.constant` is immutable (fixed value). `tf.Variable` is mutable and can be updated during training. Weights and biases are always Variables.

**Q: What is the purpose of Flatten in a CNN?**
> `Flatten` converts the 3D feature map (height × width × channels) output from Conv/Pool layers into a 1D vector so it can be fed into Dense (Fully Connected) layers.

**Q: Why is MaxPooling used in CNN?**
> MaxPooling reduces spatial dimensions (downsampling) while retaining the most important features (maximum values). This reduces computation, memory, and overfitting, and makes the model more robust to small shifts in the input.

**Q: What is the difference between FNN and CNN?**
> FNN flattens the image and treats each pixel independently — it loses spatial information. CNN preserves spatial structure using filters that detect local patterns. CNN is much better for images.

**Q: What is Backpropagation Through Time (BPTT)?**
> BPTT is the algorithm to train RNNs. It unrolls the RNN through all timesteps and applies backpropagation from the final timestep back to the first, computing gradients for each timestep's weights.

**Q: Why do autoencoders use the same data as both input and output?**
> Because the goal is to learn a compressed representation (encoding) that can faithfully reconstruct the original data. By training input=target, the model is forced to learn the most important features needed for reconstruction.

**Q: What is the bottleneck in an Autoencoder?**
> The bottleneck is the smallest layer in the encoder (e.g., 7 or 32 neurons). It forces the model to compress information into a compact representation. Data must be encoded efficiently to pass through the bottleneck.

**Q: How does an Autoencoder detect anomalies (Chit 10)?**
> It's trained only on normal data. Normal samples → low reconstruction error. Anomalies (fraud) were never seen in training → model cannot reconstruct them well → HIGH reconstruction error. A threshold separates normal from anomaly.

**Q: What is Transfer Learning and why is it useful?**
> Transfer Learning reuses a model pre-trained on a large dataset (ImageNet) for a new task. The model has already learned general features (edges, shapes). We only train new task-specific layers. Benefits: less data needed, faster training, better accuracy.

**Q: What is the difference between `model.eval()` and `torch.no_grad()`?**
> `model.eval()` changes model behavior (disables Dropout, BatchNorm uses running stats). `torch.no_grad()` stops gradient computation to save memory and speed up inference. Both are used together for inference.

**Q: What is CBOW and how is it different from Skip-gram?**
> CBOW predicts a **center word** from its surrounding **context words** (average of context embeddings). Skip-gram is the opposite — predicts surrounding context words from a single center word. CBOW is faster; Skip-gram works better for rare words.

**Q: What is Gradient Clipping and why is it used in RNNs?**
> Gradient clipping limits the magnitude of gradients to a maximum value (e.g., 1.0). It prevents "exploding gradients" — when gradients grow exponentially during BPTT in long sequences, causing weight updates to be extremely large (NaN/overflow).

---

*Generated for Deep Learning Lab Practice IV — Examination Preparation*
*All 14 assignments covered | Semester IV | Department of Information Technology*

