import tensorflow as tf
import numpy as np

# Constant addition
a = tf.constant(15)
b = tf.constant(20)
print("Addition:", a + b)

# Input and output data
x = np.random.rand(100).astype(np.float32)
y = 0.2 * x + 0.2

# Weight and bias
W = tf.Variable(0.0)
b = tf.Variable(0.0)

# Optimizer
optimizer = tf.keras.optimizers.Adam(0.1)

# Training
for step in range(1000):
    with tf.GradientTape() as tape:
        y_pred = W * x + b
        loss = tf.reduce_mean((y_pred - y) ** 2)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

print("Final Weight:", W.numpy())
print("Final Bias:", b.numpy())
print("Final Loss:", loss.numpy())

import torch

# Create tensor
x = torch.tensor([[1, 2], [3, 4]])
print("Tensor:")
print(x)

print("Ones:")
print(torch.ones(2, 2))

print("Zeros:")
print(torch.zeros(2, 2))

print("Random:")
print(torch.rand(2, 2))

print("Shape:", x.shape)
print("Data type:", x.dtype)

print("Addition:")
print(x + x)

print("Multiplication:")
print(x * x)

print("Transpose:")
print(x.T)

print("NumPy array:")
print(x.numpy())

print("Chit 1 completed successfully!")
