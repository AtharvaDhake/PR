import numpy as np

# Sample sentence
sentence = "the cat sits on the mat"
words = sentence.split()

# Create vocabulary
vocab = list(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Create CBOW training data
X, y = [], []
window = 2

for i in range(window, len(words) - window):
    context = words[i-window:i] + words[i+1:i+window+1]
    X.append([word_to_idx[w] for w in context])
    y.append(word_to_idx[words[i]])

X = np.array(X)
y = np.array(y)

# Initialize weights
vocab_size = len(vocab)
embedding_size = 5

W1 = np.random.rand(vocab_size, embedding_size)
W2 = np.random.rand(embedding_size, vocab_size)

# Train CBOW model
lr = 0.01
epochs = 1000

for epoch in range(epochs):
    for context, target in zip(X, y):
        # Average context word embeddings
        h = np.mean(W1[context], axis=0)

        # Output probabilities
        out = h @ W2
        out = np.exp(out - np.max(out))
        probs = out / np.sum(out)

        # Calculate error
        error = probs.copy()
        error[target] -= 1

        # Update weights
        W2 -= lr * np.outer(h, error)

        for idx in context:
            W1[idx] -= lr * (error @ W2.T) / len(context)

# Predict target word
context_words = ["the", "cat", "on", "the"]
context_ids = [word_to_idx[w] for w in context_words]

h = np.mean(W1[context_ids], axis=0)
predicted = np.argmax(h @ W2)

print("Context words:", context_words)
print("Predicted word:", idx_to_word[predicted])
print("CBOW model implemented successfully!")
print("Chit 14 completed successfully!")
