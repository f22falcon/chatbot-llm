import numpy as np
from utils import generate_cbow_data
from tokenizer import tokenizer


class CBOW:
    def __init__(self, vocab_size, embedding_dim=100, window_size=2, learning_rate=0.01,seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = learning_rate
        np.random.seed(seed)

        # Initialize weight matrices
        # Input embedding matrix (vocab_size × embed_dim)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Output weight matrix (embed_dim × vocab_size)
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)

    def forward(self, context_indices):
        # Average embeddings of context words
        h = np.mean(self.W1[context_indices], axis=0)  # shape: (embedding_dim,)
        u = np.dot(h, self.W2)  # shape: (vocab_size,)
        y_pred = self.softmax(u)
        return h, y_pred

    def backward(self, context_indices, target_index, h, y_pred):
        # One-hot encoding for target
        y_true = np.zeros(self.vocab_size)
        y_true[target_index] = 1

        # Output layer gradient (pred - true)
        error = y_pred - y_true

        # Gradients for W2
        dW2 = np.outer(h, error)

        # Gradients for W1
        dh = np.dot(self.W2, error)
        dW1 = np.zeros_like(self.W1)

        for idx in context_indices:
            dW1[idx] += dh / len(context_indices)

        # Weight updates
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2

        # Loss (cross entropy)
        loss = -np.log(y_pred[target_index] + 1e-9)
        return loss

    def save_embeddings(self, filepath):
        np.save(filepath, self.W1)
        print(f"Embeddings saved to {filepath}")


def train_cbow(model, training_pairs, epochs=10):
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for context, target in training_pairs:
            h, y_pred = model.forward(context)
            loss = model.backward(context, target, h, y_pred)
            total_loss += loss

        avg_loss = total_loss / len(training_pairs)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.4f}")

    return losses

if __name__ == "__main__":
    Token_ids=tokenizer()
    data_set= generate_cbow_data(Token_ids, window_size=2)
    print("cbow model is training is going on......")
   
    model1 = CBOW(
        vocab_size=len(set(Token_ids)),
        embedding_dim=100,
        window_size=2,
        learning_rate=0.01
    )

    train_cbow(model1, data_set, epochs=10000)
    print(" training completed!!")
    


