import numpy as np
from tokenizer import tokenizer
from utils.Postprocessing import negative_sampling, generate_skipgram_pairs


class SkipGramNS:
    def __init__(self, vocab_size, embedding_dim=100, lr=0.01, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr

        # Input (target) embeddings
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        # Output (context) embeddings
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_step(self, target_id, context_id, neg_ids):
        """
        target_id  : int
        context_id : int
        neg_ids    : list[int] (k negatives)
        """

        v_t = self.W_in[target_id]        # (D,)
        v_c = self.W_out[context_id]      # (D,)
        v_neg = self.W_out[neg_ids]       # (k, D)

        # -------- forward --------
        score_pos = np.dot(v_t, v_c)      # scalar
        score_neg = np.dot(v_neg, v_t)    # (k,)

        p_pos = self.sigmoid(score_pos)
        p_neg = self.sigmoid(-score_neg)

        # -------- loss --------
        loss = -np.log(p_pos + 1e-9) - np.sum(np.log(p_neg + 1e-9))

        # -------- backward --------
        grad_pos = p_pos - 1              # scalar
        grad_neg = (1 - p_neg)            # (k,)

        # gradients
        grad_v_t = grad_pos * v_c + np.sum(grad_neg[:, None] * v_neg, axis=0)
        grad_v_c = grad_pos * v_t
        grad_v_neg = grad_neg[:, None] * v_t

        # updates
        self.W_in[target_id] -= self.lr * grad_v_t
        self.W_out[context_id] -= self.lr * grad_v_c
        self.W_out[neg_ids] -= self.lr * grad_v_neg

        return loss
    
    def save_embeddings(self, filepath):
        np.save(filepath, (self.W_in+self)/2)
        print(f"Embeddings saved to {filepath}")


def train_skipgram(model, pairs, vocab_size, epochs=5, k=5):
     for epoch in range(epochs):
        total_loss = 0
        for target, context in pairs:
            neg_ids = negative_sampling(vocab_size, context, k)
            loss = model.train_step(target, context, neg_ids)
            total_loss += loss

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")


if __name__ =="__main__":  
   Token_ids=tokenizer()
   data_set=generate_skipgram_pairs(Token_ids, window_size=2)
   Model=SkipGramNS(len(set(Token_ids)),embedding_dim=100,lr=0.01)
   print("SkipGram model training is going on......")
   train_skipgram(Model, data_set, vocab_size=len(set(Token_ids)), epochs=500, k=5)
   print("Training completed!!")