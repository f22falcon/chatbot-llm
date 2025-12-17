import numpy as np
from tokenizer import clean_text, build_vocab



orpus="""In computer science, a bag of words (BoW) model is a simplifying representation used in natural language processing and information retrieval. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model is commonly used in methods of document classification where the frequency of each word is used as a feature for training a classifier."""


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

