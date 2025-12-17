import numpy as np

def generate_cbow_data(token_ids, window_size):
    pairs = []

    for i in range(window_size, len(token_ids) - window_size):
        context = token_ids[i - window_size : i] + token_ids[i + 1 : i + window_size + 1]
        target = token_ids[i]
        pairs.append((context, target))

    return pairs

def generate_skipgram_pairs(token_ids, window_size=2):
    pairs = []
    for i, target_id in enumerate(token_ids):
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context_id = token_ids[j]
                pairs.append((target_id, context_id))

    return pairs

def negative_sampling(vocab_size, positive_id, k,seed=42):
    negs = set()
    np.random.seed(seed)  # For reproducibility
    while len(negs) < k:
        n = np.random.randint(0, vocab_size - 1)
        if n != positive_id:
            negs.add(n)
    return list(negs)
