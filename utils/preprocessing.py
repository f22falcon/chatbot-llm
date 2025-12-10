def generate_cbow_data(token_ids, window_size):
    pairs = []

    for i in range(window_size, len(token_ids) - window_size):
        context = token_ids[i - window_size : i] + token_ids[i + 1 : i + window_size + 1]
        target = token_ids[i]
        pairs.append((context, target))

    return pairs
