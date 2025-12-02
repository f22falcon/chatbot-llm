import os
import json
from collections import Counter
import re



RAW_DATA_PATH = "data/raw/"
VOCAB_OUTPUT_PATH = "data/processed/vocab.json"


def clean_text(text):
    # Normalize text: lowercase + remove special characters
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[\d+\][^a-zA-Z0-9\s]", "", text)
    return text

def build_vocab():
    all_words = []

    for filename in os.listdir(RAW_DATA_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_DATA_PATH, filename), "r", encoding="utf-8") as f:
                text = f.read()
                text = clean_text(text)
                words = text.split()
                all_words.extend(words)

    word_freq = Counter(all_words)

    vocab = {
        "word_to_id": {word: i for i, (word, _) in enumerate(word_freq.items())},
        "id_to_word": {i: word for i, (word, _) in enumerate(word_freq.items())},
        "word_freq": dict(word_freq)
    }

    os.makedirs("data/processed/", exist_ok=True)

    with open(VOCAB_OUTPUT_PATH, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"Vocabulary created! Total words: {len(vocab['word_to_id'])}")

if __name__ == "__main__":
    build_vocab()