import os
import json
from collections import Counter
import re


RAW_DATA_PATH = "data/raw/"
VOCAB_OUTPUT_PATH = "data/processed/vocab.json"


def clean_text(text):
    # Normalize text: lowercase + remove special characters
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[\d+\][^a-zA-Z0-9\s]", "", text)
    return text

def build_vocab(data):
    id_to_word = {}
    word_to_id = {}
    all_words = []

    for word in data.split():
       all_words.append(word)
       if word not in word_to_id:
        idx=len(word_to_id)
        word_to_id[word] = idx
        id_to_word[idx] = word
        
        
    return  word_to_id,id_to_word ,all_words

    
if __name__ == "__main__":
    
    for filename in os.listdir(RAW_DATA_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_DATA_PATH, filename), "r", encoding="utf-8") as f:
                text = f.read()
                text = clean_text(text)
                Word_to_ID,ID_to_word,all_word=build_vocab(text)
                

    word_freq = Counter(all_word)

    vocab = {
        "word_to_id":Word_to_ID,
        "id_to_word": ID_to_word,
        "word_freq": dict(word_freq)
    }

    os.makedirs("data/processed/", exist_ok=True)

    with open(VOCAB_OUTPUT_PATH, "w") as f:
        json.dump(vocab, f, indent=4)

    print(f"Vocabulary created! Total words: {len(vocab['word_to_id'])}")

    #"word_to_id": {word: i for i, (word, _) in enumerate(word_freq.items())},{i: word for i, (word, _) in enumerate(word_freq.items())},