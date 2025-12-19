import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"

file_path = DATA / "clean_data.txt"
Vocab_path = DATA /"vocab.json"


def tokenizer( Vocab=Vocab_path,path=file_path):
    
    with open (path,'r',encoding="utf-8")as f:
        text=f.read()
    with open(Vocab,'r',encoding="utf-8")as f:
      vocab = json.load(f)
      word_to_Id = vocab["word_to_id"]
      Token_ids=[ word_to_Id[w] for w in text.lower().split()]
    return Token_ids

if __name__=="__main__":
   print(len(set(tokenizer())))