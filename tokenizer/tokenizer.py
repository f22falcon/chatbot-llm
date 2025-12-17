import json

def tokenizer( file_name="clean_data.txt",path="/media/tanmoy/B406FC5306FC1856/ProjectChatbot/data"):
    with open (path+"/processed/"+file_name,'r',encoding="utf-8")as f:
        text=f.read()
    with open(path+"/processed/vocab.json",'r',encoding="utf-8")as f:
      vocab = json.load(f)
      word_to_Id = vocab["word_to_id"]
      Token_ids=[ word_to_Id[w] for w in text.lower().split()]
    return Token_ids

if __name__=="__main__":
   print(len(set(tokenizer())))