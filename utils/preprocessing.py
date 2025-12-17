import re


def clean_text(path1,path2):
    # Normalize text: lowercase + remove special characters
    with open (path1,'r',encoding= 'utf-8') as f:
        text=f.read()
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[\d+\][^a-zA-Z0-9\s]", "", text)
    f.close()
    with open (path2,'w',encoding='utf-8') as f:
        f.write(text)
    f.close()

if __name__== "__main__":
    clean_text("/media/tanmoy/B406FC5306FC1856/ProjectChatbot/data/raw/wiki_ai.txt","/media/tanmoy/B406FC5306FC1856/ProjectChatbot/data/processed/clean_data.txt")





