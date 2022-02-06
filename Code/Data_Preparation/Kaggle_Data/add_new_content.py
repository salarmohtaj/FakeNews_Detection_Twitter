import pandas as pd
import os
import re
import pickle
from sklearn.model_selection import KFold

data_directory = "../../../Data/Kaggle_Data"
data_name = "Fake_True.tsv"

# df = pd.read_excel(os.path.join(data_directory, data_name)
#                    , engine='openpyxl')
df = pd.read_csv(os.path.join(data_directory, data_name), sep="\t")
data_name = "DIC_URL.DIC"
with open(os.path.join(data_directory, data_name), "rb") as f:
    dic_complete = pickle.load(f)


def extract_urls(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    urls = [link[0] for link in links]
    return urls

def replace_url(text, rep=""):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    text = re.sub(link_regex, rep, text)
    return text


def replace_url_with_text(text):
    urls = extract_urls(text)
    for url in urls:
        try:
            t = dic_complete[url]["text"]
            t = replace_url(t, rep="")
            t = preprocess(t)
        except Exception as s:
            t = ""
        text = text.replace(url, t)
    return text


def preprocess(text):
    try:
        text = text.replace("\t"," ")
    except:
        print(text)
    text = text.replace("\n", " ")
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    return text


df["url_remove"] = df.apply(lambda x: replace_url(x['text'], rep=""), axis=1)
df["url_replace_constant"] = df.apply(lambda x: replace_url(x['text'], rep="weblink"), axis=1)
df["url_replace_text"] = df.apply(lambda x: replace_url_with_text(x['text']), axis=1)
df["text"] = df.apply(lambda x: preprocess(x['text']), axis=1)


df = df.dropna()
df = df.sample(frac=0.5)


data_directory = "../../../Data/Kaggle_Data/final_data"

print(df["label"].value_counts())


data_name = "final_data.tsv"
df.to_csv(os.path.join(data_directory,data_name), sep="\t", index=False)

kf5 = KFold(n_splits=5, shuffle=True)
j = 1
for train_index, test_index in kf5.split(df):
    try:
        os.makedirs(os.path.join(data_directory,str(j)))
    except OSError as e:
        pass
    Train = df.iloc[train_index]
    Test = df.iloc[test_index]
    Train.to_csv(os.path.join(data_directory,str(j),"train.tsv"), sep="\t", index=False)
    Test.to_csv(os.path.join(data_directory, str(j), "test.tsv"), sep="\t", index=False)
    j += 1
