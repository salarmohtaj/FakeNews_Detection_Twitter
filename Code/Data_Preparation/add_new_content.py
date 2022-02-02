import pandas as pd
import os
import re
import pickle
from sklearn.model_selection import KFold

data_directory = "../Data"
data_name = "twitter_fakenews_USElections_2016.xlsx"

df = pd.read_excel(os.path.join(data_directory, data_name)
                   , engine='openpyxl')

data_name = "URL_TEXT_DIC.DIC"

with open(os.path.join(data_directory, data_name), "rb") as f:
    dic_complete = pickle.load(f)

df_true = df[(df["is_fake_news_2"] == 0) & (df["num_urls"] > 0)]
df_fake = df[(df["is_fake_news_2"] == 1)]
df_true = df_true.sample(n=df_fake.shape[0])
#print(df_fake.shape)
#print(df_true.shape)
df = pd.concat([df_true,df_fake])



cols= ["is_fake_news_2", "text"]
df = df[cols]
df = df.rename(columns={"is_fake_news_2": "label"})
df = df.sample(frac=1)

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
            t = t.lower()
        except Exception as s:
            t = ""
        text = text.replace(url, t)
    return text


def preprocess(text):
    text = text.replace("\t"," ")
    text = text.replace("\n", " ")
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    return text

df = df.drop(df[df.label == "UNKNOWN"].index)
df["text"] = df.apply(lambda x: preprocess(x['text']), axis=1)
df["url_remove"] = df.apply(lambda x: replace_url(x['text'], rep=""), axis=1)
df["url_replace_constant"] = df.apply(lambda x: replace_url(x['text'], rep="weblink"), axis=1)
df["url_replace_text"] = df.apply(lambda x: replace_url_with_text(x['text']), axis=1)

data_directory = "../Data/final_data"

print(df["label"].value_counts())

data_name = "final_data.tsv"
df.to_csv(os.path.join(data_directory,data_name), sep="\t")

kf5 = KFold(n_splits=5, shuffle=True)
j = 1
for train_index, test_index in kf5.split(df):
    Train = df.iloc[train_index]
    Test = df.iloc[test_index]
    Train.to_csv(os.path.join(data_directory,str(j),"train.tsv"), sep="\t")
    Test.to_csv(os.path.join(data_directory, str(j), "test.tsv"), sep="\t")
    j += 1
