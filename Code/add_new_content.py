import pandas as pd
import os
import re
import pickle

data_directory = "../Data"
data_name = "twitter_fakenews_USElections_2016.xlsx"

df = pd.read_excel(os.path.join(data_directory, data_name)
                   , engine='openpyxl')

data_name = "URL_TEXT_DIC.DIC"

with open(os.path.join(data_directory, data_name), "rb") as f:
    dic_complete = pickle.load(f)

cols= ["is_fake_news_2", "text"]
df = df[cols]


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

df["text"]=df.apply(lambda x: preprocess(x['text']), axis=1)
df["url_remove"]=df.apply(lambda x: replace_url(x['text'], rep=""), axis=1)
df["url_replace_constant"]=df.apply(lambda x: replace_url(x['text'], rep="weblink"), axis=1)
df["url_replace_text"]=df.apply(lambda x: replace_url_with_text(x['text']), axis=1)

data_name = "final_data.tsv"
df.to_csv(os.path.join(data_directory,data_name), sep="\t")
