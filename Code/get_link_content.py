from urllib.parse import urlparse
import pickle
import os
import collections
import tweepy
import requests as RQ
from bs4 import BeautifulSoup

data_directory = "../Data"
data_name = "URL_DIC.DIC"
with open(os.path.join(data_directory, data_name), "rb") as f:
    dic = pickle.load(f)

data_name = "URL_TEXT_DIC.DIC"

try:
    with open(os.path.join(data_directory, data_name), "rb") as f:
        dic_complete = pickle.load(f)
except:
    dic_complete = {}




api = tweepy.API(auth)

domains = []
twitter_errors = []

for index, item in enumerate(dic.keys()):
    net = urlparse(dic[item]).netloc
    domains.append(net)
    dic_complete[item] = {}
    dic_complete[item]["decoded_url"] = dic[item]
    if(net == "twitter.com"):
        id = urlparse(dic[item]).path.split("/")[-1]
        try:
            t = api.get_status(id)
            text = t._json["text"]
            dic_complete[item]["text"] = text
        except Exception as s:
            twitter_errors.append(s)
            print(s)
    else:
        try:
            r = RQ.get(dic[item])
            soup = BeautifulSoup(r.text, features="html.parser")
            text = soup.title.get_text()
            dic_complete[item]["text"] = text
        except:
            pass
    print(index)

print(collections.Counter(domains))
print(collections.Counter(twitter_errors))

with open(os.path.join(data_directory, "URL_TEXT_DIC.DIC"), "wb") as f:
    pickle.dump(dic_complete, f)