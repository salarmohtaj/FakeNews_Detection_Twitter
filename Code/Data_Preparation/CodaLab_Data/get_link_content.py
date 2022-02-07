from urllib.parse import urlparse
import pickle
import os
import collections
import tweepy
import requests as RQ
from bs4 import BeautifulSoup
from connection_info import connection_dic
from bs4.element import Comment
import re



def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    text = u" ".join(t.strip() for t in visible_texts)
    text = strip_text(text)
    return text

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def strip_text(text):
    text = text.rstrip().lstrip()
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub('\s+', ' ', text)
    return text

data_directory = "../../../Data/CodaLab_Data"
data_name = "DIC_URL.DIC"

with open(os.path.join(data_directory, data_name), "rb") as f:
    dic = pickle.load(f)

auth = tweepy.OAuthHandler(connection_dic["consumer_key"], connection_dic["consumer_secret"])
auth.set_access_token(connection_dic["access_token"], connection_dic["access_token_secret"])
api = tweepy.API(auth)
twitter_errors = []
total_number = len(dic)
for index, key in enumerate(dic.keys()):
    try:
        net = urlparse(dic[key]["original_url"]).netloc
    except KeyError:
        continue
    if(net == "twitter.com"):
        id = urlparse(dic[key]["original_url"]).path.split("/")[-1]
        try:
            t = api.get_status(id)
            text = t._json["text"]
            #dic[key]["text"] = text
        except Exception as s:
            print(s)
            twitter_errors.append(s)
    elif(net == "www.youtube.com"):
        try:
            text = ""
            r = RQ.get(dic[key]["original_url"])
            soup = BeautifulSoup(r.text, features="html.parser")
            text = soup.title.string
            text = strip_text(text)
        except:
            text = ""
        dic[key]["text"] = text
    else:
        try:
            try:
                text = ""
                r = RQ.get(dic[key]["original_url"])
                text = text_from_html(r.text)
            except:
                text = ""
            if(text == ""):
                try:
                    soup = BeautifulSoup(r.text, features="html.parser")
                    text = soup.title.string
                    text = strip_text(text)
                    # text = soup.title.get_text()
                except:
                    text = ""
            dic[key]["text"] = text
        except:
            print("ERRRRRRRRRRROR")
            pass
    print(f'{index} our of {total_number}')

#print(collections.Counter(domains))
#print(collections.Counter(twitter_errors))
print(list(set(twitter_errors)))
print(len(twitter_errors))
with open(os.path.join(data_directory, "DIC_URL_Final.DIC"), "wb") as f:
    pickle.dump(dic, f)