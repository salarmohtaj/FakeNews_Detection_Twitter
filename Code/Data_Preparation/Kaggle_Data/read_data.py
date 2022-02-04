import pandas as pd
import os
import re
import pickle
import requests as RQ
from urllib.parse import urlparse
import collections


data_directory = "../../Data"
data_name = "True.csv"
##url = https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csvs

df_true = pd.read_csv(os.path.join(data_directory, data_name))
col = ["text"]
df_true = df_true[col]
df_true['label'] = False
def remove_header(text):
    try:
        text = text.split(") -")[1].lstrip().rstrip()
    except:
        pass
    return text

df_true["text"] = df_true.apply(lambda x: remove_header(x['text']), axis=1)

data_name = "Fake.csv"
df_fake = pd.read_csv(os.path.join(data_directory, data_name))
col = ["text"]
df_fake = df_fake[col]
df_fake['label'] = True
df = pd.concat([df_fake,df_true])
df = df.sample(frac=1)

cols= ["label", "text"]
df = df[cols]

print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

# print(f'There are {df[df.is_fake_news_1 == 1].count()["is_fake_news_1"]} tweets which are labeled as Fake News by the first team of annotators.')
# print(f'There are {df[df.is_fake_news_2 == 1].count()["is_fake_news_2"]} tweets which are labeled as Fake News by the second team of annotators.')
# print(f'{df[(df.is_fake_news_1 == 1) & (df.is_fake_news_2 == 1)].count()["is_fake_news_2"]} tweets are labeled as Fake News by both teams of annotators.')
#
#
#
# df1 = df[df.num_urls > 0]
# #print(df1["text"].sample(n=3))
#
#
# #df1.to_csv(os.path.join(data_directory, "temp.csv"))
#
#
#
#
#
# #myString = "This is my tweet check it out http://example.com/blah http://example.com/blah"
# #print(re.search("(?P<url>https?://[^\s]+)", myString).group("url"))
# #text = "To learn more, please follow us — http://www.sql-datatools.com To Learn more, please visit our YouTube channel at — http://www.youtube.com/c/Sql-datatools To Learn more, please visit our Instagram account at — https://www.instagram.com/asp.mukesh/ To Learn more, please visit our twitter account at — https://twitter.com/macxima To Learn more, please visit our Medium account at — https://medium.com/@macxima"
#
def extract_urls(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    urls = [link[0] for link in links]
    return urls
#
url_list = []
# domains = []
#
for index, row in df.iterrows():
    urls = extract_urls(row.text)
    url_list.extend(urls)
    # for url in urls:
    #     try:
    #         r = RQ.get(url)
    #         if(r.status_code == 200):
    #             domains.append(urlparse(r.url).netloc)
    #     except:
    #         print("Error")
#
#
print(len(url_list))
print(len(list(set(url_list))))
# # dic = {}
# # for index, url in enumerate(url_list):
# #     if(index % 50 == 0):
# #         print(index)
# #     try:
# #         r = RQ.get(url)
# #         # print(r.elapsed.total_seconds())
# #         dic[url] = r.url
# #         # if(r.status_code == 200):
# #         #     pass
# #         #     domains.append(urlparse(r.url).netloc)
# #     except:
# #         print("Error")
# #         print(url)
# #
# # # counter = collections.Counter(domains)
# # # print(counter)
# # with open("URL_DIC.DIC", "w") as f:
# #     pickle.dump(dic, f)
#

chunks = [url_list[x:x+200] for x in range(0, len(url_list), 200)]
print(len(chunks))
data_name = "urls_to_check"

for index, item in enumerate(chunks):
    with open(os.path.join(data_directory, data_name+str(index+1)+".LIST"), "wb") as f:
        pickle.dump(item, f)
data_name = "Fake_True.tsv"
df.to_csv(os.path.join(data_directory, data_name), sep="\t")
