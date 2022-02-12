import pandas as pd
import os
import re
import pickle
import requests as RQ
from urllib.parse import urlparse
import collections


data_directory = "../../../Data/CodaLab_Data"
data_name = "Constraint_English_Train - Sheet1.csv"
df_train = pd.read_csv(os.path.join(data_directory, data_name))

data_name = "Constraint_English_Val - Sheet1.csv"
df_val = pd.read_csv(os.path.join(data_directory, data_name))

data_name = "english_test_with_labels - Sheet1.csv"
df_test = pd.read_csv(os.path.join(data_directory, data_name))


frames = [df_train, df_val, df_test]
df = pd.concat(frames)

# print(df.columns)
# print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

cols= ["label", "tweet"]
df = df[cols]
df = df.rename(columns={"tweet": "text"})
df = df.sample(frac=1)
print(df.columns)
print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

def preprocess(text):
    return text.replace('\0', '')

df["text"] = df.apply(lambda x: preprocess(x['text']), axis=1)

data_name = "merged_dataset.csv"
# df.to_csv(os.path.join(data_directory, data_name), sep = "\t")


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
# #
url_list = []
# # domains = []
# #
for index, row in df.iterrows():
    urls = extract_urls(row.text)
    url_list.extend(urls)
#     # for url in urls:
#     #     try:
#     #         r = RQ.get(url)
#     #         if(r.status_code == 200):
#     #             domains.append(urlparse(r.url).netloc)
#     #     except:
#     #         print("Error")
#
# #
# print(len(url_list))
# print(len(list(set(url_list))))
# # # dic = {}
# # # for index, url in enumerate(url_list):
# # #     if(index % 50 == 0):
# # #         print(index)
# # #     try:
# # #         r = RQ.get(url)
# # #         # print(r.elapsed.total_seconds())
# # #         dic[url] = r.url
# # #         # if(r.status_code == 200):
# # #         #     pass
# # #         #     domains.append(urlparse(r.url).netloc)
# # #     except:
# # #         print("Error")
# # #         print(url)
# # #
# # # # counter = collections.Counter(domains)
# # # # print(counter)
# # # with open("URL_DIC.DIC", "w") as f:
# # #     pickle.dump(dic, f)
# #
# chunks = [url_list[x:x+200] for x in range(0, len(url_list), 200)]
# print(len(chunks))
# data_name = "urls_to_check"
#
# for index, item in enumerate(chunks):
#     with open(os.path.join(data_directory, data_name+str(index+1)+".LIST"), "wb") as f:
#         pickle.dump(item, f)
print(len(url_list))