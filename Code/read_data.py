import pandas as pd
import os

data_directory = "../Data"
data_name = "twitter_fakenews_USElections_2016.xlsx"


df = pd.read_excel(os.path.join(data_directory, data_name)
                   , engine='openpyxl')

print(f'The dataframe has {df.shape[0]} rows and {df.shape[1]} columns')
print(f'There are {df[df.is_fake_news_1 == 1].count()["is_fake_news_1"]} tweets which are labeled as Fake News by the first team of annotators.')
print(f'There are {df[df.is_fake_news_2 == 1].count()["is_fake_news_2"]} tweets which are labeled as Fake News by the second team of annotators.')
print(f'{df[(df.is_fake_news_1 == 1) & (df.is_fake_news_2 == 1)].count()["is_fake_news_2"]} tweets are labeled as Fake News by both teams of annotators.')


print(df.columns)

df1 = df[df.num_urls > 0]
print(df["text"].sample(n=3))