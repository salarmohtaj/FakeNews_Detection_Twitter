import re
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
import demoji
import spacy
try:
    spacy_en = spacy.load('en_core_web_sm')
except:
    spacy_en = spacy.load('en')

data_directory = "../Data/CodaLab_Data/final_data"

text_column = "text"

def preprocess(text):
    #text = re.sub("@([A-Za-z0-9_]+)", "username", text)
    #text = re.sub(r"http\S+", "link", text)
    text = demoji.replace_with_desc(text, sep=" ")
    text = text[:2000]
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    return text
result_list_f1 = []
result_list_acc = []

for fold in range(1, 6):
    Train = pd.read_csv(os.path.join(data_directory, str(fold),"train.tsv"), sep="\t")
    Test = pd.read_csv(os.path.join(data_directory, str(fold), "test.tsv"), sep="\t")
    Train[text_column] = Train.apply(lambda x: preprocess(x[text_column]), axis=1)
    Test[text_column] = Test.apply(lambda x: preprocess(x[text_column]), axis=1)
    Train_X = Train[text_column]
    Test_X = Test[text_column]
    Train_Y = Train["label"]
    Test_Y = Test["label"]
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    #Tfidf_vect.fit(Corpus[text_column])
    Train_X_Tfidf = Tfidf_vect.fit_transform(Train_X.values)
    #Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X.values)
    #SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM = svm.SVC(C=100.0, kernel='sigmoid', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    result_list_acc.append(accuracy_score(predictions_SVM, Test_Y) * 100)
    result_list_f1.append(f1_score(predictions_SVM, Test_Y, pos_label="fake"))

result_list_f1 = np.array(result_list_f1)
result_list_acc = np.array(result_list_acc)
print(f'Results on the {text_column} Column:')
print(result_list_f1.mean(), result_list_f1.std())
print(result_list_acc.mean(), result_list_acc.std())

