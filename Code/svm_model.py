import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

data_directory = "../Data/final_data"
data_name = "final_data.tsv"

text_column = "url_replace_text"
Corpus = pd.read_csv(os.path.join(data_directory,data_name), sep="\t")

result_list_f1 = []
result_list_acc = []

for fold in range(1, 6):
    Train = pd.read_csv(os.path.join(data_directory, str(fold),"train.tsv"), sep="\t")
    Test = pd.read_csv(os.path.join(data_directory, str(fold), "test.tsv"), sep="\t")
    Train_X = Train[text_column]
    Test_X = Test[text_column]
    Train_Y = Train["label"]
    Test_Y = Test["label"]
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    #Tfidf_vect.fit(Corpus[text_column])
    Train_X_Tfidf = Tfidf_vect.fit_transform(Train_X)
    #Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    result_list_acc.append(accuracy_score(predictions_SVM, Test_Y) * 100)
    result_list_f1.append(f1_score(predictions_SVM, Test_Y, pos_label=True))

result_list_f1 = np.array(result_list_f1)
result_list_acc = np.array(result_list_acc)
print(f'Results on the {text_column} Column:')
print(result_list_f1.mean(), result_list_f1.std())
print(result_list_acc.mean(), result_list_acc.std())