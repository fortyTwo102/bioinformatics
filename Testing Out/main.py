import os
import pickle
from tkinter import *

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def predict(query):
    # print("query", query)

    data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test_best.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
    

    # the features were learned by the Genetic Algorithm
    query = np.array(query).reshape(1, -1)

    # padding to scale
    # query = np.insert(query, 2, 0)
    # query = np.insert(query, 4, 0)
    # query = np.insert(query, 7, 0)
    # query = np.insert(query, 9, 0)
    # print("query", query)

    scaler = pickle.load(open(os.path.join(data_path, 'scaler-obj.p'), 'rb'))
    query = scaler.transform(query)

    query = query[0]
    # print("query", query)
    # delete the padding

    query = np.delete(query, 2, 0)
    query = np.delete(query, 3, 0)
    query = np.delete(query, 5, 0)
    query = np.delete(query, 6, 0)





    query = query.reshape(1, -1)

    # print("query", query)

    model = LogisticRegression(C=100000, tol=0.1, penalty='l1', solver='liblinear', max_iter=100000, random_state=18318)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_pred, y_test)*100,2)
    # print(acc)

    safe, unsafe = model.predict_proba(query)[0]
    # print(model.classes_)
    return round(unsafe * 100, 2)


data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")
headers = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
                'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl', 'target']
data.columns = headers
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

print(data)
count = 0
for row in range(data.shape[0]):
    unsafe = predict(list(data.iloc[row, :])[:-1])
    if unsafe >= 50 and data.iloc[row,-1] == 2:
        print(data.iloc[row,:])
        print(unsafe)
        count = count + 1
    
print(count)