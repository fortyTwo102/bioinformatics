import os
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def get_ILPD():

    data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"
    X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
    X_test =  pd.read_csv(os.path.join(data_path, 'X_test_best.csv'))

    y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    y_test =  pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()  

def train_and_score(model, dataset):


    X_train, X_test, y_train, y_test = get_ILPD()
    

    if model['solver'] == 'liblinear':
        LR_model = LogisticRegression(C = model['C'], tol = model['tol'], solver = model['solver'], penalty = model['penalty'], max_iter = 10000, random_state = 13361)

    elif model['solver'] in ['lbfgs', 'newton-cg', 'sag']:
        LR_model = LogisticRegression(C = model['C'], tol = model['tol'], solver = model['solver'], penalty = 'l2', max_iter = 10000, random_state = 13361)
        
    LR_model.fit(X_train, y_train)
    y_pred = LR_model.predict(X_test)

    accuracy = recall_score(y_test, y_pred)

    return accuracy
